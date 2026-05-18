"""
BIC Bank Screening App
======================
Streamlit app that takes a Russian BIC code, enriches it with bank info from
multiple sources (CBR SOAP, bik-info.ru), and screens the result against
OpenSanctions via the matching API.

Run:
    streamlit run bic_screening_app.py

Secrets:
    Put OpenSanctions API key in .streamlit/secrets.toml as:
        opensanctions_api_key = "your-key-here"
    Or as env var:  OPENSANCTIONS_API_KEY=...
"""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from typing import Any

import requests
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Inline Russian → Latin transliteration (GOST 7.79 / BGN-PCGN flavored).
# Self-contained to keep deployment dependencies minimal — only streamlit and
# requests are external. Matches the output of the `transliterate` library's
# default Russian table (e.g. "ТБанк" → "TBank", "Москва" → "Moskva").
# ─────────────────────────────────────────────────────────────────────────────

_RU_TO_LAT: dict[str, str] = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "jo",
    "ж": "zh", "з": "z", "и": "i", "й": "j", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "h", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sch",
    "ъ": '"', "ы": "y", "ь": "'", "э": "je", "ю": "ju", "я": "ja",
}


def transliterate_ru(text: str | None) -> str:
    """Russian Cyrillic → Latin. Non-Cyrillic chars pass through unchanged."""
    if not text:
        return ""
    out: list[str] = []
    for ch in text:
        repl = _RU_TO_LAT.get(ch.lower())
        if repl is None:
            out.append(ch)
        elif ch.isupper():
            # For multi-char replacements only capitalize the first letter
            # so "Ц" → "Ts" not "TS", consistent with library output.
            out.append(repl[0].upper() + repl[1:])
        else:
            out.append(repl)
    return "".join(out)

# ─────────────────────────────────────────────────────────────────────────────
# Config / secrets
# ─────────────────────────────────────────────────────────────────────────────

CBR_SOAP_URL = "https://www.cbr.ru/CreditInfoWebServ/CreditOrgInfo.asmx"
CBR_NS = {"w": "http://web.cbr.ru/"}
BIK_INFO_URL = "https://bik-info.ru/api.html"


def get_secret(key: str, default: str = "") -> str:
    """Read secret from st.secrets, falling back to env var."""
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError, AttributeError):
        return os.environ.get(key.upper(), default)


OPENSANCTIONS_API_KEY = get_secret("opensanctions_api_key")

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: CBR SOAP — BIC → internal code → SWIFT
# ─────────────────────────────────────────────────────────────────────────────


def _soap_envelope(body_xml: str) -> str:
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<soap:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
        'xmlns:xsd="http://www.w3.org/2001/XMLSchema" '
        'xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">'
        f"<soap:Body>{body_xml}</soap:Body>"
        "</soap:Envelope>"
    )


def _soap_call(action: str, body_xml: str, timeout: int = 30) -> ET.Element:
    resp = requests.post(
        CBR_SOAP_URL,
        data=_soap_envelope(body_xml).encode("utf-8"),
        headers={
            "Content-Type": "text/xml; charset=utf-8",
            "SOAPAction": f'"http://web.cbr.ru/{action}"',
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return ET.fromstring(resp.content)


def _parse_int_result(root: ET.Element, result_tag: str) -> int | None:
    """Extract a numeric SOAP result, treating -1 (CBR 'not found') as None."""
    el = root.find(f".//w:{result_tag}", CBR_NS)
    if el is None or not el.text:
        return None
    try:
        val = int(float(el.text))
    except (TypeError, ValueError):
        return None
    return val if val > 0 else None  # CBR uses 0 / -1 for "not found"


def bic_to_internal_code(bic: str) -> int | None:
    """Call BicToIntCode. Returns None for branch BICs (CBR replies with -1)."""
    body = (
        f'<BicToIntCode xmlns="http://web.cbr.ru/">'
        f"<BicCode>{bic}</BicCode>"
        f"</BicToIntCode>"
    )
    root = _soap_call("BicToIntCode", body)
    return _parse_int_result(root, "BicToIntCodeResult")


def bic_to_reg_number(bic: str) -> int | None:
    """Call BicToRegNumber. Resolves to the *parent* credit org's reg number,
    so it works for branch BICs where BicToIntCode returns -1."""
    body = (
        f'<BicToRegNumber xmlns="http://web.cbr.ru/">'
        f"<BicCode>{bic}</BicCode>"
        f"</BicToRegNumber>"
    )
    root = _soap_call("BicToRegNumber", body)
    return _parse_int_result(root, "BicToRegNumberResult")


def reg_number_to_internal_code(reg_number: int) -> int | None:
    """Call RegNumToIntCode. Final step of the branch-BIC fallback chain."""
    body = (
        f'<RegNumToIntCode xmlns="http://web.cbr.ru/">'
        f"<RegNum>{reg_number}</RegNum>"
        f"</RegNumToIntCode>"
    )
    root = _soap_call("RegNumToIntCode", body)
    return _parse_int_result(root, "RegNumToIntCodeResult")


def resolve_bic_to_internal_code(
    bic: str,
    *,
    regnum_hint: str | int | None = None,
    parent_bic_hint: str | None = None,
) -> tuple[int | None, list[str]]:
    """Multi-track resolution for the internal credit-org code.

    Track A (fast path, head-office BICs): BicToIntCode → done.
    Track B (branch BICs, CBR fallback): BicToRegNumber → RegNumToIntCode.
    Track C (CBR doesn't know the BIC): BicToIntCode(parent_bic_hint).
    Track D (last resort): RegNumToIntCode(regnum_hint).

    Hints come from bik-info.ru, which carries the parent bank's BIK and
    license number for branches. These let us bridge from a branch BIC that
    CBR can't resolve back to the parent credit org's internal code.

    Returns (internal_code or None, trace) where trace is a human-readable
    list of attempted steps for surfacing in the UI.
    """
    trace: list[str] = []

    # Track A
    try:
        ic = bic_to_internal_code(bic)
        if ic is not None:
            trace.append(f"BicToIntCode({bic}) → {ic} ✓")
            return ic, trace
        trace.append(f"BicToIntCode({bic}) → -1 (branch BIC or not a credit org primary)")
    except Exception as e:
        trace.append(f"BicToIntCode({bic}) raised: {e}")

    # Track B
    try:
        rn = bic_to_reg_number(bic)
        if rn is not None:
            trace.append(f"BicToRegNumber({bic}) → reg №{rn}")
            ic = reg_number_to_internal_code(rn)
            if ic is not None:
                trace.append(f"RegNumToIntCode({rn}) → {ic} ✓")
                return ic, trace
            trace.append(f"RegNumToIntCode({rn}) → not found")
        else:
            trace.append(f"BicToRegNumber({bic}) → not found")
    except Exception as e:
        trace.append(f"BicToRegNumber chain raised: {e}")

    # Track C: parent BIK from bik-info.ru
    if parent_bic_hint and parent_bic_hint != bic:
        try:
            ic = bic_to_internal_code(parent_bic_hint)
            if ic is not None:
                trace.append(
                    f"BicToIntCode({parent_bic_hint}) [parent_bik from bik-info.ru] → {ic} ✓"
                )
                return ic, trace
            trace.append(f"BicToIntCode({parent_bic_hint}) [parent_bik] → -1")
        except Exception as e:
            trace.append(f"Parent BIK resolution raised: {e}")

    # Track D: regnum from bik-info.ru
    if regnum_hint:
        try:
            rn = int(str(regnum_hint).strip()) if str(regnum_hint).strip().isdigit() else None
            if rn:
                ic = reg_number_to_internal_code(rn)
                if ic is not None:
                    trace.append(
                        f"RegNumToIntCode({rn}) [regnum from bik-info.ru] → {ic} ✓"
                    )
                    return ic, trace
                trace.append(f"RegNumToIntCode({rn}) [from bik-info.ru] → not found")
            else:
                trace.append(f"regnum hint {regnum_hint!r} is not a positive integer")
        except Exception as e:
            trace.append(f"RegNum hint resolution raised: {e}")

    return None, trace


def internal_code_to_credit_info(int_code: int) -> ET.Element | None:
    """Call CreditInfoByIntCodeExXML and return the embedded CO XML element."""
    body = (
        f'<CreditInfoByIntCodeExXML xmlns="http://web.cbr.ru/">'
        f"<InternalCodes><double>{int_code}</double></InternalCodes>"
        f"</CreditInfoByIntCodeExXML>"
    )
    root = _soap_call("CreditInfoByIntCodeExXML", body)
    result_el = root.find(".//w:CreditInfoByIntCodeExXMLResult", CBR_NS)
    if result_el is None or len(list(result_el)) == 0:
        return None
    return list(result_el)[0]  # the actual <CreditOrgsList> / <CO> doc


def _strip_ns(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _get_attr_ci(el: ET.Element, *names: str) -> str | None:
    """Case-insensitive attribute lookup."""
    wanted = {n.upper() for n in names}
    for k, v in el.attrib.items():
        if k.upper() in wanted and v and v.strip():
            return v.strip()
    return None


def extract_swift_codes(credit_info: ET.Element) -> list[str]:
    """Walk the embedded CBR XML and pull every <SWBIC> across all branches.

    Returns every form we have evidence for: 11-char raw + 8-char institutional
    (drop the branch suffix). OpenSanctions stores SWIFTs in 8-char canonical
    form, so we must include that form for the identifier match to score.
    """
    raw: list[str] = []
    for el in credit_info.iter():
        if _strip_ns(el.tag).upper() == "SWBIC":
            code = _get_attr_ci(el, "SWBIC")
            if not code:
                code = (el.text or "").strip()
            if code:
                raw.append(code.upper())

    expanded: list[str] = []
    for code in raw:
        expanded.append(code)
        if len(code) == 11:
            expanded.append(code[:8])  # institutional form (no branch suffix)
        elif len(code) == 8:
            expanded.append(code + "XXX")
    return list(dict.fromkeys(expanded))


def extract_all_bics(credit_info: ET.Element) -> list[str]:
    """Pull every 9-digit BIC from the CBR response — head office + all branches.

    CBR's CreditInfoByIntCodeExXML returns a full credit-org record with every
    branch nested under it; each branch has its own BIC. OpenSanctions typically
    stores the head-office BIK only, so the branch BIC the user typed may not
    match — we need to surface the head-office BIC too.
    """
    bics: list[str] = []
    for el in credit_info.iter():
        tag = _strip_ns(el.tag).upper()
        # <BIC>044525187</BIC> or <BIC BIC="044525187"/>
        if tag == "BIC":
            val = _get_attr_ci(el, "BIC") or (el.text or "").strip()
            if val.isdigit() and len(val) == 9:
                bics.append(val)
        # MainBIC / BIC attribute on the root <CO> or branch elements
        for attr_name in ("MainBIC", "BIC"):
            val = _get_attr_ci(el, attr_name)
            if val and val.isdigit() and len(val) == 9:
                bics.append(val)
    return list(dict.fromkeys(bics))


def extract_all_values(credit_info: ET.Element, *attr_names: str) -> list[str]:
    """Collect every non-empty value of the named attributes across the tree."""
    seen: list[str] = []
    for el in credit_info.iter():
        for n in attr_names:
            v = el.attrib.get(n)
            if v and v.strip() and v.strip() not in seen:
                seen.append(v.strip())
    return seen


def cbr_lookup(
    bic: str,
    *,
    regnum_hint: str | int | None = None,
    parent_bic_hint: str | None = None,
) -> dict[str, Any]:
    """Full Step-1 chain. Returns {internal_code, swift_codes, names, ...} or {error}.

    Uses the multi-track resolver: tries BicToIntCode first, falls back to
    BicToRegNumber → RegNumToIntCode for branch BICs, and finally uses hints
    from bik-info.ru (parent BIK, parent regnum) when CBR alone fails.
    """
    int_code, trace = resolve_bic_to_internal_code(
        bic, regnum_hint=regnum_hint, parent_bic_hint=parent_bic_hint
    )
    if int_code is None:
        return {
            "error": f"BIC {bic} not resolvable via CBR",
            "resolution_trace": trace,
        }

    try:
        credit_info = internal_code_to_credit_info(int_code)
    except Exception as e:
        return {
            "internal_code": int_code,
            "resolution_trace": trace,
            "error": f"CBR CreditInfoByIntCodeExXML failed: {e}",
        }
    if credit_info is None:
        return {
            "internal_code": int_code,
            "resolution_trace": trace,
            "swift_codes": [],
            "warning": "Empty CO record",
        }

    swifts = extract_swift_codes(credit_info)
    all_bics = extract_all_bics(credit_info)

    # Pull every identifier and name from CBR XML. CBR's attributes vary
    # across nested elements; collect across the whole tree.
    inns = extract_all_values(credit_info, "Ind_INN", "INN")
    ogrns = extract_all_values(credit_info, "OGRN")
    regns = extract_all_values(credit_info, "RegN", "RegNumber")
    kpps = extract_all_values(credit_info, "KPP")
    addresses = extract_all_values(credit_info, "Adr", "Address", "AddrFakt")
    short_names = extract_all_values(credit_info, "ShortName", "NameP")
    full_names = extract_all_values(credit_info, "FullName", "NameMaxP")

    # Serialize the raw CBR XML for the debug expander
    try:
        raw_xml = ET.tostring(credit_info, encoding="unicode")
    except Exception:
        raw_xml = ""

    return {
        "internal_code": int_code,
        "resolution_trace": trace,                      # NEW: how we got int_code
        "bik_codes": all_bics,
        "swift_codes": swifts,
        "primary_swift": swifts[0] if swifts else None,
        "short_names_cbr": short_names,
        "full_names_cbr": full_names,
        "inn_cbr": inns[0] if inns else None,
        "ogrn_cbr": ogrns[0] if ogrns else None,
        "regn_cbr": regns[0] if regns else None,
        "kpp_codes_cbr": kpps,
        "addresses_cbr": addresses,
        "raw_xml": raw_xml,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: bik-info.ru — bank name & address
# ─────────────────────────────────────────────────────────────────────────────


def bik_info_lookup(bic: str) -> dict[str, Any]:
    """Call bik-info.ru JSON API and return the parsed payload."""
    try:
        r = requests.get(
            BIK_INFO_URL,
            params={"type": "json", "bik": bic},
            timeout=20,
            headers={"User-Agent": "Mozilla/5.0 (BIC screening)"},
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return {"error": f"bik-info.ru request failed: {e}"}

    if isinstance(data, dict) and data.get("error"):
        return {"error": data["error"]}
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: OpenSanctions matching API
# ─────────────────────────────────────────────────────────────────────────────


OPENSANCTIONS_ENTITY_URL = "https://api.opensanctions.org/entities/"


def opensanctions_get_entity(api_key: str, entity_id: str) -> dict[str, Any] | None:
    """Fetch a single entity by its OpenSanctions canonical ID.

    Returns the entity dict, or None if 404 / error. Used to pull the
    ru_cbr_banks reference record for a Russian BIK (deterministic ID
    `ru-bik-{BIC}`), which carries OGRN/INN — identifiers that branches
    share with their parent and that the sanctioned entity also stores.
    """
    if not api_key:
        return None
    try:
        r = requests.get(
            OPENSANCTIONS_ENTITY_URL + entity_id,
            headers={"Authorization": f"ApiKey {api_key}"},
            timeout=15,
        )
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


OPENSANCTIONS_SEARCH_URL = "https://api.opensanctions.org/search/"


def opensanctions_search(
    api_key: str,
    query: str,
    *,
    scope: str = "default",
    schema: str | None = "LegalEntity",
    limit: int = 25,
    countries: list[str] | None = None,
    topics: list[str] | None = None,
) -> dict[str, Any]:
    """Full-text search OpenSanctions — same behavior as the website search box.

    The /search endpoint runs an ElasticSearch query over entity names, aliases,
    addresses, identifiers, etc. Returns the same shape of results the website's
    search results page shows, ranked by relevancy.

    Parameters
    ----------
    api_key : str
        OpenSanctions API key.
    query : str
        Text query. Supports ElasticSearch operators (~, AND, OR, NOT, quoted
        phrases, field queries like `properties.swiftBic:VTBRRUMM`).
    scope : str
        Dataset collection. `default` covers everything; `sanctions` filters
        to sanction-list entities only; `peps`, `crime`, etc. are narrower.
    schema : str | None
        Entity type filter. `LegalEntity` covers Company + Organization + Public
        Body, which is what we want for banks. `Company` is narrower. None = all.
    limit : int
        Max results (default 25 — same as the website's first page).
    """
    if not api_key:
        return {"error": "OpenSanctions API key not configured"}

    params: dict[str, Any] = {"q": query, "limit": limit}
    if schema:
        params["schema"] = schema
    if countries:
        params["countries"] = countries
    if topics:
        params["topics"] = topics

    try:
        r = requests.get(
            f"{OPENSANCTIONS_SEARCH_URL}{scope}",
            params=params,
            headers={"Authorization": f"ApiKey {api_key}"},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as e:
        return {
            "error": f"OpenSanctions HTTP {e.response.status_code}",
            "detail": e.response.text[:500] if e.response is not None else "",
        }
    except Exception as e:
        return {"error": f"OpenSanctions request failed: {e}"}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def normalize_bic(raw: str) -> str | None:
    """Strip non-digits, left-pad to 9. Reject if not 9 digits after that."""
    digits = re.sub(r"\D", "", raw or "")
    if not digits:
        return None
    if len(digits) == 8:
        digits = "0" + digits
    if len(digits) != 9:
        return None
    return digits


# Banking/legal-form words that aren't distinctive on their own. Any token
# match against these is meaningless ("Bank" matches every bank).
NAME_STOPWORDS: set[str] = {
    # Russian legal forms / banking
    "банк", "банка", "банке", "банков", "банковский", "банковская",
    "акб", "ткб", "оао", "ооо", "пао", "ао", "зао",
    "акционерное", "общество", "публичное", "открытое", "закрытое",
    "коммерческий", "коммерческая", "филиал", "филиала", "отделение",
    "россии", "россия", "российский", "российская", "российской",
    "центральный", "центральная", "центрального",
    "москва", "москвы", "санкт", "петербург",
    # English / transliterated
    "bank", "banking", "banks", "branch", "office", "foreign", "trade",
    "jsc", "pjsc", "ojsc", "cjsc", "oao", "ooo", "pao", "ao",
    "ltd", "llc", "limited", "company", "co", "corp", "corporation", "group",
    "joint", "stock", "public", "private", "open", "closed",
    "filial", "tsentral", "central", "main", "head",
    "russia", "russian", "moscow", "petersburg", "saint", "st",
    "publichnoe", "obshchestvo", "obschestvo", "aktsionernoe",
    "international", "national", "federal",
    # Articles / common
    "of", "the", "and", "for", "in", "at", "to", "by",
    "и", "или", "и/или", "по",
}


def extract_parent_bank_names(branch_name: str | None) -> list[str]:
    """Pull parent bank name from Russian branch-naming patterns.

    Examples:
        'Филиал "Центральный" Банка ВТБ (ПАО)' → ['ВТБ']
        'ФИЛИАЛ ЦЕНТРАЛЬНЫЙ БАНКА ВТБ (ПАО)' → ['ВТБ']
        'Filial Tsentral\\'nyj Banka VTB (PAO)' → ['VTB']

    The pattern is `Банка X` / `Банк "X"` (Russian) or `Bank[a] X` (transliterated),
    capturing the next 1-2 distinctive tokens that aren't stopwords.
    """
    if not branch_name:
        return []
    out: list[str] = []
    # Russian: Банка X  (genitive — "of Bank X"). Case-insensitive so ALL-CAPS works.
    for m in re.finditer(
        r"\bбанк[ауие]?\s+[\"«'\u201c]?([\u0410-\u042f\u0430-\u044f][\u0410-\u042f\u0430-\u044f\w-]{1,30})[\"»'\u201d]?",
        branch_name,
        flags=re.IGNORECASE,
    ):
        token = m.group(1).strip()
        if token and token.lower() not in NAME_STOPWORDS:
            out.append(token)
    # Russian: Банк "X" (quoted name)
    for m in re.finditer(
        r"\bбанк\s+[\"«'\u201c]([^\"»'\u201d]{2,40})[\"»'\u201d]",
        branch_name,
        flags=re.IGNORECASE,
    ):
        token = m.group(1).strip()
        if token and token.lower() not in NAME_STOPWORDS:
            out.append(token)
    # English/transliterated: Bank[a] X
    for m in re.finditer(
        r"\bbank[ai]?\s+[\"'\u201c]?([A-Za-z][A-Za-z0-9-]{1,30})[\"'\u201d]?",
        branch_name,
        flags=re.IGNORECASE,
    ):
        token = m.group(1).strip()
        if token and token.lower() not in NAME_STOPWORDS:
            out.append(token)
    return list(dict.fromkeys(out))


# Topic codes from OpenSanctions' risk-tagging system. Mapping to display labels
# with emoji prefixes mirrors the website's badge style.
TOPIC_LABELS: dict[str, str] = {
    "sanction": "🔴 Sanctioned",
    "sanction.linked": "🟠 Sanction-linked",
    "sanction.counter": "🟠 Counter-sanctioned",
    "sanction.counter.linked": "🟠 Counter-sanction linked",
    "debarment": "🟠 Debarred",
    "asset.frozen": "🔴 Assets frozen",
    "wanted": "🔴 Wanted",
    "crime": "🟠 Crime-linked",
    "crime.fraud": "🟠 Fraud",
    "crime.cyber": "🟠 Cybercrime",
    "crime.fin": "🟠 Financial crime",
    "crime.theft": "🟠 Theft",
    "crime.war": "🟠 War crimes",
    "crime.boss": "🟠 Organised crime boss",
    "crime.terror": "🔴 Terrorism",
    "crime.traffick": "🟠 Trafficking",
    "forced.labor": "🟠 Forced labor",
    "role.pep": "🟡 PEP (politically exposed)",
    "role.rca": "🟡 PEP close associate",
    "role.judge": "🟡 Judge",
    "role.civil": "🟡 Civil servant",
    "role.diplo": "🟡 Diplomat",
    "role.lead": "🟡 Political leader",
    "role.spy": "🟡 Intelligence",
    "role.oligarch": "🟡 Oligarch",
    "poi": "🟡 Person of interest",
    "fin": "🏦 Financial",
    "fin.bank": "🏦 Bank",
    "fin.fund": "🏦 Investment fund",
    "fin.adivsor": "🏦 Financial advisor",
    "corp.offshore": "🏠 Offshore company",
    "corp.shell": "🏠 Shell company",
    "corp.public": "🏛️ Publicly listed",
    "reg.warn": "⚠️ Regulatory warning",
    "reg.action": "⚠️ Regulatory action",
    "export.control": "📦 Export-controlled",
    "export.risk": "📦 Export risk",
    "gov": "🏛️ Government",
    "gov.national": "🏛️ National government",
    "gov.muni": "🏛️ Municipal",
    "gov.soe": "🏛️ State-owned enterprise",
}

# Topic codes that indicate the entity carries some kind of compliance risk.
# Used to categorize results into 🔴 sanctioned / 🟡 risk / no-tag groups.
SANCTION_TOPICS = {"sanction", "sanction.linked", "sanction.counter", "sanction.counter.linked", "asset.frozen", "wanted"}
RISK_TOPICS = {"debarment", "crime", "crime.fraud", "crime.cyber", "crime.fin", "crime.theft",
               "crime.war", "crime.boss", "crime.terror", "crime.traffick", "forced.labor",
               "role.pep", "role.rca", "role.oligarch", "role.spy", "poi",
               "reg.warn", "reg.action", "export.control", "export.risk"}


def categorize_result(result: dict[str, Any]) -> str:
    """Bucket a search result into 'sanctioned', 'risk', or 'context' by topics."""
    topics = set(result.get("properties", {}).get("topics", []) or [])
    if topics & SANCTION_TOPICS:
        return "sanctioned"
    if topics & RISK_TOPICS:
        return "risk"
    return "context"


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="BIC Bank Screening",
    page_icon="🏦",
    layout="wide",
)

st.title("🏦 BIC Bank Screening")
st.caption(
    "Resolve a Russian BIC to bank details and search OpenSanctions by name "
    "(same behavior as the website search). Sources: CBR SOAP, bik-info.ru, OpenSanctions `/search`."
)

with st.sidebar:
    st.header("Configuration")
    if OPENSANCTIONS_API_KEY:
        st.success("OpenSanctions API key loaded ✓")
    else:
        st.warning(
            "OpenSanctions API key missing. Set `opensanctions_api_key` in "
            "`.streamlit/secrets.toml` or env var `OPENSANCTIONS_API_KEY`."
        )
    st.markdown(
        "**Result categorisation**  \n"
        "🔴 Sanctioned — any sanction topic  \n"
        "🟡 Risk-tagged — PEP, debarment, crime, regulatory, export-control  \n"
        "⚪ Context — no risk tags (informational only)"
    )

col1, col2 = st.columns([3, 1])
with col1:
    bic_raw = st.text_input(
        "BIC (9 digits — leading 0 will be added if missing)",
        placeholder="044525974",
    )
with col2:
    st.write("")
    st.write("")
    run = st.button("Screen bank", type="primary", use_container_width=True)

if not run:
    st.info("Enter a BIC and press **Screen bank**.")
    st.stop()

bic = normalize_bic(bic_raw)
if not bic:
    st.error(f"Invalid BIC: {bic_raw!r}. Expected 8–9 digits.")
    st.stop()

st.markdown(f"### Screening BIC `{bic}`")

# Pre-fetch bik-info.ru silently so its data can be used as fallback hints
# for CBR resolution (parent BIK and parent regnum unlock branch-BIC cases).
with st.spinner("Looking up bank reference data…"):
    bi_pre = bik_info_lookup(bic)

# Extract hints — try multiple field names since bik-info.ru is loose with casing
def _bi_get(*keys: str) -> str | None:
    if not isinstance(bi_pre, dict):
        return None
    for k in keys:
        v = bi_pre.get(k)
        if v and str(v).strip() and str(v).strip().lower() != "null":
            return str(v).strip()
    return None

regnum_hint = _bi_get("regnum", "RegN", "regNumber", "registration_number")
parent_bic_hint = _bi_get("bik_p", "parent_bik", "parentBik", "bikP")
inn_hint = _bi_get("inn", "INN", "innCode", "inn_code")

# ─── STEP 1 ──────────────────────────────────────────────────────────────────
with st.status("Step 1 · CBR: resolve BIC → internal code → SWIFT", expanded=True) as status:
    cbr = cbr_lookup(bic, regnum_hint=regnum_hint, parent_bic_hint=parent_bic_hint)

    # Always show how we tried to resolve the BIC (helps diagnose -1 / branch BICs)
    trace = cbr.get("resolution_trace") or []
    if trace:
        st.markdown("**Resolution trace:**")
        for line in trace:
            st.markdown(f"- {line}")

    if cbr.get("error"):
        st.warning(
            f"⚠️ {cbr['error']}. Continuing with bik-info.ru — the bank can still "
            "be screened by name + INN + BIC even without CBR's full record."
        )
        status.update(label=f"Step 1 · CBR unresolved (continuing)", state="error")
        # Don't stop — fall through to Step 2. cbr dict has empty fields, which
        # the downstream code handles via `cbr.get(...)` everywhere.
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Internal CBR code", cbr.get("internal_code") or "—")
        c2.metric("SWIFTs found", len(cbr.get("swift_codes", [])))
        c3.metric("BICs in record", len(cbr.get("bik_codes", [])))

        if cbr.get("swift_codes"):
            st.write("**SWIFT codes registered:**", ", ".join(f"`{s}`" for s in cbr["swift_codes"]))
        else:
            st.warning(
                "No SWIFT codes registered in CBR for this BIC. "
                "Bank likely operates only domestically (RUB) or routes FX through correspondents."
            )

        if cbr.get("bik_codes"):
            st.write(
                "**All BICs in CBR record (head office + branches):**",
                ", ".join(f"`{b}`" for b in cbr["bik_codes"]),
            )
            if bic in cbr["bik_codes"] and len(cbr["bik_codes"]) > 1:
                st.info(
                    f"`{bic}` is one of {len(cbr['bik_codes'])} BICs registered for this credit organization. "
                    f"OpenSanctions typically keys on the head-office BIC — sending all of them ensures the match lands."
                )

        with st.expander("Raw CBR XML response (debug)"):
            st.code(cbr.get("raw_xml") or "(empty)", language="xml")
        status.update(label="Step 1 · CBR resolved", state="complete")

# ─── STEP 2 ──────────────────────────────────────────────────────────────────
with st.status("Step 2 · bik-info.ru: bank name & address + transliteration", expanded=True) as status:
    bi = bi_pre  # reuse pre-fetched data
    if isinstance(bi, dict) and bi.get("error"):
        st.error(bi["error"])
        status.update(label=f"Step 2 · {bi['error']}", state="error")
        bi = {}

    cbr_short = (cbr.get("short_names_cbr") or [None])[0]
    cbr_full = (cbr.get("full_names_cbr") or [None])[0]
    cbr_addr_primary = (cbr.get("addresses_cbr") or [None])[0]

    name_ru = bi.get("name") or bi.get("namemini") or cbr_short or ""
    short_name_ru = bi.get("namemini") or ""
    address_ru = bi.get("address") or cbr_addr_primary or ""
    # Try multiple case/snake-case variants for INN since bik-info.ru is inconsistent
    inn = inn_hint or bi.get("inn") or bi.get("INN")
    kpp = bi.get("kpp") or bi.get("KPP")
    ks = bi.get("ks")
    phone = bi.get("phone")

    name_en = transliterate_ru(name_ru)
    short_name_en = transliterate_ru(short_name_ru)
    address_en = transliterate_ru(address_ru)

    left, right = st.columns(2)
    with left:
        st.markdown("**🇷🇺 Russian (original)**")
        st.write(f"**Name:** {name_ru or '—'}")
        if short_name_ru and short_name_ru != name_ru:
            st.write(f"**Short:** {short_name_ru}")
        st.write(f"**Address:** {address_ru or '—'}")
    with right:
        st.markdown("**🇬🇧 English (transliterated)**")
        st.write(f"**Name:** {name_en or '—'}")
        if short_name_en and short_name_en != name_en:
            st.write(f"**Short:** {short_name_en}")
        st.write(f"**Address:** {address_en or '—'}")

    with st.expander("Other reference data"):
        st.write(
            {
                "Input BIC": bic,
                "All BICs (CBR)": cbr.get("bik_codes"),
                "INN (bik-info)": inn,
                "INN (CBR)": cbr.get("inn_cbr"),
                "OGRN (bik-info)": bi.get("ogrn"),
                "OGRN (CBR)": cbr.get("ogrn_cbr"),
                "Bank license № (CBR)": cbr.get("regn_cbr"),
                "KPP (bik-info)": kpp,
                "KPPs (CBR, all branches)": cbr.get("kpp_codes_cbr"),
                "Addresses (CBR, all branches)": cbr.get("addresses_cbr"),
                "Correspondent acct (КС)": ks,
                "Phone": phone,
                "CBR internal code": cbr.get("internal_code"),
            }
        )

    with st.expander("Raw bik-info.ru response (debug)"):
        st.json(bi or {"(no response)": None})

    status.update(label="Step 2 · Bank identified", state="complete")

# ─── STEP 3 ──────────────────────────────────────────────────────────────────
with st.status("Step 3 · OpenSanctions /search — search by name", expanded=True) as status:
    # Build candidate search queries. Order matters — most distinctive first.
    # Each query is a (label, query_string) tuple. We'll run them in order
    # until we get useful results, then dedupe by entity ID.
    queries: list[tuple[str, str]] = []

    # 1. Extracted parent bank name (most distinctive — e.g. "ВТБ", "VTB")
    parent_names: list[str] = []
    for source_name in [name_ru, short_name_ru, name_en, short_name_en] + (
        cbr.get("short_names_cbr") or []
    ) + (cbr.get("full_names_cbr") or []):
        for p in extract_parent_bank_names(source_name):
            if p not in parent_names:
                parent_names.append(p)
            tr = transliterate_ru(p)
            if tr and tr != p and tr not in parent_names:
                parent_names.append(tr)
    if parent_names:
        st.caption(
            "📛 Extracted parent bank name(s) from branch name: "
            + ", ".join(f"`{p}`" for p in parent_names)
        )
        # OR-joined query covers both scripts in one call
        joined = " OR ".join(parent_names)
        queries.append(("parent bank name", joined))

    # 2. CBR short name (head-office name, when CBR resolved)
    for n in cbr.get("short_names_cbr") or []:
        if n:
            queries.append(("CBR short name", n))
            break  # one is enough

    # 3. bik-info.ru short name
    if short_name_ru and short_name_ru not in [q[1] for q in queries]:
        queries.append(("bik-info short name", short_name_ru))

    # 4. Full bank name from bik-info.ru (broad fallback)
    if name_ru and name_ru not in [q[1] for q in queries]:
        queries.append(("bik-info full name", name_ru))

    if not queries:
        st.error("No name to search with. CBR and bik-info.ru both returned nothing.")
        status.update(label="Step 3 · nothing to search", state="error")
        st.stop()

    # Let the user override the query
    primary_label, primary_query = queries[0]
    user_query = st.text_input(
        "🔍 Search query (edit to refine, then press Enter to re-search)",
        value=primary_query,
        help="Default = the most distinctive name we extracted. "
        "Supports `OR`, `AND`, `NOT`, and ElasticSearch fuzzy operators like `Aliyev~2`.",
    )

    # Always run with whatever's in the input box (Enter triggers re-run)
    with st.spinner(f"Searching OpenSanctions for `{user_query}`…"):
        search_result = opensanctions_search(OPENSANCTIONS_API_KEY, user_query, limit=25)

    if search_result.get("error"):
        st.error(search_result["error"])
        if search_result.get("detail"):
            st.code(search_result["detail"])
        status.update(label=f"Step 3 · {search_result['error']}", state="error")
        st.stop()

    results = search_result.get("results", []) or []
    total = search_result.get("total")
    if isinstance(total, dict):
        total_count = total.get("value", len(results))
        total_rel = total.get("relation", "eq")
    else:
        total_count = total or len(results)
        total_rel = "eq"
    total_label = f"{total_count}{'+' if total_rel == 'gte' else ''}"

    st.write(
        f"**{total_label}** total hit(s) — showing top **{len(results)}** ranked by relevance."
    )

    with st.expander("🔧 Alternative queries to try"):
        st.caption("Click to copy any of these into the search box above:")
        for label, q in queries:
            st.code(q, language="text")
            st.caption(f"_({label})_")

    status.update(
        label=f"Step 3 · {len(results)} result(s) from /search",
        state="complete",
    )

# ─── STEP 4 ──────────────────────────────────────────────────────────────────
st.markdown("## Step 4 · Results")

if not results:
    st.success(f"🟢 **No matches in OpenSanctions** for query `{user_query}`.")
    st.stop()

# Categorize results by risk topic
sanctioned: list[dict] = []
risky: list[dict] = []
context: list[dict] = []
for r in results:
    bucket = categorize_result(r)
    if bucket == "sanctioned":
        sanctioned.append(r)
    elif bucket == "risk":
        risky.append(r)
    else:
        context.append(r)

# Top-level summary
if sanctioned:
    st.error(
        f"🔴 **{len(sanctioned)} SANCTIONED entity(ies) found** — "
        f"do not transact without compliance review."
    )
elif risky:
    st.warning(
        f"🟡 **{len(risky)} risk-tagged entity(ies) found** — manual review recommended."
    )
else:
    st.info(
        f"ℹ️ **{len(context)} entity(ies) returned**, none carry sanction or risk tags."
    )


def render_entity(r: dict[str, Any]) -> None:
    """Render a single search result, styled like the OpenSanctions website."""
    caption = r.get("caption") or "(no name)"
    eid = r.get("id")
    schema = r.get("schema") or ""
    props = r.get("properties") or {}
    topics = props.get("topics") or []
    datasets = r.get("datasets") or []

    badges = [TOPIC_LABELS.get(t, t) for t in topics]

    # Header with link to OpenSanctions website
    if eid:
        st.markdown(
            f"#### [{caption}](https://www.opensanctions.org/entities/{eid}/)"
        )
    else:
        st.markdown(f"#### {caption}")

    meta_line = f"`{schema}`"
    if badges:
        meta_line += " · " + " · ".join(badges)
    st.caption(meta_line)

    # Aliases — what the website shows under "also known as"
    aliases = (props.get("alias") or [])[:5]
    name_list = props.get("name") or []
    other_names = [n for n in name_list + aliases if n != caption][:5]
    if other_names:
        st.caption("Also known as: " + " · ".join(other_names))

    # Country / jurisdiction
    countries = props.get("country") or props.get("jurisdiction") or []
    if countries:
        st.caption("Jurisdiction: " + ", ".join(countries))

    # Key identifiers in a tidy row
    id_chips = []
    for prop, label in [
        ("swiftBic", "SWIFT"),
        ("bikCode", "BIK"),
        ("innCode", "INN"),
        ("ogrnCode", "OGRN"),
        ("leiCode", "LEI"),
        ("registrationNumber", "Reg №"),
    ]:
        vals = props.get(prop) or []
        if vals:
            id_chips.append(f"**{label}:** " + ", ".join(f"`{v}`" for v in vals[:3]))
    if id_chips:
        st.markdown(" · ".join(id_chips))

    with st.expander(f"Full details · {len(datasets)} dataset(s)"):
        if datasets:
            st.markdown("**Datasets** _(source watchlists this entity appears on)_:")
            for d in datasets:
                st.markdown(f"- `{d}`")
        st.markdown("**All entity properties:**")
        st.json(props)


# Display by risk bucket — most severe first
if sanctioned:
    st.markdown("### 🔴 Sanctioned Entities")
    for r in sanctioned:
        with st.container(border=True):
            render_entity(r)

if risky:
    st.markdown("### 🟡 Risk-tagged Entities")
    for r in risky:
        with st.container(border=True):
            render_entity(r)

if context:
    with st.expander(
        f"⚪ Other matches without risk tags ({len(context)}) — informational only"
    ):
        for r in context:
            with st.container(border=True):
                render_entity(r)

with st.expander("🔍 Raw payload (debug)"):
    st.json(
        {
            "cbr": cbr,
            "bik_info": bi,
            "opensanctions_search": {
                "query": user_query,
                "total": total,
                "n_results": len(results),
                "first_result": results[0] if results else None,
            },
        }
    )
