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
OPENSANCTIONS_MATCH_URL = "https://api.opensanctions.org/match/default"

# OpenSanctions match score thresholds (per their docs, ~0.7 is "probable match")
SCORE_HIT = 0.70
SCORE_REVIEW = 0.50


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


def opensanctions_match(
    api_key: str,
    *,
    names: list[str],
    addresses: list[str],
    swift_codes: list[str],
    bik_codes: list[str],
    inn: str | None = None,
    ogrn: str | None = None,
    reg_number: str | None = None,
    kpp_codes: list[str] | None = None,
    cutoff: float = 0.35,
) -> dict[str, Any]:
    """Query OpenSanctions /match/default with semantic identifier properties.

    Why semantic property names matter: OpenSanctions' matcher scores each
    identifier type separately (swiftBic, bikCode, innCode, ogrnCode, kppCode
    are all type `identifier` but with their own matchers). Stuffing the BIC
    into `registrationNumber` works as a fallback but produces weaker scores
    than putting it in the dedicated `bikCode` slot.

    Note: we pass *every* BIC and KPP from the CBR record (head office + all
    branches), since OpenSanctions usually only stores the head-office BIK and
    the user may have entered a branch BIC.
    """
    if not api_key:
        return {"error": "OpenSanctions API key not configured"}

    properties: dict[str, list[str]] = {
        "name": [n for n in dict.fromkeys(names) if n],
        "jurisdiction": ["Russia"],
        "country": ["Russia"],
    }
    if addresses:
        properties["address"] = [a for a in dict.fromkeys(addresses) if a]
    if swift_codes:
        properties["swiftBic"] = swift_codes
    if bik_codes:
        properties["bikCode"] = bik_codes
    if kpp_codes:
        properties["kppCode"] = kpp_codes
    if inn:
        properties["innCode"] = [inn]
    if ogrn:
        properties["ogrnCode"] = [ogrn]
    if reg_number:
        properties["registrationNumber"] = [reg_number]

    body = {
        "queries": {
            "bank": {
                "schema": "Company",
                "properties": properties,
            }
        }
    }

    try:
        r = requests.post(
            OPENSANCTIONS_MATCH_URL,
            json=body,
            params={"algorithm": "best", "cutoff": str(cutoff)},
            headers={
                "Authorization": f"ApiKey {api_key}",
                "Content-Type": "application/json",
            },
            timeout=30,
        )
        r.raise_for_status()
        payload = r.json()
    except requests.HTTPError as e:
        return {
            "error": f"OpenSanctions HTTP {e.response.status_code}",
            "detail": e.response.text[:500] if e.response is not None else "",
        }
    except Exception as e:
        return {"error": f"OpenSanctions request failed: {e}"}

    responses = payload.get("responses", {})
    bank_resp = responses.get("bank", {})
    return {
        "query": bank_resp.get("query"),
        "results": bank_resp.get("results", []),
        "sent_properties": properties,
    }


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


def tokenize_name(s: str | None) -> set[str]:
    """Tokenize a bank name into distinctive tokens (lowercase, 3+ chars, non-stopword)."""
    if not s:
        return set()
    words = re.findall(r"\w+", s.lower())
    return {w for w in words if len(w) >= 3 and w not in NAME_STOPWORDS}


def name_token_overlap(input_names: list[str], candidate_props: dict) -> list[str]:
    """Compute distinctive-token overlap between input and a candidate's names.

    The candidate has many name fields (name, alias, previousName, weakAlias).
    We tokenize them all + the input names, filter stopwords, and intersect.
    A non-empty result means there's a distinctive bank name token they share —
    a strong signal even when fuzzy name scoring is unimpressed.
    """
    candidate_names: list[str] = []
    for key in ("name", "alias", "previousName", "weakAlias"):
        v = candidate_props.get(key)
        if not v:
            continue
        candidate_names.extend(v if isinstance(v, list) else [v])

    input_tokens: set[str] = set()
    for n in input_names:
        input_tokens |= tokenize_name(n)

    candidate_tokens: set[str] = set()
    for n in candidate_names:
        candidate_tokens |= tokenize_name(n)

    return sorted(input_tokens & candidate_tokens)


# Topic codes from OpenSanctions' risk-tagging system.
# https://www.opensanctions.org/docs/topics/
# Sanction topics: entity is on an active sanctions list (any jurisdiction).
SANCTION_TOPICS: set[str] = {
    "sanction", "sanction.linked", "sanction.counter", "sanction.counter.linked",
    "asset.frozen", "wanted",
}
# Risk topics: entity carries elevated compliance risk but is not directly sanctioned.
# Covers PEPs, organised crime, regulatory action, export control, debarment.
RISK_TOPICS: set[str] = {
    "debarment",
    "crime", "crime.fraud", "crime.cyber", "crime.fin", "crime.theft",
    "crime.war", "crime.boss", "crime.terror", "crime.traffick",
    "forced.labor",
    "role.pep", "role.rca", "role.oligarch", "role.spy", "poi",
    "reg.warn", "reg.action",
    "export.control", "export.risk",
}
# Display labels for topic chips (most common ones).
TOPIC_LABELS: dict[str, str] = {
    "sanction": "🔴 Sanctioned",
    "sanction.linked": "🟠 Sanction-linked",
    "sanction.counter": "🟠 Counter-sanctioned",
    "asset.frozen": "🔴 Assets frozen",
    "wanted": "🔴 Wanted",
    "debarment": "🟠 Debarred",
    "crime": "🟠 Crime-linked",
    "crime.fraud": "🟠 Fraud",
    "crime.fin": "🟠 Financial crime",
    "crime.terror": "🔴 Terrorism",
    "crime.war": "🔴 War crimes",
    "role.pep": "🟡 PEP",
    "role.rca": "🟡 PEP close associate",
    "role.oligarch": "🟡 Oligarch",
    "reg.warn": "⚠️ Regulatory warning",
    "reg.action": "⚠️ Regulatory action",
    "export.control": "📦 Export-controlled",
    "fin.bank": "🏦 Bank",
    "corp.public": "🏛️ Publicly listed",
}


def classify_risk(topics: list[str] | None) -> tuple[str, list[str]]:
    """Classify a candidate's risk based on its OpenSanctions topics.

    Returns (class, matched_topics) where class is one of:
        - SANCTIONED: entity has at least one sanction topic
        - RISK_TAGGED: entity has risk topics but no sanction topics
        - NO_TAGS: entity in OS but has no risk indications at all (e.g. just `fin.bank`)
    """
    t = set(topics or [])
    s_match = sorted(t & SANCTION_TOPICS)
    if s_match:
        return "SANCTIONED", s_match
    r_match = sorted(t & RISK_TOPICS)
    if r_match:
        return "RISK_TAGGED", r_match
    return "NO_TAGS", []


def compute_verdict(
    score: float,
    token_overlap: list[str],
    candidate_props: dict | None = None,
) -> tuple[str, str, str]:
    """Verdict combining identity confidence + OS risk classification.

    Identity confidence:
        HIGH: score ≥ SCORE_HIT, OR token_overlap with score ≥ 0.30
        MODERATE: score ≥ SCORE_REVIEW, OR token_overlap alone
        LOW: neither
    Risk class (from candidate's OS topics):
        SANCTIONED / RISK_TAGGED / NO_TAGS

    Combined label hierarchy (worst signal wins):
        🔴 SANCTIONED — identity ≥ MODERATE AND risk = SANCTIONED
        🟠 RISK-TAGGED — identity ≥ MODERATE AND risk = RISK_TAGGED
        🟡 IDENTIFIED — identity = HIGH AND risk = NO_TAGS (Kontur Bank case)
        🟡 POSSIBLE — identity = MODERATE AND risk = NO_TAGS
        🟢 NOT IN OS — identity = LOW

    The 🟡 IDENTIFIED outcome is the important one: we found the entity in OS
    but OS has no risk tags. This does NOT mean the entity is safe — only that
    OS hasn't tagged it. For Russian banks this is common (OS only tracks
    individually-designated entities, not sectoral risk).
    """
    candidate_props = candidate_props or {}
    topics = candidate_props.get("topics") or []
    risk_class, matched = classify_risk(topics)

    if score >= SCORE_HIT or (token_overlap and score >= 0.30):
        identity = "HIGH"
    elif score >= SCORE_REVIEW or token_overlap:
        identity = "MODERATE"
    else:
        identity = "LOW"

    if identity != "LOW" and risk_class == "SANCTIONED":
        return (
            "SANCTIONED",
            "🔴",
            f"matched entity is on a sanctions list (topics: {', '.join(matched)})",
        )
    if identity != "LOW" and risk_class == "RISK_TAGGED":
        return (
            "RISK-TAGGED",
            "🟠",
            f"matched entity carries risk topics: {', '.join(matched)}",
        )
    if identity == "HIGH":
        reason = (
            f"identity confirmed (score {score:.2f}"
            + (f", overlap: {', '.join(token_overlap)}" if token_overlap else "")
            + ") but OpenSanctions has no risk tags — verify against OFAC/EU/UK lists"
        )
        return ("IDENTIFIED", "🟡", reason)
    if identity == "MODERATE":
        return (
            "POSSIBLE MATCH",
            "🟡",
            "possibly the same entity — review evidence and check external sources",
        )
    return ("NOT IN OS", "🟢", f"score {score:.2f}, no overlap — probably not this entity")


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
    "Resolve a Russian BIC to bank details and screen against OpenSanctions. "
    "Sources: CBR SOAP (`BicToIntCode` → `CreditInfoByIntCodeExXML`), bik-info.ru, OpenSanctions `/match`."
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
        "**Verdict legend**  \n"
        "🔴 Sanctioned — OS sanction topic + identity match  \n"
        "🟠 Risk-tagged — OS risk topic (PEP, debarment, crime, regulatory)  \n"
        "🟡 Identified — found in OS but no risk tags (verify externally)  \n"
        "🟡 Possible — moderate identity confidence  \n"
        "🟢 Not in OS — no good identity match"
    )
    st.markdown(
        "---  \n"
        "**Russian banks: always EDD.**  \n"
        "OpenSanctions only tracks individually-designated entities. "
        "All Russian banks carry sectoral, secondary, and evasion-conduit risk."
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
with st.status("Step 3 · OpenSanctions /match — screening", expanded=True) as status:
    # Pre-step: pull the ru_cbr_banks reference entity from OpenSanctions for
    # this BIC. Branches share OGRN/INN with their parent legal entity, so
    # even when CBR's SOAP API can't resolve a branch BIC, OS's own copy of
    # the Russian banking registry gives us the parent identifiers directly.
    enrichment_props: dict[str, list[str]] = {}
    enrichment_source = None
    os_ref = opensanctions_get_entity(OPENSANCTIONS_API_KEY, f"ru-bik-{bic}")
    if os_ref and isinstance(os_ref, dict) and "properties" in os_ref:
        enrichment_props = os_ref.get("properties") or {}
        enrichment_source = f"ru-bik-{bic}"
        st.success(
            f"✓ Found OpenSanctions reference entity `{enrichment_source}` "
            f"({os_ref.get('caption') or 'unnamed'}) — enriching query with its identifiers"
        )
    else:
        st.caption(f"(no `ru-bik-{bic}` reference entity in OpenSanctions — using CBR + bik-info.ru only)")

    # Pull every name variant from all sources, RU + transliterated
    name_pool: set[str] = set()
    name_pool.update([name_ru, short_name_ru, name_en, short_name_en])
    for n in (cbr.get("short_names_cbr") or []) + (cbr.get("full_names_cbr") or []):
        name_pool.add(n)
        name_pool.add(transliterate_ru(n))
    # OS reference entity names — these often include the parent's canonical name
    for n in enrichment_props.get("name") or []:
        name_pool.add(n)
        name_pool.add(transliterate_ru(n))

    # Extract parent bank name from branch-naming patterns
    # ("Филиал X Банка Y" → "Y"). Without it, fuzzy name matching only sees
    # the branch's full noun phrase ("Filial Tsentral'nyj Banka VTB PAO") vs
    # the parent's primary name ("VTB BANK OAO"), which only share one
    # distinctive token after stopword removal.
    parent_names: set[str] = set()
    for source_name in [name_ru, short_name_ru, name_en, short_name_en] + (
        cbr.get("short_names_cbr") or []
    ) + (cbr.get("full_names_cbr") or []):
        for p in extract_parent_bank_names(source_name):
            parent_names.add(p)
            parent_names.add(transliterate_ru(p))
    if parent_names:
        st.caption(
            "📛 Extracted parent bank name(s) from branch name: "
            + ", ".join(f"`{p}`" for p in sorted(parent_names))
        )
        name_pool |= parent_names

    names = [n for n in name_pool if n]

    # Every address variant from all sources, RU + transliterated
    addr_pool: set[str] = set()
    addr_pool.update([address_ru, address_en])
    for a in (cbr.get("addresses_cbr") or []):
        addr_pool.add(a)
        addr_pool.add(transliterate_ru(a))
    for a in enrichment_props.get("address") or []:
        addr_pool.add(a)
    addresses = [a for a in addr_pool if a]

    # Identifiers: prefer existing sources, fall back to OS reference enrichment
    inn_final = inn or cbr.get("inn_cbr") or (enrichment_props.get("innCode") or [None])[0]
    ogrn_final = (
        bi.get("ogrn")
        or cbr.get("ogrn_cbr")
        or (enrichment_props.get("ogrnCode") or [None])[0]
    )
    reg_final = cbr.get("regn_cbr") or (enrichment_props.get("registrationNumber") or [None])[0]

    # Pass ALL BIKs and KPPs (head office + branches + anything OS has on file)
    bik_pool: set[str] = {bic}
    bik_pool.update(cbr.get("bik_codes") or [])
    bik_pool.update(enrichment_props.get("bikCode") or [])
    all_bics = sorted(bik_pool)

    kpp_pool: set[str] = set()
    kpp_pool.update(cbr.get("kpp_codes_cbr") or [])
    if kpp:
        kpp_pool.add(kpp)
    kpp_pool.update(enrichment_props.get("kppCode") or [])
    all_kpps = sorted(kpp_pool)

    # SWIFTs: combine CBR-extracted + any in the OS reference entity
    swift_pool: set[str] = set(cbr.get("swift_codes") or [])
    for s in enrichment_props.get("swiftBic") or []:
        swift_pool.add(s.upper())
        # Also synthesize the alternate length so both forms reach the matcher
        if len(s) == 8:
            swift_pool.add(s.upper() + "XXX")
        elif len(s) == 11:
            swift_pool.add(s[:8].upper())
    all_swifts = sorted(swift_pool)

    os_result = opensanctions_match(
        OPENSANCTIONS_API_KEY,
        names=names,
        addresses=addresses,
        swift_codes=all_swifts,
        bik_codes=all_bics,
        inn=inn_final,
        ogrn=ogrn_final,
        reg_number=reg_final,
        kpp_codes=all_kpps,
    )
    if os_result.get("error"):
        st.error(os_result["error"])
        if os_result.get("detail"):
            st.code(os_result["detail"])
        status.update(label=f"Step 3 · {os_result['error']}", state="error")
        st.stop()

    results = os_result.get("results", [])
    with st.expander("Query sent to OpenSanctions"):
        st.json(os_result.get("sent_properties", {}))
    status.update(label=f"Step 3 · {len(results)} candidate(s) returned", state="complete")

# ─── STEP 4 ──────────────────────────────────────────────────────────────────
st.markdown("## Step 4 · Verdict")

# Jurisdiction risk banner — shown for every Russian BIC, regardless of OS findings.
# This addresses the gap where OpenSanctions only tracks individually-designated
# entities and misses sectoral/secondary risk for unflagged Russian banks.
# Source basis:
#   - OFAC Treasury press release JY2725 (Nov 2024): 50+ Russian banks designated
#     under E.O. 14024 in a single action, with explicit warning that "Sanctioned
#     actors have been known to turn to smaller banks ... in an attempt to evade
#     sanctions as Russia seeks new ways to access the international financial system"
#   - EU 20th sanctions package (April 2026): introduces "anti-circumvention" tool
#     that treats evasion architectures as sanctionable in their own right
#   - SWIFT exclusion of most major Russian banks since 2022
st.warning(
    "🇷🇺 **Russian banking sector — Enhanced Due Diligence required regardless of below findings.**  \n\n"
    "OpenSanctions only tracks **individually-designated** entities. It does NOT cover sectoral risk. "
    "All Russian banks carry elevated compliance risk because:\n"
    "- OFAC has designated 50+ Russian banks under E.O. 14024 since 2022, and explicitly warns that "
    "smaller regional banks are used to evade sanctions  \n"
    "- EU's 20-package sanctions regime now includes anti-circumvention measures (April 2026)  \n"
    "- Russian banks face SWIFT restrictions and secondary-sanctions exposure for non-Russian counterparties  \n"
    "- Cross-border payment infrastructure has fundamentally degraded since 2022\n\n"
    "**A 🟡 IDENTIFIED or 🟢 NOT IN OS verdict below does NOT mean safe.** "
    "It only means OpenSanctions has not individually flagged this bank."
)

# External-source verification links — always available for the analyst
display_name = name_ru or short_name_ru or ""
display_name_en = transliterate_ru(display_name) if display_name else ""
search_handle = display_name_en or display_name or bic
from urllib.parse import quote_plus
q = quote_plus(search_handle)
inn_q = quote_plus(inn_final or "")
st.markdown(
    "**Cross-check against primary sanctions sources:**  \n"
    f"🇺🇸 [OFAC SDN Search](https://sanctionssearch.ofac.treas.gov/) · "
    f"🇪🇺 [EU Sanctions Map](https://www.sanctionsmap.eu/#/main) · "
    f"🇬🇧 [UK Sanctions List](https://www.gov.uk/government/publications/the-uk-sanctions-list) · "
    f"🇨🇭 [Swiss SECO](https://www.seco.admin.ch/seco/en/home/Aussenwirtschaftspolitik_Wirtschaftliche_Zusammenarbeit/Wirtschaftsbeziehungen/exportkontrollen-und-sanktionen/sanktionen-embargos/sanktionsmassnahmen.html) · "
    f"📰 [Google News search](https://www.google.com/search?tbm=nws&q={q}+sanctions) · "
    f"🌐 [DuckDuckGo](https://duckduckgo.com/?q={q}+sanctions+OFAC+EU)  \n"
    f"_Bank name to search:_ `{display_name}` / `{display_name_en}`"
    + (f"  \n_INN:_ `{inn_final}`" if inn_final else "")
)

st.markdown("---")

if not results:
    st.info(
        "🟢 **No matches in OpenSanctions** — no candidate entities returned. "
        "This is consistent with the bank not being individually flagged. "
        "Russian banking jurisdiction risk (see banner above) still applies."
    )
else:
    # Compute per-candidate verdicts using score + token overlap + OS topics
    candidate_verdicts = []
    for r in results:
        score = r.get("score") or 0
        props = r.get("properties", {})
        overlap = name_token_overlap(names, props)
        label, emoji, reason = compute_verdict(score, overlap, props)
        candidate_verdicts.append((r, score, overlap, label, emoji, reason))

    # Overall verdict = worst (most severe) of all candidates
    severity = {
        "SANCTIONED": 5,
        "RISK-TAGGED": 4,
        "IDENTIFIED": 3,
        "POSSIBLE MATCH": 2,
        "NOT IN OS": 1,
    }
    candidate_verdicts.sort(key=lambda t: (severity.get(t[3], 0), t[1]), reverse=True)
    _, top_score, _top_overlap, top_label, top_emoji, top_reason = candidate_verdicts[0]

    if top_label == "SANCTIONED":
        st.error(f"{top_emoji} **{top_label}** — {top_reason}.  Do not transact.")
    elif top_label == "RISK-TAGGED":
        st.error(f"{top_emoji} **{top_label}** — {top_reason}.  Investigate before transacting.")
    elif top_label == "IDENTIFIED":
        st.warning(
            f"{top_emoji} **{top_label}** — {top_reason}.  "
            f"Apply Russian banking EDD per banner above."
        )
    elif top_label == "POSSIBLE MATCH":
        st.warning(f"{top_emoji} **{top_label}** — {top_reason}.")
    else:
        st.info(f"{top_emoji} **{top_label}** — {top_reason}.")

    st.markdown("### Candidates")
    for r, score, overlap, label, emoji, reason in candidate_verdicts:
        props = r.get("properties", {})
        topics = props.get("topics") or []
        topic_chips = " ".join(TOPIC_LABELS.get(t, f"`{t}`") for t in topics) if topics else "_(no risk topics in OS)_"
        title_extra = f" · 🔗 overlap: {', '.join(overlap)}" if overlap else ""
        with st.expander(
            f"{emoji}  {r.get('caption', '(no caption)')}  ·  score {score:.3f}{title_extra}"
        ):
            cols = st.columns([1, 2])
            with cols[0]:
                st.write("**Verdict:**", f"{emoji} {label}")
                st.write("**Reason:**", reason)
                st.write("**ID:**", r.get("id"))
                st.write("**Schema:**", r.get("schema"))
                st.write("**Score:**", f"{score:.4f}")
                st.write("**Match?**", r.get("match"))
                st.markdown(f"**OS topics:** {topic_chips}")
                if overlap:
                    st.write("**Name overlap:**", ", ".join(f"`{t}`" for t in overlap))
                if r.get("datasets"):
                    st.write("**Datasets:**")
                    for d in r["datasets"]:
                        st.write(f"- `{d}`")
            with cols[1]:
                relevant_keys = [
                    "name", "alias", "address", "country", "jurisdiction",
                    "registrationNumber", "swiftBic", "innCode", "ogrnCode",
                    "bikCode", "kppCode", "topics", "program", "sanctions",
                ]
                shown = {k: props[k] for k in relevant_keys if k in props}
                if shown:
                    st.write("**Properties:**")
                    st.json(shown)
                rid = r.get("id")
                if rid:
                    st.markdown(f"[Open in OpenSanctions ↗](https://opensanctions.org/entities/{rid}/)")

with st.expander("🔍 Raw payload (debug)"):
    st.json(
        {
            "cbr": cbr,
            "bik_info": bi,
            "opensanctions": os_result,
        }
    )
