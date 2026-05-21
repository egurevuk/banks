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

import pandas as pd
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
# Step 1 (assist): iban.ru SWIFT BIC ↔ BIK directory
# ─────────────────────────────────────────────────────────────────────────────

IBAN_RU_SWIFT_URL = "https://www.iban.ru/swift-bic-kodov"

# After stripping HTML tags and pipe separators, SWIFT and BIK appear as adjacent
# whitespace-separated tokens. SWIFT BIC: 4 letters (institution) + 2 letters
# (country) + 2 alphanumeric (location) + optional 3 alphanumeric (branch) — so
# 8 or 11 chars. BIK: 9 digits. Word boundaries on both ends prevent false
# matches against longer runs.
IBAN_RU_PAIR_RE = re.compile(
    r"\b([A-Z]{4}[A-Z]{2}[A-Z0-9]{2}(?:[A-Z0-9]{3})?)\s+(\d{9})\b"
)


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_iban_ru_swift_table() -> dict[str, str]:
    """Fetch & parse iban.ru's BIK ↔ SWIFT mapping table.

    Returns a dict {BIK: SWIFT_code} covering ~300 Russian banks and many of
    their regional branches with branch-level SWIFTs. Per the iban.ru footer,
    the source is S.W.I.F.T. SCRL with permission.

    Why this matters: CBR's SOAP API (Step 1) does include SWIFTs but nested
    deep in the credit-org XML and not always reliable for branches. This
    table gives us a clean per-BIC SWIFT lookup with branch granularity.

    Parser strategy: iban.ru serves raw HTML (`<td>SWIFT</td><td>BIK</td>...`).
    We strip HTML tags and pipe characters first, normalize whitespace, then
    use a single regex to find SWIFT immediately followed by BIK — same code
    works whether the response comes back as HTML, markdown, or plain text.

    Cached for 24 hours in Streamlit so each screening doesn't hit iban.ru.
    """
    try:
        r = requests.get(
            IBAN_RU_SWIFT_URL,
            timeout=20,
            headers={"User-Agent": "Mozilla/5.0 BIC-screening-tool"},
        )
        r.raise_for_status()
    except Exception:
        return {}
    # Strip HTML tags → space, then pipes → space, then collapse whitespace.
    # Result is a single line of space-separated tokens where SWIFT and BIK
    # are immediately adjacent in row order.
    text = re.sub(r"<[^>]+>", " ", r.text)
    text = re.sub(r"\|", " ", text)
    text = re.sub(r"\s+", " ", text)
    table: dict[str, str] = {}
    for m in IBAN_RU_PAIR_RE.finditer(text):
        swift = m.group(1).upper()
        bik = m.group(2)
        # First occurrence wins (head office is usually listed before branches
        # for the same BIK, though duplicates are rare since BIK is unique).
        if bik not in table:
            table[bik] = swift
    return table


def iban_ru_swift_for_bic(bic: str) -> str | None:
    """Look up a single BIC in the iban.ru table. Returns 11-char SWIFT or None."""
    return fetch_iban_ru_swift_table().get(bic)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 (assist): Dadata findById/bank — authoritative INN/KPP/OGRN/state
# ─────────────────────────────────────────────────────────────────────────────

DADATA_FIND_BANK_URL = (
    "https://suggestions.dadata.ru/suggestions/api/4_1/rs/findById/bank"
)


@st.cache_data(ttl=3600, show_spinner=False)
def dadata_find_bank(bic: str, api_key: str) -> dict[str, Any]:
    """Look up a Russian bank by BIK using Dadata's findById/bank API.

    Dadata is the authoritative commercial source for Russian legal-entity
    reference data (it ingests CBR, FNS, and EGRUL feeds). For our pipeline
    its main value is providing a reliable INN per BIK — CBR's SOAP sometimes
    omits INN for branch BICs and bik-info.ru's INN field is inconsistent.

    API contract: POST a JSON body ``{"query": "<BIC>"}`` with header
    ``Authorization: Token <KEY>``. The response is
    ``{"suggestions": [{"value": "...", "data": {...}}]}`` where ``data``
    carries inn, kpp, ogrn, swift, registration_number, name (short/full/payment),
    address, correspondent_account, state (status/actuality_date), etc.

    Returns a flattened dict with the fields we care about, or
    ``{"error": "..."}`` on auth/network/empty failures.

    Cached for 1 hour — short enough to pick up bank-state changes (license
    revocation matters) but long enough to keep cost low during burst screening.
    """
    if not api_key:
        return {"error": "no_api_key"}
    try:
        r = requests.post(
            DADATA_FIND_BANK_URL,
            json={"query": bic},
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Token {api_key}",
            },
            timeout=10,
        )
        if r.status_code in (401, 403):
            return {"error": f"auth_failed_{r.status_code}"}
        r.raise_for_status()
        payload = r.json()
    except Exception as e:
        return {"error": f"request_failed: {e}"}

    suggestions = payload.get("suggestions") or []
    if not suggestions:
        return {"error": "no_suggestions"}
    first = suggestions[0]
    d = first.get("data") or {}
    name = d.get("name") if isinstance(d.get("name"), dict) else {}
    address = d.get("address") if isinstance(d.get("address"), dict) else {}
    state = d.get("state") if isinstance(d.get("state"), dict) else {}
    return {
        "value": first.get("value"),
        "inn": d.get("inn"),
        "kpp": d.get("kpp"),
        "ogrn": d.get("ogrn"),
        "swift": d.get("swift"),
        "okpo": d.get("okpo"),
        "reg_number": d.get("registration_number"),
        "correspondent_account": d.get("correspondent_account"),
        "name_short": name.get("short") if isinstance(name, dict) else None,
        "name_full": name.get("full") if isinstance(name, dict) else None,
        "name_payment": name.get("payment") if isinstance(name, dict) else None,
        "name_english": name.get("english") if isinstance(name, dict) else None,
        "address": address.get("value") if isinstance(address, dict) else None,
        "state_status": state.get("status") if isinstance(state, dict) else None,
        "state_actuality_date": state.get("actuality_date") if isinstance(state, dict) else None,
        "raw_data": d,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 (assist): OhMySwift whitelist of non-sanctioned Russian banks
# ─────────────────────────────────────────────────────────────────────────────

# File name (kept in the repo root alongside the script). The xlsx is a
# curated list — col[0] is the 8-char SWIFT BIC, col[1] is the Russian bank
# name, col[2] is the English name. Header/navigation rows are at the top and
# we skip them by filtering for the SWIFT regex on col[0].
WHITELIST_XLSX_FILENAME = "Ohmyswift.xlsx"

# Strict 8-char SWIFT pattern: 4 letters (institution) + 2 letters (country)
# + 2 alphanumeric (location). The whitelist file uses 8-char form throughout.
SWIFT_8_RE = re.compile(r"^[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}$")


@st.cache_data(ttl=86400, show_spinner=False)
def load_whitelist_swifts() -> dict[str, str]:
    """Load the curated 'Russian banks NOT under US/EU sanctions' SWIFT list.

    Returns a dict {SWIFT_8char: bank_display_name}. The file is the iban.ru-
    sourced 'Российские банки не под санкциями США и ЕС' compilation — Russian
    banks confirmed absent from both the US SDN list and EU sanctions lists.

    File expectations:
      - Sheet 0, no header (we read header=None and filter rows by content)
      - Column 0: 8-char SWIFT BIC (skips rows where col[0] doesn't match)
      - Column 1: Russian bank name (used if English unavailable)
      - Column 2: English bank name (preferred for display)

    Lookup is attempted from multiple paths so the file can live alongside
    the script (production) or in the upload folder (dev/Claude environment).

    Cached for 24h — the whitelist itself rarely changes.
    """
    candidate_paths = [
        WHITELIST_XLSX_FILENAME,                                   # cwd
        os.path.join(os.path.dirname(__file__), WHITELIST_XLSX_FILENAME),
        os.path.join("/mnt/user-data/uploads", WHITELIST_XLSX_FILENAME),
    ]
    df = None
    for path in candidate_paths:
        if os.path.exists(path):
            try:
                df = pd.read_excel(path, sheet_name=0, dtype=str, header=None)
                break
            except Exception:
                continue
    if df is None or df.empty:
        return {}

    out: dict[str, str] = {}
    for _, row in df.iterrows():
        cell = row.iloc[0] if len(row) > 0 else None
        if not isinstance(cell, str):
            continue
        swift = cell.strip().upper()
        if not SWIFT_8_RE.match(swift):
            continue  # header/navigation rows fail this filter
        name_en = row.iloc[2] if len(row) > 2 and isinstance(row.iloc[2], str) else ""
        name_ru = row.iloc[1] if len(row) > 1 and isinstance(row.iloc[1], str) else ""
        out[swift] = (name_en or name_ru or "").strip()
    return out


def check_whitelist_matches(swifts: list[str] | set[str]) -> list[tuple[str, str]]:
    """Return [(swift_8char, bank_name)] for any of the bank's SWIFTs that
    appear on the OhMySwift whitelist. 11-char inputs are normalized to 8-char
    so the iban.ru-merged 11-char SWIFTs (e.g. ``VTBRRUMMXXX``) match the
    whitelist's 8-char entries (e.g. ``VTBRRUMM``) — though VTB itself is
    sanctioned and won't be on the list, this normalization is needed for
    every comparison."""
    whitelist = load_whitelist_swifts()
    if not whitelist or not swifts:
        return []
    matches: list[tuple[str, str]] = []
    seen: set[str] = set()
    for s in swifts:
        if not isinstance(s, str):
            continue
        s8 = s.strip().upper()[:8]
        if len(s8) != 8:
            continue
        if s8 in whitelist and s8 not in seen:
            matches.append((s8, whitelist[s8]))
            seen.add(s8)
    return matches


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: OpenSanctions matching API
# ─────────────────────────────────────────────────────────────────────────────


OPENSANCTIONS_ENTITY_URL = "https://api.opensanctions.org/entities/"
OPENSANCTIONS_SEARCH_URL = "https://api.opensanctions.org/search/default"

# US OFAC dataset identifiers in OpenSanctions. The OFAC distinction matters
# for the verdict logic: a bank on OFAC's SDN list cannot transact in USD
# through any US correspondent bank, while a bank on EU/UK/CH/JP sanctions
# alone may still be reachable for non-USD flows. So a strict identifier hit
# carrying OFAC coverage is 🔴 MATCH (block), while a hit without OFAC is
# 🟡 REVIEW (manual judgement required). Other OFAC sub-lists like
# `us_ofac_ns_pp` (Palestinian Legislative Council, sectoral) are not
# included here since they don't apply to banks; if they should later, just
# extend this set.
OFAC_DATASETS: set[str] = {"us_ofac_sdn", "us_ofac_cons"}


def entity_has_ofac(entity: dict[str, Any]) -> bool:
    """True if the entity is listed in any OFAC dataset."""
    return bool(set(entity.get("datasets") or []) & OFAC_DATASETS)


# Keyword-to-head-office-BIK mapping. If the bank name contains one of these
# keywords (case-insensitive), the BIK pool is extended with the head-office
# BIK(s) of that bank. This rescues branch-BIK screenings: regional branches
# of Sberbank/VTB have BIKs that OpenSanctions doesn't typically index,
# whereas the head-office BIKs do, and they share the sanctions designation.
# Applied in both the interactive Step 3 (with merged name variants from CBR)
# and the bulk screener (with the single name from base.xml).
SPECIAL_BANK_HEAD_OFFICE_BICS: dict[str, list[str]] = {
    "SBERBANK": ["044525225"],
    "СБЕРБАНК": ["044525225"],
    "VTB":      ["044030707", "044525187"],
    "ВТБ":      ["044030707", "044525187"],
}


def head_office_bics_for_name(name: str | None) -> list[str]:
    """Return any head-office BIKs implied by keywords in the bank name.

    Used to expand the BIK pool before strict OpenSanctions search so that
    branch-BIK lookups still find the parent sanctioned entity.
    """
    if not name:
        return []
    h = name.upper()
    out: list[str] = []
    seen: set[str] = set()
    for keyword, head_office_bics in SPECIAL_BANK_HEAD_OFFICE_BICS.items():
        if keyword in h:
            for b in head_office_bics:
                if b not in seen:
                    seen.add(b)
                    out.append(b)
    return out


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


def opensanctions_search_by_property(
    api_key: str,
    prop_name: str,
    value: str,
) -> dict[str, Any]:
    """Strict-equality search OpenSanctions by an indexed property value.

    Uses the ``/search/default`` endpoint with a field-targeted query in the
    form ``q=properties.<prop_name>:<value>``. This is OpenSanctions'
    Elasticsearch-backed field query (see https://www.opensanctions.org/docs/api/search/);
    it performs exact-equality matching on the indexed property value.

    The ``filter:<prop>`` syntax that exists on the ``/match`` endpoint does
    NOT exist on ``/search`` — passing it there is silently ignored and the
    search degrades to returning the highest-ranked sanctioned entities in
    the database (which produces egregious false positives like Revival of
    Islamic Heritage Society dominating any unfiltered Russian-bank screening).
    Use this helper everywhere — never construct ``filter:`` URLs for ``/search``.

    ``schema=LegalEntity`` is hierarchical — covers Company, Organization,
    PublicBody, Asset, and any other LegalEntity subschema.

    Used in Step 4's strict-identifier verdict logic for:
      - ``bikCode``: 9-digit Russian Bank Identification Code
      - ``swiftBic``: 8- or 11-char SWIFT BIC
      - ``innCode``: 10- or 12-digit Russian tax ID

    Returns ``{"results": [...], "total": N, "error": None, "url": "..."}``.
    """
    if not api_key or not value:
        return {"results": [], "total": 0, "error": None}
    value_clean = str(value).strip().upper()
    if not value_clean:
        return {"results": [], "total": 0, "error": "empty_value"}
    # Build the URL manually so the colon in `properties.X:` is NOT
    # percent-encoded by requests. Values are alphanumeric so no escaping needed.
    url = (
        f"{OPENSANCTIONS_SEARCH_URL}"
        f"?q=properties.{prop_name}:{value_clean}"
        f"&schema=LegalEntity"
        f"&limit=20"
    )
    try:
        r = requests.get(
            url,
            headers={"Authorization": f"ApiKey {api_key}"},
            timeout=20,
        )
        r.raise_for_status()
        payload = r.json()
    except requests.HTTPError as e:
        return {
            "results": [],
            "total": 0,
            "error": f"HTTP {e.response.status_code}",
            "detail": e.response.text[:500] if e.response is not None else "",
            "url": url,
        }
    except Exception as e:
        return {"results": [], "total": 0, "error": f"request failed: {e}", "url": url}
    return {
        "results": payload.get("results", []),
        "total": (payload.get("total") or {}).get("value", 0),
        "error": None,
        "url": url,
    }


def opensanctions_strict_screening(
    api_key: str,
    *,
    bics: list[str] | set[str],
    swifts: list[str] | set[str],
    inn: str | None,
) -> dict[str, Any]:
    """Run the full strict-identifier screening for a bank.

    Three queries — one per identifier type — to find any OpenSanctions entity
    whose indexed property equals one of our resolved identifiers. Each hit is
    near-certain to be the same legal entity since these identifiers are
    globally unique (BIK assigned by CBR, INN by FNS, SWIFT BIC by SCRL).

    Validation:
      - BIKs must be exactly 9 digits (Russian routing code shape)
      - SWIFTs must be 8 or 11 chars and start with 4 letters + 2 letters
        (institution + country); we normalize to 8-char form for the query
        since OpenSanctions canonically stores 8-char form
      - INN must be exactly 10 or 12 digits (legal entity or individual);
        anything else is rejected as it cannot be a real INN

    Returns:
        ``{
            "hits_by_entity_id": {entity_id: {"entity": dict, "matches": [(prop, value), ...]}},
            "searches": [{"prop": str, "value": str, "url": str, "total": int, "error": str|None}],
        }``
    """
    hits_by_entity_id: dict[str, dict[str, Any]] = {}
    searches: list[dict[str, Any]] = []

    def _record(prop: str, value: str, result: dict) -> None:
        searches.append({
            "prop": prop,
            "value": value,
            "url": result.get("url"),
            "total": result.get("total", 0),
            "error": result.get("error"),
            "n_hits": len(result.get("results", [])),
        })
        for entity in result.get("results", []):
            eid = entity.get("id")
            if not eid:
                continue
            if eid not in hits_by_entity_id:
                hits_by_entity_id[eid] = {"entity": entity, "matches": []}
            hits_by_entity_id[eid]["matches"].append((prop, value))

    # 1. BIK strict search — one query per unique 9-digit BIK
    seen_biks: set[str] = set()
    for bik in bics or []:
        clean = "".join(c for c in str(bik) if c.isdigit())
        if len(clean) != 9 or clean in seen_biks:
            continue
        seen_biks.add(clean)
        _record("bikCode", clean, opensanctions_search_by_property(api_key, "bikCode", clean))

    # 2. SWIFT strict search — normalize to 8-char canonical form, dedupe
    seen_swifts: set[str] = set()
    for s in swifts or []:
        s8 = str(s).strip().upper()[:8]
        if len(s8) != 8 or s8 in seen_swifts:
            continue
        # Validate SWIFT BIC shape: 4 letters + 2 letters + 2 alphanumeric
        if not re.match(r"^[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}$", s8):
            continue
        seen_swifts.add(s8)
        _record("swiftBic", s8, opensanctions_search_by_property(api_key, "swiftBic", s8))

    # 3. INN strict search — validate 10 or 12 digits
    if inn:
        clean_inn = "".join(c for c in str(inn) if c.isdigit())
        if len(clean_inn) in (10, 12):
            _record("innCode", clean_inn, opensanctions_search_by_property(api_key, "innCode", clean_inn))
        else:
            searches.append({
                "prop": "innCode",
                "value": clean_inn or str(inn),
                "url": None,
                "total": 0,
                "error": f"invalid_inn_shape (got {len(clean_inn)} digits, expected 10 or 12)",
                "n_hits": 0,
            })

    return {"hits_by_entity_id": hits_by_entity_id, "searches": searches}


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


def compute_verdict(score: float, token_overlap: list[str]) -> tuple[str, str, str]:
    """(label, emoji, reason) for a single candidate. Score and token overlap
    are independent signals — token overlap can escalate verdict on its own."""
    if score >= SCORE_HIT:
        return ("MATCH", "🔴", f"score {score:.2f} ≥ {SCORE_HIT:.2f}")
    if score >= SCORE_REVIEW:
        return ("REVIEW", "🟡", f"score {score:.2f} ≥ {SCORE_REVIEW:.2f}")
    if token_overlap:
        return (
            "REVIEW",
            "🟡",
            f"name token overlap: {', '.join(token_overlap)} (score {score:.2f} alone is below review)",
        )
    return ("LIKELY CLEAR", "🟢", f"score {score:.2f}, no name overlap")


# ─────────────────────────────────────────────────────────────────────────────
# Sanctioning-jurisdiction mapping
# ─────────────────────────────────────────────────────────────────────────────
# Maps OpenSanctions dataset codes to (flag, country/issuer, list label, category).
# Categories drive how each finding is presented:
#   "sanctions"         — government sanctions list (the headline finding)
#   "export_control"    — export control / military end-user list
#   "counter_sanctions" — sanctions imposed BY this country (e.g. Russia, China)
#                         against foreign targets — usually inverted risk meaning
#   "debarment"         — barred from public procurement
#   "regulatory"        — regulatory warning / enforcement action
#   "crime"             — wanted lists / criminal databases
#   "pep"               — politically exposed persons
#   "reference"         — registries, FATCA, KYB (not a risk indicator)
#   "leak"              — leaked records (Panama Papers etc.)
#
# Curated for the major sanctioning jurisdictions. Unmapped codes fall through
# to country-prefix detection via COUNTRY_FLAGS/COUNTRY_NAMES.
DATASET_INFO: dict[str, tuple[str, str, str, str]] = {
    # United States
    "us_ofac_sdn":         ("🇺🇸", "United States", "OFAC SDN List", "sanctions"),
    "us_ofac_cons":        ("🇺🇸", "United States", "OFAC Consolidated (non-SDN)", "sanctions"),
    "us_trade_csl":        ("🇺🇸", "United States", "Consolidated Screening List", "sanctions"),
    "us_state_terrorism":  ("🇺🇸", "United States", "State Dept Terrorism Designations", "sanctions"),
    "us_bis_denied":       ("🇺🇸", "United States", "BIS Denied Persons", "export_control"),
    "us_bis_entity":       ("🇺🇸", "United States", "BIS Entity List", "export_control"),
    "us_bis_meu":          ("🇺🇸", "United States", "BIS Military End User", "export_control"),
    "us_sam_exclusions":   ("🇺🇸", "United States", "SAM Exclusions", "debarment"),
    "us_fbi_most_wanted":  ("🇺🇸", "United States", "FBI Most Wanted", "crime"),
    # European Union
    "eu_fsf":              ("🇪🇺", "European Union", "Financial Sanctions Files (FSF)", "sanctions"),
    "eu_sanctions_map":    ("🇪🇺", "European Union", "EU Sanctions Map", "sanctions"),
    "eu_travel_bans":      ("🇪🇺", "European Union", "Travel Bans", "sanctions"),
    "eu_journal_sanctions":("🇪🇺", "European Union", "Council Official Journal", "sanctions"),
    "eu_esma_sanctions":   ("🇪🇺", "European Union", "ESMA Sanctions", "regulatory"),
    "eu_esma_saris":       ("🇪🇺", "European Union", "ESMA Suspensions/Removals", "regulatory"),
    "eu_edes":             ("🇪🇺", "European Union", "EDES (Early Detection/Exclusion)", "debarment"),
    "eu_europol_wanted":   ("🇪🇺", "European Union", "Europol Most Wanted", "crime"),
    # United Kingdom
    "gb_hmt_sanctions":    ("🇬🇧", "United Kingdom", "HMT Consolidated List", "sanctions"),
    "gb_fcdo_sanctions":   ("🇬🇧", "United Kingdom", "FCDO Sanctions", "sanctions"),
    # Switzerland
    "ch_seco_sanctions":   ("🇨🇭", "Switzerland", "SECO Sanctions", "sanctions"),
    # Canada
    "ca_dfatd_sema_sanctions": ("🇨🇦", "Canada", "SEMA Consolidated Sanctions", "sanctions"),
    "ca_facfoa":           ("🇨🇦", "Canada", "FACFOA (Corrupt Officials)", "sanctions"),
    "ca_listed_terrorists":("🇨🇦", "Canada", "Listed Terrorist Entities", "sanctions"),
    "ca_named_research_orgs":("🇨🇦", "Canada", "Research Orgs of Concern", "export_control"),
    # Australia
    "au_dfat_sanctions":   ("🇦🇺", "Australia", "DFAT Consolidated", "sanctions"),
    "au_listed_terrorist_orgs": ("🇦🇺", "Australia", "Listed Terrorist Orgs", "sanctions"),
    "au_abf_sanctioned_sponsors":("🇦🇺", "Australia", "ABF Sanctioned Sponsors", "sanctions"),
    # Japan
    "jp_mof_sanctions":    ("🇯🇵", "Japan", "MOF Economic Sanctions", "sanctions"),
    "jp_meti_eul":         ("🇯🇵", "Japan", "METI End User List", "export_control"),
    "jp_meti_ru":          ("🇯🇵", "Japan", "METI Russia List", "sanctions"),
    # Ukraine
    "ua_nsdc_sanctions":   ("🇺🇦", "Ukraine", "NSDC Sanctions", "sanctions"),
    "ua_sfms_blacklist":   ("🇺🇦", "Ukraine", "SFMS Blacklist", "sanctions"),
    "ua_nabc_sanctions":   ("🇺🇦", "Ukraine", "NABC War & Sanctions", "sanctions"),
    "ua_war_sanctions":    ("🇺🇦", "Ukraine", "War Sanctions", "sanctions"),
    # New Zealand
    "nz_russia_sanctions": ("🇳🇿", "New Zealand", "Russia Sanctions", "sanctions"),
    "nz_un_sanctions":     ("🇳🇿", "New Zealand", "UN-implemented Sanctions", "sanctions"),
    # France
    "fr_tresor_gels_avoir":("🇫🇷", "France", "Trésor National Asset Freezing", "sanctions"),
    "fr_amf_regulatory_sanctions":("🇫🇷", "France", "AMF Regulatory Sanctions", "regulatory"),
    "fr_illegal_financial_services":("🇫🇷", "France", "AMF Illegal Financial Services", "regulatory"),
    # Other EU member states
    "be_fod_sanctions":    ("🇧🇪", "Belgium", "FOD Financial Sanctions", "sanctions"),
    "at_nbter_sanctions":  ("🇦🇹", "Austria", "OeNB Terrorism Restrictions", "sanctions"),
    "ee_international_sanctions":("🇪🇪", "Estonia", "International Sanctions Act", "sanctions"),
    "cz_national_sanctions":("🇨🇿", "Czechia", "National Sanctions", "sanctions"),
    "cz_terrorists":       ("🇨🇿", "Czechia", "Anti-Terrorism Designations", "sanctions"),
    "de_bka_wanted":       ("🇩🇪", "Germany", "BKA Wanted Fugitives", "crime"),
    # Middle East / Asia
    "il_wmd_sanctions":    ("🇮🇱", "Israel", "WMD Sanctions", "sanctions"),
    "il_mod_terrorists":   ("🇮🇱", "Israel", "Terrorists Organizations", "sanctions"),
    "il_mod_crypto":       ("🇮🇱", "Israel", "Sanctioned Crypto Wallets", "sanctions"),
    "ir_sanctions":        ("🇮🇷", "Iran", "Iran Sanctions List", "sanctions"),
    "iq_aml_list":         ("🇮🇶", "Iraq", "Terrorist Fund Freezing", "sanctions"),
    "in_mha_banned":       ("🇮🇳", "India", "MHA Banned Organizations", "sanctions"),
    "in_nse_debarred":     ("🇮🇳", "India", "NSE Debarred Entities", "debarment"),
    "id_dttot":            ("🇮🇩", "Indonesia", "Suspected Terrorists", "sanctions"),
    "az_fiu_sanctions":    ("🇦🇿", "Azerbaijan", "Domestic Sanctions List", "sanctions"),
    "eg_terrorists":       ("🇪🇬", "Egypt", "Domestic Terrorist List", "sanctions"),
    "ar_repet":            ("🇦🇷", "Argentina", "RePET Sanctions", "sanctions"),
    "br_ceis":             ("🇧🇷", "Brazil", "CEIS Disreputable Companies", "debarment"),
    "br_tcu_debarred":     ("🇧🇷", "Brazil", "TCU Debarred Bidders", "debarment"),
    "br_slavery":          ("🇧🇷", "Brazil", "Slavery Prevention List", "regulatory"),
    # Counter-sanctions — inverted risk meaning
    "ru_treasury_sanctions":("🇷🇺", "Russia (counter)", "Treasury counter-sanctions", "counter_sanctions"),
    "cn_sanctions":        ("🇨🇳", "China (counter)", "Counter-sanctions research", "counter_sanctions"),
    # International / development banks
    "un_sc_sanctions":     ("🇺🇳", "United Nations", "Security Council Sanctions", "sanctions"),
    "interpol_red_notices":("🚨", "INTERPOL", "Red Notices", "crime"),
    "afdb_sanctions":      ("🏦", "African Dev. Bank", "Debarred Entities", "debarment"),
    "adb_sanctions":       ("🏦", "Asian Dev. Bank", "Sanctions", "debarment"),
    "ebrd_ineligible":     ("🏦", "EBRD", "Ineligible Entities", "debarment"),
    "iadb_sanctions":      ("🏦", "Inter-American Dev. Bank", "Sanctions", "debarment"),
    "wb_sanctions":        ("🏦", "World Bank", "Ineligible Firms", "debarment"),
    # Reference data (not risk indicators) — surfaced separately
    "ru_cbr_banks":        ("🇷🇺", "Russia", "Banking Registry (CBR)", "reference"),
    "ext_us_irs_ffi":      ("🇺🇸", "United States", "FATCA FFI Registry", "reference"),
    "ext_ru_egrul":        ("🇷🇺", "Russia", "EGRUL Company Registry", "reference"),
    "ext_icij_offshoreleaks":("📄", "ICIJ", "Offshore Leaks", "leak"),
    "iso9362_bic":         ("🌐", "SWIFT (ISO 9362)", "BIC Reference Data", "reference"),
    "wikidata":            ("📚", "Wikidata", "Wikidata", "reference"),
    "ext_wd_peps":         ("📚", "Wikidata", "PEPs", "pep"),
}

# Fallback for unmapped datasets — uses the 2-letter prefix before "_"
COUNTRY_FLAGS: dict[str, str] = {
    "us": "🇺🇸", "eu": "🇪🇺", "gb": "🇬🇧", "ch": "🇨🇭", "ca": "🇨🇦",
    "au": "🇦🇺", "jp": "🇯🇵", "ua": "🇺🇦", "nz": "🇳🇿", "fr": "🇫🇷",
    "de": "🇩🇪", "be": "🇧🇪", "at": "🇦🇹", "nl": "🇳🇱", "ee": "🇪🇪",
    "cz": "🇨🇿", "pl": "🇵🇱", "es": "🇪🇸", "it": "🇮🇹", "il": "🇮🇱",
    "ir": "🇮🇷", "iq": "🇮🇶", "kp": "🇰🇵", "ar": "🇦🇷", "br": "🇧🇷",
    "az": "🇦🇿", "eg": "🇪🇬", "in": "🇮🇳", "id": "🇮🇩", "by": "🇧🇾",
    "ru": "🇷🇺", "cn": "🇨🇳", "kz": "🇰🇿", "lv": "🇱🇻", "lt": "🇱🇹",
    "no": "🇳🇴", "se": "🇸🇪", "fi": "🇫🇮", "dk": "🇩🇰", "ie": "🇮🇪",
    "is": "🇮🇸", "gg": "🇬🇬", "ky": "🇰🇾", "im": "🇮🇲", "mt": "🇲🇹",
    "cy": "🇨🇾", "hr": "🇭🇷", "bg": "🇧🇬", "ro": "🇷🇴", "rs": "🇷🇸",
    "ge": "🇬🇪", "am": "🇦🇲", "tr": "🇹🇷", "sa": "🇸🇦", "ae": "🇦🇪",
    "za": "🇿🇦", "mx": "🇲🇽", "co": "🇨🇴", "ve": "🇻🇪", "cu": "🇨🇺",
    "hk": "🇭🇰", "tw": "🇹🇼", "sg": "🇸🇬", "kr": "🇰🇷", "un": "🇺🇳",
}
COUNTRY_NAMES: dict[str, str] = {
    "us": "United States", "eu": "European Union", "gb": "United Kingdom",
    "ch": "Switzerland", "ca": "Canada", "au": "Australia", "jp": "Japan",
    "ua": "Ukraine", "nz": "New Zealand", "fr": "France", "de": "Germany",
    "be": "Belgium", "at": "Austria", "nl": "Netherlands", "ee": "Estonia",
    "cz": "Czechia", "pl": "Poland", "es": "Spain", "it": "Italy",
    "il": "Israel", "ir": "Iran", "iq": "Iraq", "kp": "DPRK",
    "ar": "Argentina", "br": "Brazil", "az": "Azerbaijan", "eg": "Egypt",
    "in": "India", "id": "Indonesia", "by": "Belarus", "ru": "Russia",
    "cn": "China", "kz": "Kazakhstan", "lv": "Latvia", "lt": "Lithuania",
    "no": "Norway", "se": "Sweden", "fi": "Finland", "dk": "Denmark",
    "ie": "Ireland", "is": "Iceland", "gg": "Guernsey", "ky": "Cayman Islands",
    "im": "Isle of Man", "mt": "Malta", "cy": "Cyprus", "hr": "Croatia",
    "bg": "Bulgaria", "ro": "Romania", "rs": "Serbia", "ge": "Georgia",
    "am": "Armenia", "tr": "Turkey", "sa": "Saudi Arabia", "ae": "UAE",
    "za": "South Africa", "mx": "Mexico", "co": "Colombia", "ve": "Venezuela",
    "cu": "Cuba", "hk": "Hong Kong", "tw": "Taiwan", "sg": "Singapore",
    "kr": "Korea", "un": "United Nations",
}

# Categories that count as "the country has sanctioned this entity"
SANCTIONING_CATEGORIES = {"sanctions", "export_control"}


def categorize_datasets(datasets: list[str] | None) -> dict[str, Any]:
    """Group a result's datasets by sanctioning category and country.

    Returns a dict with these keys:
        sanctions: {(flag, country): [list_labels]} — primary findings
        counter_sanctions: same shape — inverted meaning, explained inline
        other_risk: same shape — debarment, regulatory, crime
        reference: [(flag, country, label)] — registries, FATCA, KYB
        unmapped: [dataset_codes] — codes we couldn't classify
    """
    out: dict[str, Any] = {
        "sanctions": {},
        "counter_sanctions": {},
        "other_risk": {},
        "reference": [],
        "unmapped": [],
    }
    for ds in datasets or []:
        info = DATASET_INFO.get(ds)
        if info is not None:
            flag, country, label, category = info
        else:
            # Unmapped — try prefix detection
            prefix = ds.split("_", 1)[0] if "_" in ds else ds
            if prefix == "ext":
                # External enrichment — silently skip (not sanctions)
                continue
            flag = COUNTRY_FLAGS.get(prefix)
            country = COUNTRY_NAMES.get(prefix)
            if not (flag and country):
                out["unmapped"].append(ds)
                continue
            # Country known but specific dataset unknown — assume sanctions-grade
            # (most government-prefixed OS datasets are sanctions lists)
            label = ds
            category = "sanctions"

        key = (flag, country)
        if category in SANCTIONING_CATEGORIES:
            out["sanctions"].setdefault(key, []).append(label)
        elif category == "counter_sanctions":
            out["counter_sanctions"].setdefault(key, []).append(label)
        elif category in ("debarment", "regulatory", "crime"):
            out["other_risk"].setdefault(key, []).append(label)
        elif category in ("reference", "leak", "pep"):
            out["reference"].append((flag, country, label))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Bulk screening: fetch full bank list, screen each, render PDF
# ─────────────────────────────────────────────────────────────────────────────

BIK_INFO_BASE_XML_URL = "https://bik-info.ru/base/base.xml"


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_bank_list_from_base_xml() -> list[dict[str, str]]:
    """Fetch and parse the full list of Russian banks from bik-info.ru/base.xml.

    Returns a list of dicts ``[{"bik": "...", "name": "...", "inn": "..."}, ...]``.
    Deduplicates by BIK (the registry's natural primary key) and only includes
    rows where BIK is a valid 9-digit numeric string.

    The parser is intentionally defensive: bik-info.ru's XML schema isn't
    documented publicly and could express bank data either as element children
    (``<bank><bik>X</bik><name>Y</name></bank>``) or as attributes (``<bank
    bik="X" name="Y"/>``). The function walks every element in the tree,
    extracts candidate fields from both attributes and children, and accepts
    any element that yielded a BIK-shaped value. Multiple namespace prefixes
    are stripped via the ``}`` split for the same reason.

    Cached for 24 hours since the bank registry only changes when CBR
    licenses, revokes, or merges banks (handful of events per year).
    """
    try:
        r = requests.get(BIK_INFO_BASE_XML_URL, timeout=60)
        r.raise_for_status()
    except Exception:
        return []
    try:
        root = ET.fromstring(r.content)
    except Exception:
        return []

    def _extract_fields(elem: ET.Element) -> dict[str, str]:
        data: dict[str, str] = {}
        # Attributes — strip namespace prefixes if any
        for attr_name, attr_value in (elem.attrib or {}).items():
            key = attr_name.lower().split("}")[-1]
            val = (attr_value or "").strip()
            if not val:
                continue
            if key in ("bik", "bic") and not data.get("bik"):
                data["bik"] = val
            elif key in ("name", "namebank", "shortname", "bank_name") and not data.get("name"):
                data["name"] = val
            elif key in ("namep", "fullname", "name_full") and not data.get("name_full"):
                data["name_full"] = val
            elif key in ("inn", "innbank") and not data.get("inn"):
                data["inn"] = val
        # Children
        for child in elem:
            tag = (child.tag or "").lower().split("}")[-1]
            text = (child.text or "").strip()
            if not text:
                continue
            if tag in ("bik", "bic") and not data.get("bik"):
                data["bik"] = text
            elif tag in ("name", "namebank", "shortname", "bank_name") and not data.get("name"):
                data["name"] = text
            elif tag in ("namep", "fullname", "name_full") and not data.get("name_full"):
                data["name_full"] = text
            elif tag in ("inn", "innbank") and not data.get("inn"):
                data["inn"] = text
        return data

    banks: list[dict[str, str]] = []
    seen_biks: set[str] = set()
    for elem in root.iter():
        data = _extract_fields(elem)
        bik = data.get("bik", "")
        if len(bik) != 9 or not bik.isdigit() or bik in seen_biks:
            continue
        seen_biks.add(bik)
        # Promote name_full to name if we don't have a shorter name
        if not data.get("name") and data.get("name_full"):
            data["name"] = data["name_full"]
        banks.append(data)
    return banks


# Verdict-state constants for the bulk-screening path. Mirror the colour
# scheme used in Step 4's UI.
VERDICT_WHITELISTED = "WHITELISTED"
VERDICT_MATCH = "MATCH"           # 🔴 strict hit AND OFAC
VERDICT_REVIEW = "REVIEW"         # 🟡 strict hit, no OFAC
VERDICT_CLEAR = "CLEAR"           # 🟢 no strict hit
VERDICT_ERROR = "ERROR"           # ⚠️ screening failed (network/parse/etc)

VERDICT_EMOJI: dict[str, str] = {
    VERDICT_WHITELISTED: "✅",
    VERDICT_MATCH: "🔴",
    VERDICT_REVIEW: "🟡",
    VERDICT_CLEAR: "🟢",
    VERDICT_ERROR: "⚠️",
}


@st.cache_data(ttl=3600, show_spinner=False)
def screen_bank_simple(
    bik: str,
    name: str,
    inn: str,
    api_key: str,
) -> dict[str, str]:
    """Headless verdict for a single bank — used by the bulk PDF generator.

    Same verdict logic as the interactive Step 4 (whitelist → strict-search →
    OS-side whitelist → OFAC split), but without UI side effects so it can be
    called in a tight loop over hundreds of banks.

    Caching: 1h TTL per (bik, name, inn, api_key) tuple. The TTL is shorter
    than the 24h on ``fetch_bank_list_from_base_xml`` because sanctions lists
    can change overnight (mid-week OFAC updates are common) and the bulk
    PDF should pick those up promptly when regenerated.

    Returns ``{"verdict": str, "emoji": str, "detail": str}`` where verdict
    is one of the ``VERDICT_*`` constants and detail is a short human-readable
    explanation suitable for a PDF cell.
    """
    try:
        whitelist = load_whitelist_swifts()
        wl_swifts = set(whitelist.keys())

        # 1. Resolve SWIFT from iban.ru (the cheap cached lookup; we don't
        # call CBR/Dadata in bulk mode to keep the per-bank cost minimal)
        iban_swift = iban_ru_swift_for_bic(bik)
        swifts_8: set[str] = set()
        if iban_swift:
            s8 = iban_swift[:8].upper()
            if re.match(r"^[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}$", s8):
                swifts_8.add(s8)

        # 2. Input-side whitelist check
        wl_hit = swifts_8 & wl_swifts
        if wl_hit:
            return {
                "verdict": VERDICT_WHITELISTED,
                "emoji": VERDICT_EMOJI[VERDICT_WHITELISTED],
                "detail": f"SWIFT {next(iter(wl_hit))} on OhMySwift",
            }

        if not api_key:
            return {
                "verdict": VERDICT_ERROR,
                "emoji": VERDICT_EMOJI[VERDICT_ERROR],
                "detail": "no OpenSanctions API key",
            }

        # 3. Strict identifier search — BIK, SWIFT, INN (validated).
        # Mirror the interactive Step 3 by expanding the BIK pool with
        # head-office BICs when the bank name contains SBERBANK/VTB keywords.
        # Without this, a Sberbank or VTB *branch* BIK from base.xml would
        # screen as CLEAR because OpenSanctions indexes the head-office BIK
        # not the branch BIKs — the interactive flow rescues these via name
        # detection, so the bulk flow must do the same to stay consistent.
        bik_pool: set[str] = {bik}
        for hob in head_office_bics_for_name(name):
            bik_pool.add(hob)

        hits: dict[str, dict] = {}
        # BIK search — one per unique BIK in the expanded pool
        for b in bik_pool:
            rb = opensanctions_search_by_property(api_key, "bikCode", b)
            for e in rb.get("results", []) or []:
                if e.get("id"):
                    hits[e["id"]] = e
        # SWIFT
        for s8 in swifts_8:
            rs = opensanctions_search_by_property(api_key, "swiftBic", s8)
            for e in rs.get("results", []) or []:
                if e.get("id"):
                    hits[e["id"]] = e
        # INN (validate first)
        clean_inn = "".join(c for c in str(inn or "") if c.isdigit())
        if len(clean_inn) in (10, 12):
            ri = opensanctions_search_by_property(api_key, "innCode", clean_inn)
            for e in ri.get("results", []) or []:
                if e.get("id"):
                    hits[e["id"]] = e

        # 4. OS-side per-entity whitelist filter — if an entity's OS-stored
        # SWIFT is on the OhMySwift list, treat it as whitelisted
        real_hits: dict[str, dict] = {}
        any_os_wl = False
        for eid, entity in hits.items():
            entity_swifts = (entity.get("properties") or {}).get("swiftBic") or []
            e8 = {s.strip().upper()[:8] for s in entity_swifts if isinstance(s, str) and len(s.strip()) >= 8}
            if e8 & wl_swifts:
                any_os_wl = True
                continue
            real_hits[eid] = entity

        if not real_hits:
            if any_os_wl:
                return {
                    "verdict": VERDICT_WHITELISTED,
                    "emoji": VERDICT_EMOJI[VERDICT_WHITELISTED],
                    "detail": "OS-side: matched entity SWIFT on whitelist",
                }
            return {
                "verdict": VERDICT_CLEAR,
                "emoji": VERDICT_EMOJI[VERDICT_CLEAR],
                "detail": "no strict identifier match",
            }

        # 5. OFAC split
        if any(entity_has_ofac(e) for e in real_hits.values()):
            return {
                "verdict": VERDICT_MATCH,
                "emoji": VERDICT_EMOJI[VERDICT_MATCH],
                "detail": f"OFAC sanctioned ({len(real_hits)} entity)",
            }
        return {
            "verdict": VERDICT_REVIEW,
            "emoji": VERDICT_EMOJI[VERDICT_REVIEW],
            "detail": f"non-OFAC match ({len(real_hits)} entity)",
        }
    except Exception as exc:
        return {
            "verdict": VERDICT_ERROR,
            "emoji": VERDICT_EMOJI[VERDICT_ERROR],
            "detail": f"screening failed: {exc}",
        }


def _register_cyrillic_font_for_pdf() -> str:
    """Register a Unicode-capable font with reportlab for Cyrillic support.

    ReportLab's built-in fonts (Helvetica, Times-Roman) don't include Cyrillic
    glyphs. DejaVu Sans is the standard fallback on Linux servers (including
    Streamlit Cloud's Debian image) and covers full Cyrillic + Latin Extended.

    Returns the registered font name, or "Helvetica" if no suitable font was
    found (in which case Cyrillic bank names will render as boxes — degraded
    but the PDF still produces correctly).

    Imports are local because reportlab is an optional dependency only needed
    when bulk PDFs are generated; the rest of the app shouldn't fail to load
    if reportlab is missing from requirements.txt.
    """
    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
    except ImportError:
        return "Helvetica"

    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/Library/Fonts/Arial Unicode.ttf",  # macOS dev fallback
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont("BankFont", path))
                return "BankFont"
            except Exception:
                continue
    return "Helvetica"


def generate_screening_pdf(
    banks_with_verdicts: list[tuple[dict[str, str], dict[str, str]]],
) -> bytes:
    """Render the bulk-screening table as a PDF.

    Args:
        banks_with_verdicts: list of (bank_info, verdict_info) tuples where
            bank_info has bik/name/inn keys and verdict_info has verdict/
            emoji/detail keys.

    Layout: cover paragraph with summary counts, then a single sorted table
    (red first, then yellow, then white, green, error). Row backgrounds are
    color-coded by verdict so the file is scannable when printed.

    Raises a clear error message if reportlab is not installed — this is
    common during initial deployment when ``requirements.txt`` hasn't been
    refreshed alongside the source file.
    """
    try:
        from io import BytesIO
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        )
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        from reportlab.lib.units import cm
    except ImportError as exc:
        raise RuntimeError(
            f"reportlab not installed ({exc}). "
            "Commit the updated `requirements.txt` (it must include "
            "`reportlab>=4.0`) and reboot the Streamlit app — secrets "
            "or settings changes don't pull new packages, a deploy push "
            "or manual reboot does."
        ) from exc

    font_name = _register_cyrillic_font_for_pdf()
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=1.5 * cm,
        rightMargin=1.5 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        title="Russian Banks — OpenSanctions Screening",
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "BulkTitle", parent=styles["Title"], fontName=font_name, fontSize=16, spaceAfter=8,
    )
    body_style = ParagraphStyle(
        "BulkBody", parent=styles["Normal"], fontName=font_name, fontSize=9,
    )
    cell_style = ParagraphStyle(
        "BulkCell", parent=styles["Normal"], fontName=font_name, fontSize=8, leading=10,
    )

    counts: dict[str, int] = {}
    for _, v in banks_with_verdicts:
        counts[v["verdict"]] = counts.get(v["verdict"], 0) + 1

    story: list[Any] = []
    story.append(Paragraph("Russian Banks — OpenSanctions Screening", title_style))
    story.append(Paragraph(
        f"Source: bik-info.ru/base.xml &nbsp;·&nbsp; "
        f"Total banks: <b>{len(banks_with_verdicts)}</b> &nbsp;·&nbsp; "
        f"Verdict logic: input-SWIFT whitelist → strict identifier search "
        f"(BIK / SWIFT / INN) → OS-SWIFT whitelist → OFAC split",
        body_style,
    ))
    story.append(Spacer(1, 8))
    summary_html = " &nbsp;·&nbsp; ".join(
        f"{VERDICT_EMOJI[k]} <b>{k}:</b> {counts.get(k, 0)}"
        for k in (VERDICT_MATCH, VERDICT_REVIEW, VERDICT_WHITELISTED, VERDICT_CLEAR, VERDICT_ERROR)
    )
    story.append(Paragraph(summary_html, body_style))
    story.append(Spacer(1, 12))

    # Sort: most severe first (red, yellow, then white, green, errors)
    order = {
        VERDICT_MATCH: 0,
        VERDICT_REVIEW: 1,
        VERDICT_WHITELISTED: 2,
        VERDICT_CLEAR: 3,
        VERDICT_ERROR: 4,
    }
    sorted_banks = sorted(
        banks_with_verdicts,
        key=lambda x: (order.get(x[1]["verdict"], 99), x[0].get("bik", "")),
    )

    table_data: list[list[Any]] = [["BIC", "Bank Name", "Verdict"]]
    for bank, verdict in sorted_banks:
        name = bank.get("name", "") or "(no name)"
        # Truncate very long names to keep the row at a sane height
        if len(name) > 120:
            name = name[:117] + "…"
        verdict_cell = f"{verdict['emoji']} {verdict['verdict']}"
        if verdict.get("detail"):
            verdict_cell += f"<br/><font size='7' color='#666666'>{verdict['detail']}</font>"
        table_data.append([
            Paragraph(bank.get("bik", ""), cell_style),
            Paragraph(name, cell_style),
            Paragraph(verdict_cell, cell_style),
        ])

    table = Table(
        table_data,
        colWidths=[2.4 * cm, 10.5 * cm, 4.6 * cm],
        repeatRows=1,  # repeat header on every page
    )
    table_style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, -1), font_name),
        ("FONTSIZE",   (0, 0), (-1, 0), 10),
        ("ALIGN",      (0, 0), (-1, 0), "CENTER"),
        ("VALIGN",     (0, 0), (-1, -1), "TOP"),
        ("GRID",       (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
    ]
    # Per-row tinting by verdict
    row_tint = {
        VERDICT_MATCH:       colors.HexColor("#fee2e2"),  # light red
        VERDICT_REVIEW:      colors.HexColor("#fef3c7"),  # light amber
        VERDICT_WHITELISTED: colors.HexColor("#dcfce7"),  # light green
        VERDICT_CLEAR:       colors.HexColor("#f9fafb"),  # very light grey
        VERDICT_ERROR:       colors.HexColor("#e5e7eb"),  # neutral grey
    }
    for i, (_, v) in enumerate(sorted_banks, start=1):
        tint = row_tint.get(v["verdict"])
        if tint is not None:
            table_style_cmds.append(("BACKGROUND", (0, i), (-1, i), tint))
    table.setStyle(TableStyle(table_style_cmds))
    story.append(table)

    doc.build(story)
    return buffer.getvalue()


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
        "**Verdict rules**  \n"
        "✅ **Whitelisted** — SWIFT on OhMySwift list  \n"
        "🔴 **Match** — strict hit AND on US OFAC  \n"
        "🟡 **Review** — strict hit, no US OFAC  \n"
        "🟢 **Clear** — no whitelist, no strict hit"
    )

    # ── Bulk screening: full Russian-bank list as PDF ─────────────────
    st.markdown("---")
    with st.expander("📊 Full bank list (PDF)", expanded=False):
        st.caption(
            "Bulk-screen every Russian bank in "
            "[bik-info.ru/base.xml](https://bik-info.ru/base/base.xml) and "
            "produce a PDF with **BIC**, **bank name**, and **verdict**."
        )
        if st.button("🔄 Generate / refresh PDF", use_container_width=True):
            st.session_state["bulk_pdf_run"] = True
            st.session_state["bulk_pdf"] = None
        if st.session_state.get("bulk_pdf"):
            st.success(
                f"PDF ready — {st.session_state.get('bulk_pdf_count', '?')} banks "
                f"screened {st.session_state.get('bulk_pdf_when', '')}"
            )
            st.download_button(
                "⬇️ Download PDF",
                data=st.session_state["bulk_pdf"],
                file_name="russian_banks_screening.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        else:
            st.caption(
                "_The first run takes a few minutes (one /search per bank). "
                "Subsequent runs reuse the per-bank cache (1h TTL)._"
            )

# ── Bulk PDF generation (runs in place of regular screening) ─────────────
# Triggered by the sidebar button which sets session_state["bulk_pdf_run"].
# We handle it here, before the regular BIC input, so the bulk flow can run
# without requiring the user to also enter a BIC. The final `st.stop()`
# below prevents the regular screening UI from rendering on top.
if st.session_state.get("bulk_pdf_run") and not st.session_state.get("bulk_pdf"):
    st.markdown("## 📊 Bulk screening — generating PDF")
    st.caption(
        "Fetching the full Russian bank list from bik-info.ru/base.xml, "
        "running the same verdict logic as the interactive tool, and "
        "compiling a PDF. The first run takes a few minutes; per-bank "
        "results are cached for 1 hour so subsequent runs are faster."
    )
    bulk_progress = st.progress(0.0)
    bulk_status = st.empty()

    bulk_status.info("Step 1/3 · Fetching base.xml from bik-info.ru…")
    banks = fetch_bank_list_from_base_xml()
    if not banks:
        st.error(
            "Couldn't fetch or parse `https://bik-info.ru/base/base.xml`. "
            "The endpoint may be temporarily unavailable or the XML schema "
            "may have changed. Click **Generate / refresh PDF** again later, "
            "or check the network from the deployment."
        )
        st.session_state["bulk_pdf_run"] = False
        st.stop()
    bulk_status.info(
        f"Step 2/3 · Screening {len(banks)} banks against OpenSanctions "
        "(cached per BIC, so already-screened banks are instant)…"
    )

    # Per-bank screening with progress updates
    results: list[tuple[dict[str, str], dict[str, str]]] = []
    for i, bank in enumerate(banks):
        verdict = screen_bank_simple(
            bank["bik"],
            bank.get("name", ""),
            bank.get("inn", ""),
            OPENSANCTIONS_API_KEY,
        )
        results.append((bank, verdict))
        # Update progress + status every bank (cheap), or every 10 if many
        bulk_progress.progress((i + 1) / len(banks))
        if i % 10 == 0 or i == len(banks) - 1:
            short_name = (bank.get("name") or bank["bik"])[:60]
            bulk_status.info(
                f"Step 2/3 · {i + 1}/{len(banks)} — `{bank['bik']}` "
                f"{short_name} → {verdict['emoji']} {verdict['verdict']}"
            )

    bulk_status.info("Step 3/3 · Rendering PDF…")
    try:
        pdf_bytes = generate_screening_pdf(results)
    except Exception as exc:
        st.error(f"PDF generation failed: {exc}")
        st.session_state["bulk_pdf_run"] = False
        st.stop()

    # Persist in session state so the sidebar's download button picks it up
    from datetime import datetime
    st.session_state["bulk_pdf"] = pdf_bytes
    st.session_state["bulk_pdf_count"] = len(banks)
    st.session_state["bulk_pdf_when"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    st.session_state["bulk_pdf_run"] = False

    bulk_progress.empty()
    bulk_status.success(
        f"✅ PDF generated — {len(banks)} banks screened. "
        "**Download from the sidebar.**"
    )
    # Verdict summary
    counts: dict[str, int] = {}
    for _, v in results:
        counts[v["verdict"]] = counts.get(v["verdict"], 0) + 1
    summary = " · ".join(
        f"{VERDICT_EMOJI[k]} **{k}**: {counts.get(k, 0)}"
        for k in (VERDICT_MATCH, VERDICT_REVIEW, VERDICT_WHITELISTED, VERDICT_CLEAR, VERDICT_ERROR)
    )
    st.markdown(summary)
    st.stop()

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

# Placeholder for the bank summary block. Filled after Step 2 resolves all
# identifiers and names. Using a placeholder lets the summary render in this
# position (right under the search field) while the data that populates it
# is gathered by later code — Streamlit otherwise renders strictly top-down.
bank_summary_placeholder = st.empty()

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

    # ── iban.ru SWIFT BIC lookup ──────────────────────────────────────────
    # Independent of CBR's SOAP result, look the BIC up in iban.ru's SWIFT↔BIK
    # directory. iban.ru publishes the table with SWIFT SCRL's permission and
    # has per-branch SWIFTs that CBR's SOAP doesn't always expose. The lookup
    # is cached for 24h so this is effectively a one-time HTTP call per day
    # regardless of how many BICs are screened.
    iban_ru_swift = iban_ru_swift_for_bic(bic)
    if iban_ru_swift:
        st.markdown(
            f"📡 **iban.ru SWIFT lookup:** `{iban_ru_swift}` "
            "_(source: iban.ru BIK↔SWIFT directory, S.W.I.F.T. SCRL data)_"
        )
        # Merge into cbr's swift_codes pool so Step 3 picks it up alongside
        # any SWIFTs CBR extracted. Both 11- and 8-char forms — OpenSanctions
        # stores SWIFTs canonically as 8-char so sending both forms guarantees
        # the matcher finds the entity regardless of how it's keyed.
        existing = list(cbr.get("swift_codes") or [])
        added = []
        if iban_ru_swift not in existing:
            existing.append(iban_ru_swift)
            added.append(iban_ru_swift)
        if len(iban_ru_swift) == 11 and iban_ru_swift[:8] not in existing:
            existing.append(iban_ru_swift[:8])
            added.append(iban_ru_swift[:8])
        cbr["swift_codes"] = existing
        if added and trace is not None:
            # Track in the resolution_trace so the audit log shows the source
            cbr.setdefault("resolution_trace", []).append(
                f"iban.ru: added SWIFT(s) {', '.join(added)} for BIC {bic}"
            )
    else:
        st.caption(
            "_(iban.ru: no SWIFT entry for this BIC — bank may not be on the table "
            "or the table doesn't cover this branch)_"
        )

    # ── Dadata findById/bank — authoritative INN/KPP/OGRN/state ───────────
    # Dadata sources from CBR + FNS + EGRUL, so it's the most reliable single
    # source for the bank's INN. We merge what it returns into the cbr dict
    # without overwriting any value CBR's SOAP already provided (CBR is the
    # primary regulator-owned source; Dadata is the gap-filler).
    try:
        dadata_api_key = st.secrets.get("DADATA_API_KEY", "")
    except Exception:
        # Raises StreamlitSecretNotFoundError if no secrets.toml exists at all
        # (typical for local dev without setup). Treat as "not configured".
        dadata_api_key = ""
    dadata = dadata_find_bank(bic, dadata_api_key) if dadata_api_key else {"error": "no_api_key"}
    if dadata.get("error"):
        if dadata["error"] == "no_api_key":
            st.caption("_(Dadata: `DADATA_API_KEY` not configured in Streamlit secrets — skipping)_")
        elif dadata["error"] == "no_suggestions":
            st.caption(f"_(Dadata: no bank found for BIC `{bic}`)_")
        elif dadata["error"].startswith("auth_failed"):
            st.warning(f"⚠️ Dadata authentication failed ({dadata['error']}) — check `DADATA_API_KEY`")
        else:
            st.caption(f"_(Dadata: {dadata['error']})_")
    else:
        # Display the key identifiers as a single dense line
        pieces = []
        if dadata.get("inn"):
            pieces.append(f"INN `{dadata['inn']}`")
        if dadata.get("kpp"):
            pieces.append(f"KPP `{dadata['kpp']}`")
        if dadata.get("ogrn"):
            pieces.append(f"OGRN `{dadata['ogrn']}`")
        if dadata.get("reg_number"):
            pieces.append(f"Reg# `{dadata['reg_number']}`")
        if pieces:
            st.markdown(f"🧾 **Dadata lookup:** {' · '.join(pieces)}")
        if dadata.get("name_full") or dadata.get("name_english"):
            name_line = dadata.get("name_full") or ""
            if dadata.get("name_english"):
                name_line = f"{name_line} _({dadata['name_english']})_" if name_line else dadata["name_english"]
            st.caption(f"_Name (Dadata): {name_line}_")
        # Bank state — important: a LIQUIDATING or LICENSE_REVOKED bank should
        # not be transacted with, regardless of whether it's on a sanctions list.
        status_val = dadata.get("state_status")
        if status_val and status_val != "ACTIVE":
            st.warning(
                f"⚠️ **Dadata reports bank state: `{status_val}`** "
                f"(as of {dadata.get('state_actuality_date') or 'unknown'}). "
                "A non-active bank may not be able to receive or send payments."
            )

        # Merge Dadata findings into cbr dict — fill empty slots only, never
        # overwrite. This way Step 3 (OpenSanctions match) and downstream
        # logic see Dadata's values when CBR didn't provide them.
        added_to_trace: list[str] = []
        if dadata.get("inn") and not cbr.get("inn_cbr"):
            cbr["inn_cbr"] = dadata["inn"]
            added_to_trace.append(f"INN {dadata['inn']}")
        if dadata.get("ogrn") and not cbr.get("ogrn_cbr"):
            cbr["ogrn_cbr"] = dadata["ogrn"]
            added_to_trace.append(f"OGRN {dadata['ogrn']}")
        if dadata.get("kpp"):
            kpp_list = list(cbr.get("kpp_codes_cbr") or [])
            if dadata["kpp"] not in kpp_list:
                kpp_list.append(dadata["kpp"])
                cbr["kpp_codes_cbr"] = kpp_list
                added_to_trace.append(f"KPP {dadata['kpp']}")
        # Merge SWIFT into swift_codes pool, both 11- and 8-char forms
        if dadata.get("swift"):
            sw = dadata["swift"].strip().upper()
            sw_pool = list(cbr.get("swift_codes") or [])
            for s in [sw, sw + "XXX" if len(sw) == 8 else sw, sw[:8] if len(sw) == 11 else sw]:
                if s and s not in sw_pool:
                    sw_pool.append(s)
                    added_to_trace.append(f"SWIFT {s}")
            cbr["swift_codes"] = sw_pool
        if added_to_trace:
            cbr.setdefault("resolution_trace", []).append(
                f"Dadata: added {', '.join(added_to_trace)}"
            )

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
                "No SWIFT codes registered for this BIC (neither CBR nor iban.ru). "
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

    # ── Special-case keyword routing ──────────────────────────────────────
    # If the resolved bank name mentions SBERBANK or VTB, extend the BIK pool
    # with the head-office BICs of those banks. This ensures the head-office
    # entity is screened in OpenSanctions even when the entered BIC is a
    # regional branch or affiliate whose own BIK doesn't appear in OS.
    # Mapping lives at module level (`SPECIAL_BANK_HEAD_OFFICE_BICS`) so the
    # bulk screener uses the same rule and produces identical verdicts.
    # Concatenate all available name variants and uppercase once for
    # case-insensitive substring detection across Russian + transliterated forms.
    name_haystack = " ".join(
        n for n in [
            name_ru, short_name_ru, name_en, short_name_en,
            *(cbr.get("short_names_cbr") or []),
            *(cbr.get("full_names_cbr") or []),
        ] if n
    ).upper()
    extra_bics_added: list[tuple[str, str]] = []  # (matched_keyword, bic_added)
    for keyword, head_office_bics in SPECIAL_BANK_HEAD_OFFICE_BICS.items():
        if keyword in name_haystack:
            for head_office_bic in head_office_bics:
                if head_office_bic not in bik_pool:
                    bik_pool.add(head_office_bic)
                    extra_bics_added.append((keyword, head_office_bic))
    if extra_bics_added:
        extra_lines = "\n".join(
            f"- `{b}` (matched `{kw}` in bank name)"
            for kw, b in extra_bics_added
        )
        st.info(
            f"📛 **Extended screening: {len(extra_bics_added)} additional "
            f"head-office BIC(s) added** based on bank name detection:\n\n"
            f"{extra_lines}\n\n"
            "These BICs are sent to OpenSanctions alongside the entered BIC, "
            "so the head-office legal entity is screened even when the "
            "entered BIC belongs to a regional branch or affiliate."
        )

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

    # ── Step 3: deferred. The strict-identifier screening that drives the
    # verdict happens in Step 4 below, since it consumes the same identifier
    # pools (`all_bics`, `all_swifts`, `inn_final`) and can be short-circuited
    # by the OhMySwift whitelist — running it here would mean making API calls
    # for whitelisted banks unnecessarily.
    status.update(
        label=f"Step 3 · resolved {len(all_bics)} BIC(s), {len(all_swifts)} SWIFT(s)"
        + (f", INN {inn_final}" if inn_final else ", no INN"),
        state="complete",
    )

# ── Populate the bank-details summary block (placeholder near the top) ────
# All identifier pools and names are now resolved (Step 1: CBR + iban.ru +
# Dadata, Step 2: bik-info.ru, Step 3: OS reference enrichment). Filling
# the placeholder here makes the summary visible right under the search
# field while the verdict logic still runs below.
with bank_summary_placeholder.container():
    st.markdown("#### 🏦 Bank details")

    # English name: Dadata's `name_english` is authoritative (from FNS/EGRUL)
    # when present; fall back to the transliteration of the Russian name
    # otherwise. Dadata may not return one for every bank.
    name_english_display = (
        dadata.get("name_english")
        if isinstance(dadata, dict) and dadata.get("name_english")
        else name_en
    ) or "_(not resolved)_"

    # BIC display: show the entered one prominently. If CBR returned more
    # BICs for this credit organization (head office + branches), note that.
    bic_extras = [b for b in (all_bics or []) if b != bic]
    if bic_extras:
        bic_display = f"`{bic}` _(+ {len(bic_extras)} other BIC(s) in the same credit org: {', '.join(f'`{b}`' for b in bic_extras[:5])}{' …' if len(bic_extras) > 5 else ''})_"
    else:
        bic_display = f"`{bic}`"

    # SWIFT display: 8-char and 11-char forms together. Dedupe while
    # preserving canonical 11-char first when both are present for the same
    # institution, since the 11-char form is more informative (carries the
    # branch code, e.g. VTBRRUM2MS2 vs the 8-char VTBRRUM2).
    swift_display_list = []
    seen_8: set[str] = set()
    # 11-char first
    for s in all_swifts or []:
        if len(s) == 11 and s[:8] not in seen_8:
            swift_display_list.append(s)
            seen_8.add(s[:8])
    # then 8-char that didn't have an 11-char companion
    for s in all_swifts or []:
        if len(s) == 8 and s not in seen_8:
            swift_display_list.append(s)
            seen_8.add(s)
    swift_display = (
        ", ".join(f"`{s}`" for s in swift_display_list)
        if swift_display_list
        else "_(none registered)_"
    )

    # INN — surface invalid shapes loudly so the analyst sees data-flow issues
    inn_display = "_(not resolved)_"
    if inn_final:
        inn_clean = "".join(c for c in str(inn_final) if c.isdigit())
        if len(inn_clean) in (10, 12):
            inn_display = f"`{inn_clean}`"
        else:
            inn_display = (
                f"⚠️ `{inn_final}` "
                f"_(invalid INN shape: {len(inn_clean)} digits; "
                "INNs must be 10 or 12)_"
            )

    rows = [
        ("Original name", name_ru or "_(not resolved)_"),
        ("Name in English", name_english_display),
        ("BIC", bic_display),
        ("INN", inn_display),
        ("SWIFT", swift_display),
    ]
    table_md = "| | |\n|---|---|\n" + "\n".join(
        f"| **{label}** | {value} |" for label, value in rows
    )
    st.markdown(table_md)

# ─── STEP 4 ──────────────────────────────────────────────────────────────────
st.markdown("## Step 4 · Verdict")
st.caption(
    "Verdict logic: **(1)** if any SWIFT matches the OhMySwift whitelist of "
    "Russian banks NOT under US/EU sanctions → ✅ WHITELISTED. **(2)** otherwise, "
    "run a strict OpenSanctions search on BIK, SWIFT, and INN — any hit → 🔴 MATCH; "
    "no hits → 🟢 CLEAR."
)

# ── (1) Whitelist check ──────────────────────────────────────────────────
# Cross-reference the bank's resolved SWIFTs against the OhMySwift curated
# list of Russian banks NOT under US (SDN) and EU sanctions. If matched, we
# short-circuit and skip the OpenSanctions calls entirely — that's the spec.
whitelist_hits = check_whitelist_matches(swift_pool)
if whitelist_hits:
    hits_str = "\n".join(
        f"- `{swift_8}`" + (f" — _{name}_" if name else "")
        for swift_8, name in whitelist_hits
    )
    st.success(
        "## ✅ WHITELISTED\n\n"
        "This bank's SWIFT appears on the curated list of **Russian banks NOT under "
        "US (SDN) and EU sanctions**:\n\n"
        f"{hits_str}\n\n"
        "_Source: OhMySwift.xlsx (compiled from iban.ru's classification of Russian "
        "banks against the US SDN list and EU consolidated sanctions)._\n\n"
        "_Per the verdict spec, OpenSanctions strict-identifier search is skipped "
        "when whitelisted. The whitelist covers US + EU only — if you need to "
        "verify against UK, JP, CA, AU, CH, or UA sanctions specifically, run an "
        "ad-hoc OpenSanctions lookup outside this tool._"
    )

else:
    # ── (2) Strict identifier screening ──────────────────────────────────
    # Three /search calls against OpenSanctions, one per identifier type.
    # Each uses the field-targeted q=properties.X:Y syntax (NOT filter:X,
    # which doesn't exist on the search endpoint — passing it returns
    # unfiltered top-ranked entities, which is the false-positive trap that
    # earlier surfaced unrelated entities like Revival of Islamic Heritage
    # Society).
    with st.status("Step 4 · OpenSanctions strict identifier screening", expanded=True) as v_status:
        screening = opensanctions_strict_screening(
            OPENSANCTIONS_API_KEY,
            bics=all_bics,
            swifts=all_swifts,
            inn=inn_final,
        )
        # Audit table: what was searched, what came back
        st.markdown("**Searches performed:**")
        any_run = False
        for s in screening["searches"]:
            any_run = True
            if s.get("error"):
                st.markdown(
                    f"- ⚠️ `{s['prop']}={s['value']}` → {s['error']}"
                )
            elif s.get("n_hits", 0) == 0:
                st.markdown(
                    f"- 🟢 `{s['prop']}={s['value']}` → no hit"
                )
            else:
                st.markdown(
                    f"- 🔴 `{s['prop']}={s['value']}` → **{s['n_hits']} hit(s)**"
                )
        if not any_run:
            st.warning(
                "No identifiers available to screen. Steps 1 and 2 didn't resolve "
                "any of BIK, SWIFT, or INN. Cannot reach a verdict."
            )
        # Show URLs for transparency / debugging
        with st.expander("Strict-search URLs (debug)"):
            for s in screening["searches"]:
                if s.get("url"):
                    st.code(s["url"], language="text")
        v_status.update(
            label=f"Step 4 · {len(screening['hits_by_entity_id'])} unique entity(ies) hit",
            state="complete",
        )

    hits_by_id = screening["hits_by_entity_id"]

    # ── Per-entity whitelist filter (OS-side SWIFT check) ────────────────
    # The input-side whitelist check above only saw SWIFTs we resolved via
    # CBR, iban.ru and Dadata. OpenSanctions stores its own ``swiftBic``
    # property on every entity, which can carry SWIFTs we didn't surface —
    # the head-office canonical form, alternate branch codes, historical
    # SWIFTs the bank used before a rebrand. If a strict identifier hit
    # comes back from an entity whose OS-stored SWIFT is itself on the
    # OhMySwift whitelist, that entity IS the same clean bank reached via
    # a different identifier path (commonly the INN or head-office BIK
    # injection), and the match shouldn't drive a MATCH/REVIEW verdict.
    # Move such entities to a separate bucket and recompute the verdict
    # from the surviving hits.
    whitelist_swifts_set = set(load_whitelist_swifts().keys())
    os_whitelisted_entities: dict[str, dict[str, Any]] = {}
    for eid in list(hits_by_id.keys()):
        entity = hits_by_id[eid]["entity"]
        entity_swifts = (entity.get("properties") or {}).get("swiftBic") or []
        # Normalize each OS-stored SWIFT to 8-char canonical form and check
        # against the OhMySwift whitelist (which is 8-char-keyed)
        entity_swifts_8 = {
            s.strip().upper()[:8]
            for s in entity_swifts
            if isinstance(s, str) and len(s.strip()) >= 8
        }
        matched_wl = entity_swifts_8 & whitelist_swifts_set
        if matched_wl:
            os_whitelisted_entities[eid] = {
                **hits_by_id[eid],
                "matched_whitelist_swifts": sorted(matched_wl),
            }
            del hits_by_id[eid]

    if os_whitelisted_entities:
        # Surface what was filtered, so the analyst sees that some OpenSanctions
        # entities matched on identifier but were re-cleared via SWIFT-whitelist.
        n_wl = len(os_whitelisted_entities)
        names_str = ", ".join(
            f"_{h['entity'].get('caption', '(no caption)')}_"
            for h in list(os_whitelisted_entities.values())[:3]
        )
        if n_wl > 3:
            names_str += f" (+{n_wl - 3} more)"
        st.info(
            f"✅ **{n_wl} OpenSanctions entity(ies) matched on identifier but "
            "their OS-stored SWIFT is on the OhMySwift whitelist** — treating "
            f"as whitelisted, not counting toward MATCH/REVIEW verdict: {names_str}"
        )

    if hits_by_id:
        # ─── MATCH or REVIEW (depending on OFAC presence) ───────────────
        # Per the spec: any hit means the bank's identifier is on a watchlist.
        # The colour of the verdict depends on whether *any* matched entity
        # has US OFAC coverage. OFAC is the strictest globally because it
        # cuts off USD correspondent banking — a non-OFAC sanctions hit
        # (EU/UK/CH/JP/CA/AU/UA only) still warrants manual review but may
        # be transactable for non-USD flows.
        ofac_entity_ids = {
            eid for eid, h in hits_by_id.items() if entity_has_ofac(h["entity"])
        }
        n_ofac = len(ofac_entity_ids)
        n_total = len(hits_by_id)
        non_ofac_entity_ids = set(hits_by_id.keys()) - ofac_entity_ids

        if ofac_entity_ids:
            # 🔴 MATCH — at least one matched entity has US OFAC coverage
            st.error(
                f"## 🔴 MATCH — US OFAC sanctioned\n\n"
                f"**{n_total} OpenSanctions entity(ies) match this bank on a strict "
                f"identifier (BIK, SWIFT, or INN), and {n_ofac} of them is listed "
                "on a US OFAC sanctions list** (SDN and/or Consolidated).\n\n"
                "OFAC designation cuts off USD correspondent banking — US banks "
                "are prohibited from processing transactions touching this entity. "
                "**Do not transact in USD. Investigate before any flow.**"
            )
        else:
            # 🟡 REVIEW — hits exist but none are OFAC
            # Surface what jurisdictions DO appear, so the analyst sees what
            # they're reviewing against
            non_ofac_jurisdictions: set[str] = set()
            for hit in hits_by_id.values():
                bks = categorize_datasets(hit["entity"].get("datasets") or [])
                for (flag, country) in bks["sanctions"].keys():
                    non_ofac_jurisdictions.add(f"{flag} {country}")
            juris_line = (
                ", ".join(sorted(non_ofac_jurisdictions))
                if non_ofac_jurisdictions
                else "non-sanctions risk lists only (PEP/regulatory/debarment)"
            )
            st.warning(
                f"## 🟡 REVIEW — non-OFAC match\n\n"
                f"**{n_total} OpenSanctions entity(ies) match this bank on a strict "
                "identifier (BIK, SWIFT, or INN), but none are listed by US OFAC.**\n\n"
                f"Match jurisdictions: {juris_line}.\n\n"
                "USD correspondent banking is not blocked by OFAC for this entity. "
                "However, the bank is on a non-US sanctions, PEP, or risk list — "
                "**manual review required** to decide whether the specific transaction "
                "is permissible given the involved currencies, counterparties, and "
                "your own jurisdictional exposure."
            )

        # ── Top-level Sanctioning Jurisdictions (aggregated across all hits) ──
        agg_buckets = {"sanctions": {}, "counter_sanctions": {}, "other_risk": {}, "reference": [], "unmapped": []}
        for hit in hits_by_id.values():
            buckets = categorize_datasets(hit["entity"].get("datasets") or [])
            for k in ("sanctions", "counter_sanctions", "other_risk"):
                for key, lists in buckets[k].items():
                    agg_buckets[k].setdefault(key, []).extend(lists)
            agg_buckets["reference"].extend(buckets["reference"])
            agg_buckets["unmapped"].extend(buckets["unmapped"])

        st.markdown("### 🌍 Sanctioning Jurisdictions")
        if agg_buckets["sanctions"]:
            st.markdown(
                f"**{len(agg_buckets['sanctions'])} jurisdiction(s)** with sanctions / "
                "export-control coverage of the matched entity(ies):"
            )
            for (flag, country), lists in sorted(
                agg_buckets["sanctions"].items(), key=lambda kv: kv[0][1]
            ):
                deduped = list(dict.fromkeys(lists))
                list_str = " · ".join(f"_{l}_" for l in deduped)
                st.markdown(f"- {flag} **{country}** — {list_str}")
        else:
            st.info(
                "No government sanctioning jurisdictions in the matched entity's "
                "dataset list. The hit may be from PEP/RCA, regulatory warning, "
                "or other risk lists rather than from a sanctions program."
            )
        if agg_buckets["counter_sanctions"]:
            st.markdown("**Counter-sanctions (entity is target of):**")
            for (flag, country), lists in sorted(
                agg_buckets["counter_sanctions"].items(), key=lambda kv: kv[0][1]
            ):
                deduped = list(dict.fromkeys(lists))
                list_str = " · ".join(f"_{l}_" for l in deduped)
                st.markdown(f"- {flag} **{country}** — {list_str}")
        if agg_buckets["other_risk"]:
            st.markdown("**Other risk findings** _(debarment / regulatory / crime):_")
            for (flag, country), lists in sorted(
                agg_buckets["other_risk"].items(), key=lambda kv: kv[0][1]
            ):
                deduped = list(dict.fromkeys(lists))
                list_str = " · ".join(f"_{l}_" for l in deduped)
                st.markdown(f"- {flag} **{country}** — {list_str}")

        # ── Matched entities (one expander per unique entity) ─────────────
        st.markdown("### Matched entities")
        # Identifier-type labels for prettier display
        PROP_LABEL = {"bikCode": "BIK", "swiftBic": "SWIFT", "innCode": "INN"}
        # Order: OFAC-listed entities first so the most severe matches read first
        sorted_entity_ids = sorted(
            hits_by_id.keys(),
            key=lambda eid: 0 if eid in ofac_entity_ids else 1,
        )
        for eid in sorted_entity_ids:
            hit = hits_by_id[eid]
            entity = hit["entity"]
            matches = hit["matches"]
            match_summary = " · ".join(
                f"{PROP_LABEL.get(prop, prop)} `{val}`" for prop, val in matches
            )
            # Per-entity colour: 🔴 if this entity itself has OFAC, 🟡 otherwise
            entity_emoji = "🔴" if eid in ofac_entity_ids else "🟡"
            ofac_tag = " · 🇺🇸 OFAC" if eid in ofac_entity_ids else ""
            with st.expander(
                f"{entity_emoji} {entity.get('caption', '(no caption)')} · "
                f"matched by {match_summary}{ofac_tag}"
            ):
                cols = st.columns([1, 2])
                with cols[0]:
                    st.write("**OpenSanctions ID:**", eid)
                    st.write("**Schema:**", entity.get("schema"))
                    st.write("**Matched on:**")
                    for prop, val in matches:
                        st.write(f"- `{PROP_LABEL.get(prop, prop)}` = `{val}`")
                    if entity.get("datasets"):
                        st.write("**Datasets:**")
                        for d in entity["datasets"]:
                            st.write(f"- `{d}`")
                    st.markdown(
                        f"[Open in OpenSanctions ↗]"
                        f"(https://www.opensanctions.org/entities/{eid}/)"
                    )
                with cols[1]:
                    # Per-entity sanctioning jurisdictions
                    e_buckets = categorize_datasets(entity.get("datasets") or [])
                    if e_buckets["sanctions"]:
                        st.markdown("**🌍 Sanctioning jurisdictions for this entity:**")
                        for (flag, country), lists in sorted(
                            e_buckets["sanctions"].items(), key=lambda kv: kv[0][1]
                        ):
                            deduped = list(dict.fromkeys(lists))
                            list_str = " · ".join(f"_{l}_" for l in deduped)
                            st.markdown(f"- {flag} **{country}** — {list_str}")
                    # Topics (PEP, sanction, crime, etc.)
                    if entity.get("topics"):
                        st.write("**Topics:**", ", ".join(f"`{t}`" for t in entity["topics"]))
                    # Show key entity properties
                    props = entity.get("properties", {}) or {}
                    if props.get("name"):
                        st.write("**Names:**", ", ".join(props["name"][:5]))
                    if props.get("country"):
                        st.write("**Country:**", ", ".join(props["country"]))
                    if props.get("address"):
                        st.write("**Addresses:**", "; ".join(props["address"][:3]))
                    if props.get("innCode"):
                        st.write("**INN(s) on file:**", ", ".join(f"`{i}`" for i in props["innCode"]))
                    if props.get("bikCode"):
                        st.write("**BIK(s) on file:**", ", ".join(f"`{b}`" for b in props["bikCode"]))
                    if props.get("swiftBic"):
                        st.write("**SWIFT(s) on file:**", ", ".join(f"`{s}`" for s in props["swiftBic"]))
    else:
        # No surviving hits. Distinguish two sub-cases:
        # (a) There were no strict identifier hits at all → 🟢 CLEAR
        # (b) There were hits but all got OS-whitelisted via swiftBic → ✅
        if os_whitelisted_entities:
            # ─── ✅ WHITELISTED (via OS-SWIFT post-filter) ──────────────
            wl_lines = "\n".join(
                f"- _{h['entity'].get('caption', '(no caption)')}_ "
                f"(OS-stored SWIFT: "
                f"{', '.join(f'`{s}`' for s in h['matched_whitelist_swifts'])})"
                for h in os_whitelisted_entities.values()
            )
            st.success(
                "## ✅ WHITELISTED\n\n"
                f"OpenSanctions returned **{len(os_whitelisted_entities)} entity(ies)** "
                "matching this bank on strict identifier, **but every one of them "
                "has an OS-stored SWIFT that's on the OhMySwift whitelist** — i.e. "
                "the matched legal entities are themselves on the curated list of "
                "Russian banks not under US (SDN) and EU sanctions.\n\n"
                f"{wl_lines}\n\n"
                "_The strict-search hits here are most likely OpenSanctions' own "
                "Russian bank reference records (`ru_cbr_banks` dataset), not "
                "sanctions designations._"
            )
        else:
            # ─── 🟢 CLEAR ───────────────────────────────────────────────
            st.success(
                "## 🟢 CLEAR\n\n"
                "**No OpenSanctions entity matches this bank on a strict identifier** "
                "(BIK, SWIFT, or INN). The bank is not currently indexed in OpenSanctions "
                "under any of its resolved identifiers.\n\n"
                "_Caveat: this verdict is based on exact identifier matching. A bank "
                "could in theory be sanctioned under a name/address only if OpenSanctions "
                "hasn't yet attached structured identifiers to the entity — uncommon but "
                "possible for newly-listed entities before metadata enrichment. For "
                "elevated-risk transactions, also check the bank's name directly against "
                "current sanctions list updates._"
            )

# ── Raw payload (debug) ──────────────────────────────────────────────────
with st.expander("🔍 Raw payload (debug)"):
    debug_payload: dict[str, Any] = {
        "cbr": cbr,
        "bik_info": bi,
        "identifiers_screened": {
            "bics": list(all_bics),
            "swifts": list(all_swifts),
            "inn": inn_final,
            "ogrn": ogrn_final,
            "kpps": list(all_kpps),
        },
    }
    if whitelist_hits:
        debug_payload["whitelist_hits"] = whitelist_hits
        debug_payload["opensanctions_strict_screening"] = "skipped (whitelisted via input SWIFT)"
    else:
        debug_payload["opensanctions_strict_screening"] = screening
        if os_whitelisted_entities:
            debug_payload["os_whitelisted_entities"] = {
                eid: {
                    "caption": h["entity"].get("caption"),
                    "matched_whitelist_swifts": h["matched_whitelist_swifts"],
                    "datasets": h["entity"].get("datasets"),
                }
                for eid, h in os_whitelisted_entities.items()
            }
    st.json(debug_payload)
