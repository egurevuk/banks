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


def bic_to_internal_code(bic: str) -> int | None:
    """Call BicToIntCode and return the internal CBR credit-org code."""
    body = (
        f'<BicToIntCode xmlns="http://web.cbr.ru/">'
        f"<BicCode>{bic}</BicCode>"
        f"</BicToIntCode>"
    )
    root = _soap_call("BicToIntCode", body)
    el = root.find(".//w:BicToIntCodeResult", CBR_NS)
    if el is None or not el.text or float(el.text) == 0:
        return None
    return int(float(el.text))


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


def cbr_lookup(bic: str) -> dict[str, Any]:
    """Full Step-1 chain. Returns {internal_code, swift_codes, names, ...} or {error}."""
    try:
        int_code = bic_to_internal_code(bic)
    except Exception as e:
        return {"error": f"CBR BicToIntCode failed: {e}"}
    if int_code is None:
        return {"error": f"BIC {bic} not found in CBR registry"}

    try:
        credit_info = internal_code_to_credit_info(int_code)
    except Exception as e:
        return {
            "internal_code": int_code,
            "error": f"CBR CreditInfoByIntCodeExXML failed: {e}",
        }
    if credit_info is None:
        return {"internal_code": int_code, "swift_codes": [], "warning": "Empty CO record"}

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
        "bik_codes": all_bics,                          # NEW: all BICs (head + branches)
        "swift_codes": swifts,
        "primary_swift": swifts[0] if swifts else None,
        "short_names_cbr": short_names,                 # NEW: list
        "full_names_cbr": full_names,                   # NEW: list
        "inn_cbr": inns[0] if inns else None,
        "ogrn_cbr": ogrns[0] if ogrns else None,
        "regn_cbr": regns[0] if regns else None,
        "kpp_codes_cbr": kpps,                          # NEW: list
        "addresses_cbr": addresses,                     # NEW: list
        "raw_xml": raw_xml,                             # NEW: debug
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


def verdict_for_score(score: float) -> tuple[str, str]:
    """Return (label, color_class) for a match score."""
    if score >= SCORE_HIT:
        return "MATCH", "🔴"
    if score >= SCORE_REVIEW:
        return "REVIEW", "🟡"
    return "LIKELY CLEAR", "🟢"


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
        "**Score thresholds**  \n"
        f"🔴 Match: ≥ {SCORE_HIT:.2f}  \n"
        f"🟡 Review: ≥ {SCORE_REVIEW:.2f}  \n"
        "🟢 Likely clear: below"
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

# ─── STEP 1 ──────────────────────────────────────────────────────────────────
with st.status("Step 1 · CBR: resolve BIC → internal code → SWIFT", expanded=True) as status:
    cbr = cbr_lookup(bic)
    if cbr.get("error"):
        st.error(cbr["error"])
        status.update(label=f"Step 1 · {cbr['error']}", state="error")
        st.stop()

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
        # Flag if the input BIC isn't the head office — relevant for sanctions screening
        if bic in cbr["bik_codes"] and len(cbr["bik_codes"]) > 1:
            other_bics = [b for b in cbr["bik_codes"] if b != bic]
            st.info(
                f"`{bic}` is one of {len(cbr['bik_codes'])} BICs registered for this credit organization. "
                f"OpenSanctions typically keys on the head-office BIC — sending all of them ensures the match lands."
            )

    with st.expander("Raw CBR XML response (debug)"):
        st.code(cbr.get("raw_xml") or "(empty)", language="xml")
    status.update(label="Step 1 · CBR resolved", state="complete")

# ─── STEP 2 ──────────────────────────────────────────────────────────────────
with st.status("Step 2 · bik-info.ru: bank name & address + transliteration", expanded=True) as status:
    bi = bik_info_lookup(bic)
    if bi.get("error"):
        st.error(bi["error"])
        status.update(label=f"Step 2 · {bi['error']}", state="error")
        # don't stop — we may still screen on what we have from Step 1
        bi = {}

    cbr_short = (cbr.get("short_names_cbr") or [None])[0]
    cbr_full = (cbr.get("full_names_cbr") or [None])[0]
    cbr_addr_primary = (cbr.get("addresses_cbr") or [None])[0]

    name_ru = bi.get("name") or bi.get("namemini") or cbr_short or ""
    short_name_ru = bi.get("namemini") or ""
    address_ru = bi.get("address") or cbr_addr_primary or ""
    inn = bi.get("inn")
    kpp = bi.get("kpp")
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
    status.update(label="Step 2 · Bank identified", state="complete")

# ─── STEP 3 ──────────────────────────────────────────────────────────────────
with st.status("Step 3 · OpenSanctions /match — screening", expanded=True) as status:
    # Pull every name variant from both sources, RU + transliterated
    name_pool: set[str] = set()
    name_pool.update([name_ru, short_name_ru, name_en, short_name_en])
    for n in (cbr.get("short_names_cbr") or []) + (cbr.get("full_names_cbr") or []):
        name_pool.add(n)
        name_pool.add(transliterate_ru(n))
    names = [n for n in name_pool if n]

    # Every address variant from both sources, RU + transliterated
    addr_pool: set[str] = set()
    addr_pool.update([address_ru, address_en])
    for a in (cbr.get("addresses_cbr") or []):
        addr_pool.add(a)
        addr_pool.add(transliterate_ru(a))
    addresses = [a for a in addr_pool if a]

    # Prefer bik-info.ru's INN/OGRN, fall back to CBR XML
    inn_final = inn or cbr.get("inn_cbr")
    ogrn_final = bi.get("ogrn") or cbr.get("ogrn_cbr")
    reg_final = cbr.get("regn_cbr")  # bank license number (e.g. "2673")

    # Pass ALL BIKs and KPPs (head office + branches) — OpenSanctions typically
    # only stores the head-office BIK, so the user's input branch BIC alone
    # would miss the match.
    all_bics = cbr.get("bik_codes") or [bic]
    if bic not in all_bics:
        all_bics = [bic] + all_bics
    all_kpps = cbr.get("kpp_codes_cbr") or ([kpp] if kpp else [])

    os_result = opensanctions_match(
        OPENSANCTIONS_API_KEY,
        names=names,
        addresses=addresses,
        swift_codes=cbr.get("swift_codes", []),
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

if not results:
    st.success("🟢 **No matches in OpenSanctions** — no candidate entities returned for this bank.")
else:
    top_score = max((r.get("score") or 0) for r in results)
    label, emoji = verdict_for_score(top_score)
    if label == "MATCH":
        st.error(f"{emoji} **{label}** — top score {top_score:.2f}. Investigate before transacting.")
    elif label == "REVIEW":
        st.warning(f"{emoji} **{label}** — top score {top_score:.2f}. Manual review recommended.")
    else:
        st.success(f"{emoji} **{label}** — top score {top_score:.2f}. Likely false positive.")

    st.markdown("### Candidates")
    for r in sorted(results, key=lambda x: x.get("score") or 0, reverse=True):
        score = r.get("score") or 0
        _, emoji = verdict_for_score(score)
        with st.expander(f"{emoji}  {r.get('caption', '(no caption)')}  ·  score {score:.3f}"):
            cols = st.columns([1, 2])
            with cols[0]:
                st.write("**ID:**", r.get("id"))
                st.write("**Schema:**", r.get("schema"))
                st.write("**Score:**", f"{score:.4f}")
                st.write("**Match?**", r.get("match"))
                if r.get("datasets"):
                    st.write("**Datasets:**")
                    for d in r["datasets"]:
                        st.write(f"- `{d}`")
            with cols[1]:
                props = r.get("properties", {})
                relevant_keys = [
                    "name", "alias", "address", "country", "jurisdiction",
                    "registrationNumber", "swiftBic", "innCode", "ogrnCode",
                    "topics", "program", "sanctions",
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
