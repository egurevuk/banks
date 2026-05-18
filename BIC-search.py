"""
BIK Sanctions Screener
======================

A minimal Streamlit app that:

1. Takes a Russian BIK (банковский идентификационный код) as input.
2. Looks up the bank's name via bik-info.ru's JSON API.
3. Searches that name on OpenSanctions.
4. Shows whether the bank is sanctioned, by which countries, with a link
   to the canonical OpenSanctions entity page.
"""
from __future__ import annotations

import re
from typing import Any

import requests
import streamlit as st


BIK_INFO_API = "https://bik-info.ru/api.html"
OPENSANCTIONS_API = "https://api.opensanctions.org"

# Browser-like headers — bik-info.ru's CDN sometimes 403s default UAs
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9,ru;q=0.8",
}

# Country code → display name (only the sanctioning authorities we expect)
COUNTRY_NAMES = {
    "us": "🇺🇸 United States",
    "eu": "🇪🇺 European Union",
    "gb": "🇬🇧 United Kingdom",
    "ua": "🇺🇦 Ukraine",
    "ca": "🇨🇦 Canada",
    "ch": "🇨🇭 Switzerland",
    "jp": "🇯🇵 Japan",
    "au": "🇦🇺 Australia",
    "nz": "🇳🇿 New Zealand",
    "fr": "🇫🇷 France",
    "de": "🇩🇪 Germany",
    "pl": "🇵🇱 Poland",
    "tw": "🇹🇼 Taiwan",
    "kr": "🇰🇷 South Korea",
    "sg": "🇸🇬 Singapore",
    "ru": "🇷🇺 Russia",
}

# Substrings that mark a dataset name as a sanctions list (not a registry,
# not a company database). A dataset only counts as "sanctioning country X"
# if its name contains one of these markers AND starts with a country prefix.
# This prevents ``ru_cbr_banks`` (the Central Bank's bank registry) from
# being misread as "Russia sanctioned this bank".
SANCTIONS_DATASET_MARKERS = (
    "ofac", "sdn", "fsf", "csl", "hmt", "seco", "dfat", "nsdc",
    "sanction", "designated", "freez", "consolidated",
    "mof_sanctions", "ws_sanctions",
)

# Map dataset name prefixes to country codes. OpenSanctions dataset names
# encode the sanctioning authority — e.g. "us_ofac_sdn" → US, "eu_fsf" → EU,
# "gb_hmt_sanctions" → UK, "ua_nsdc_sanctions" → Ukraine.
DATASET_COUNTRY_PREFIX = {
    "us_": "us",
    "eu_": "eu",
    "gb_": "gb",
    "uk_": "gb",
    "ua_": "ua",
    "ca_": "ca",
    "ch_": "ch",
    "jp_": "jp",
    "au_": "au",
    "nz_": "nz",
    "fr_": "fr",
    "de_": "de",
    "pl_": "pl",
    "tw_": "tw",
    "kr_": "kr",
    "sg_": "sg",
}


# ---------------------------------------------------------------------------
# Step 1 — bik-info.ru lookup
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=3600)
def lookup_bik(bik: str) -> dict[str, Any]:
    """Fetch bank info for a BIK from bik-info.ru.

    Returns a dict with at least ``{"ok": bool, "name": str | None,
    "raw": dict | None, "error": str | None}``. The ``raw`` field carries
    the unmodified API response so we can show all available fields in a
    "More details" expander.
    """
    try:
        resp = requests.get(
            BIK_INFO_API,
            params={"type": "json", "bik": bik},
            headers=BROWSER_HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        return {"ok": False, "name": None, "raw": None,
                "error": f"bik-info.ru request failed: {exc}"}

    try:
        data = resp.json()
    except ValueError:
        return {"ok": False, "name": None, "raw": None,
                "error": "bik-info.ru returned non-JSON"}

    if not isinstance(data, dict):
        return {"ok": False, "name": None, "raw": None,
                "error": "bik-info.ru returned an unexpected shape"}

    # API returns {"error": "BIK not found"} when the BIK isn't in the
    # registry. Otherwise it returns the bank record as a flat dict.
    if "error" in data and len(data) <= 2:
        return {"ok": False, "name": None, "raw": data,
                "error": str(data.get("error", "BIK not found"))}

    # The API uses Russian-ish field names; we accept whichever turns up.
    name = (
        data.get("namebank")
        or data.get("name")
        or data.get("namefull")
        or data.get("bank")
        or data.get("name_bank")
    )
    if not name:
        return {"ok": False, "name": None, "raw": data,
                "error": "Couldn't find a bank name in the response"}

    return {"ok": True, "name": str(name).strip(), "raw": data, "error": None}


# ---------------------------------------------------------------------------
# Step 2 — OpenSanctions search by name
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=3600)
def search_opensanctions(name: str, api_key: str | None = None) -> dict[str, Any]:
    """Search OpenSanctions for a bank by name.

    Uses POST ``/match/sanctions`` with a Company-schema query carrying
    just the name and ``country=ru`` — this is OpenSanctions' recommended
    endpoint for screening and gives logic-v2 fuzzy-match scores rather
    than the brittle text-search of ``/search``. Returns the parsed JSON
    response or an error dict.
    """
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"ApiKey {api_key}"

    payload = {
        "queries": {
            "q1": {
                "schema": "Company",
                "properties": {
                    "name":         [name],
                    "country":      ["ru"],
                    "jurisdiction": ["Russia"],
                },
            }
        }
    }
    try:
        resp = requests.post(
            f"{OPENSANCTIONS_API}/match/sanctions",
            json=payload,
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        return {"ok": False, "error": f"OpenSanctions request failed: {exc}",
                "results": []}

    try:
        data = resp.json()
    except ValueError:
        return {"ok": False, "error": "OpenSanctions returned non-JSON",
                "results": []}

    results = (
        data.get("responses", {})
        .get("q1", {})
        .get("results", [])
    ) or []
    return {"ok": True, "error": None, "results": results}


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_full_entity(entity_id: str, api_key: str | None = None) -> dict | None:
    """Hydrate an entity ID into the full record including all sanctions."""
    headers = {}
    if api_key:
        headers["Authorization"] = f"ApiKey {api_key}"
    try:
        resp = requests.get(
            f"{OPENSANCTIONS_API}/entities/{entity_id}",
            params={"nested": "true"},
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        return None


# ---------------------------------------------------------------------------
# Sanctions extraction
# ---------------------------------------------------------------------------

def extract_sanction_countries(entity: dict) -> list[str]:
    """Return a sorted list of country display labels for the entity.

    Pulls country codes from two sources:
    * Each ``Sanction`` entity nested under ``properties.sanctions`` —
      these are the formal designations and carry ``country`` codes.
    * Dataset names (``us_ofac_sdn``, ``eu_fsf``, ``gb_hmt_sanctions``,
      ``ua_nsdc_sanctions``, ...) — the prefix maps to the sanctioning
      authority's country.
    """
    codes: set[str] = set()
    properties = entity.get("properties", {}) or {}

    # 1) nested Sanction entities
    for sanction in properties.get("sanctions", []) or []:
        if not isinstance(sanction, dict):
            continue
        sp = sanction.get("properties", {}) or {}
        for c in sp.get("country", []) or []:
            if isinstance(c, str):
                codes.add(c.lower())
        # Authority might encode country in dataset-style strings
        for ds in (sanction.get("datasets") or []):
            country = _country_from_dataset(ds)
            if country:
                codes.add(country)

    # 2) entity's own datasets
    for ds in entity.get("datasets", []) or []:
        country = _country_from_dataset(ds)
        if country:
            codes.add(country)

    return sorted(COUNTRY_NAMES.get(c, c.upper()) for c in codes)


def _country_from_dataset(dataset_name: str) -> str | None:
    """Return the sanctioning country code for an OpenSanctions dataset
    name, or None if the dataset isn't a sanctions list.

    A dataset only counts as sanctioning if its name both:
    1. Contains a sanctions marker (ofac, fsf, sanction, sdn, csl, hmt, ...)
    2. Starts with a country prefix we recognize

    This is intentionally conservative — false positives here would taint
    the verdict. Reference datasets like ``ru_cbr_banks`` or ``de_handelsregister``
    correctly return None.
    """
    if not dataset_name:
        return None
    name = dataset_name.lower()
    if not any(marker in name for marker in SANCTIONS_DATASET_MARKERS):
        return None
    for prefix, country in DATASET_COUNTRY_PREFIX.items():
        if name.startswith(prefix):
            return country
    return None


def is_sanctioned(entity: dict) -> bool:
    """Is this entity actually on a sanctions list?

    We require BOTH a ``sanction`` topic AND a non-empty list of
    sanction-related designations or sanctions-encoding datasets. Without
    both, the entity is "of interest" but not actually sanctioned.
    """
    topics = entity.get("properties", {}).get("topics") or []
    if isinstance(topics, list) and any(
        isinstance(t, str) and t.startswith("sanction") and t != "sanction.linked"
        for t in topics
    ):
        return True
    # Fallback: any dataset that maps to a sanctioning country
    for ds in entity.get("datasets", []) or []:
        if _country_from_dataset(ds):
            return True
    return False


def _topics_for_entity(entity: dict) -> list[str]:
    return list(entity.get("properties", {}).get("topics") or [])


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def normalize_bik(raw: str) -> str:
    return re.sub(r"\D", "", raw or "")[:9]


def is_valid_bik(bik: str) -> bool:
    return bool(bik) and bik.isdigit() and len(bik) == 9


def screen(bik: str, api_key: str | None) -> dict[str, Any]:
    """Full pipeline: bik-info → OpenSanctions /match → enriched entity."""
    info = lookup_bik(bik)
    if not info["ok"]:
        return {"step": "lookup", "bik": bik, **info}

    name = info["name"]
    match = search_opensanctions(name, api_key=api_key)
    if not match["ok"]:
        return {"step": "match", "bik": bik, "name": name,
                "raw_bik_info": info["raw"], **match}

    results = match["results"]
    # Pick the best confirmed match, if any. logic-v2 sets match=True only
    # when the candidate clears its similarity threshold.
    confirmed = [r for r in results if r.get("match")]
    candidate = confirmed[0] if confirmed else (results[0] if results else None)

    enriched = None
    if candidate and candidate.get("id"):
        enriched = fetch_full_entity(candidate["id"], api_key=api_key)

    return {
        "step": "ok",
        "ok": True,
        "bik": bik,
        "name": name,
        "raw_bik_info": info["raw"],
        "results": results,
        "candidate": candidate,
        "enriched": enriched,
    }


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="BIK Sanctions Screener",
    page_icon="🏦",
    layout="centered",
)

st.title("🏦 BIK Sanctions Screener")
st.markdown(
    "Enter a Russian **BIK** (9-digit банковский идентификационный код). "
    "The app looks up the bank's name via [bik-info.ru](https://bik-info.ru) "
    "and screens it against [OpenSanctions](https://www.opensanctions.org)."
)

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input(
        "OpenSanctions API key (optional)",
        type="password",
        help=(
            "Anonymous access works for low traffic but is rate-limited. "
            "Get a free key at opensanctions.org for higher quotas."
        ),
    )
    st.markdown("---")
    st.markdown(
        "**Pipeline**\n\n"
        "1. `bik-info.ru/api.html?type=json&bik={BIK}` → bank name\n"
        "2. `api.opensanctions.org/match/sanctions` → name → sanctions hit"
    )

bik_input = st.text_input("BIK", placeholder="e.g. 044525225", max_chars=20)
run = st.button("Screen", type="primary")

if run:
    bik = normalize_bik(bik_input)
    if not is_valid_bik(bik):
        st.error("BIK must be 9 digits (numbers only).")
        st.stop()

    with st.spinner(f"Looking up BIK {bik}..."):
        result = screen(bik, api_key=api_key or None)

    if result["step"] == "lookup":
        st.error(f"❌ {result['error']}")
        st.stop()

    if result["step"] == "match":
        st.warning(f"Found bank: **{result['name']}**")
        st.error(f"❌ {result['error']}")
        st.stop()

    # Got here → bank resolved + OpenSanctions queried
    st.markdown("---")
    bank_name = result["name"]
    candidate = result["candidate"]
    enriched = result["enriched"]

    if not candidate:
        st.success("✅ **CLEAR** — no OpenSanctions match")
        st.markdown(f"**Bank:** {bank_name}")
        st.caption(
            "No entity on any sanctions list matched this bank's name. "
            "OpenSanctions covers US OFAC, EU, UK, Canada, Switzerland, "
            "Japan, Australia, Ukraine, and more."
        )
    else:
        # We have a candidate. Was it confirmed?
        confirmed = bool(candidate.get("match"))
        target_entity = enriched or candidate
        sanctioned = is_sanctioned(target_entity)

        if sanctioned and confirmed:
            st.markdown(
                '<div style="background:#dc2626;color:white;padding:12px 16px;'
                'border-radius:8px;font-weight:600;font-size:1.1rem;'
                'display:inline-block;">❌ SANCTIONED</div>',
                unsafe_allow_html=True,
            )
        elif sanctioned and not confirmed:
            st.markdown(
                '<div style="background:#f59e0b;color:white;padding:12px 16px;'
                'border-radius:8px;font-weight:600;font-size:1.1rem;'
                'display:inline-block;">⚠️ POSSIBLE MATCH — REVIEW</div>',
                unsafe_allow_html=True,
            )
            st.caption(
                "OpenSanctions returned a sanctioned entity with a name "
                "similar to this bank, but the automatic matcher wasn't "
                "confident. Inspect the candidate below."
            )
        else:
            st.success("✅ **CLEAR** — best match isn't on a sanctions list")

        st.markdown(f"**Bank (from bik-info.ru):** {bank_name}")
        st.markdown(
            f"**Matched OpenSanctions entity:** "
            f"{target_entity.get('caption', '—')}"
        )
        score = candidate.get("score")
        if score is not None:
            st.caption(
                f"Match score: {score:.2f} "
                f"({'confirmed' if confirmed else 'below confidence threshold'})"
            )

        if sanctioned:
            countries = extract_sanction_countries(target_entity)
            if countries:
                st.markdown("### Sanctioning jurisdictions")
                for c in countries:
                    st.markdown(f"- {c}")

        entity_id = target_entity.get("id") or candidate.get("id")
        if entity_id:
            os_url = f"https://www.opensanctions.org/entities/{entity_id}/"
            st.markdown(
                f"### Full record\n"
                f"[Open on OpenSanctions ↗]({os_url})"
            )

    # --- Diagnostic details -------------------------------------------------
    with st.expander("Raw bik-info.ru response"):
        st.json(result["raw_bik_info"])

    if result.get("results"):
        with st.expander(f"All OpenSanctions candidates ({len(result['results'])})"):
            rows = []
            for r in result["results"]:
                rows.append({
                    "Match": "✅" if r.get("match") else "—",
                    "Score": f"{r.get('score', 0):.2f}",
                    "Entity": r.get("caption", ""),
                    "ID": r.get("id", ""),
                })
            st.table(rows)
