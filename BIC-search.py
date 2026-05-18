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
        return {"ok": False, "name": None, "swift": None, "raw": None,
                "error": f"bik-info.ru request failed: {exc}"}

    try:
        data = resp.json()
    except ValueError:
        return {"ok": False, "name": None, "swift": None, "raw": None,
                "error": "bik-info.ru returned non-JSON"}

    if not isinstance(data, dict):
        return {"ok": False, "name": None, "swift": None, "raw": None,
                "error": "bik-info.ru returned an unexpected shape"}

    # API returns {"error": "BIK not found"} when the BIK isn't in the
    # registry. Otherwise it returns the bank record as a flat dict.
    if "error" in data and len(data) <= 2:
        return {"ok": False, "name": None, "swift": None, "raw": data,
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
        return {"ok": False, "name": None, "swift": None, "raw": data,
                "error": "Couldn't find a bank name in the response"}

    swift = (
        data.get("swift")
        or data.get("swiftbic")
        or data.get("swift_code")
        or data.get("bic")
        or data.get("code_swift")
    )
    swift = str(swift).strip().upper() if swift else None

    return {"ok": True, "name": str(name).strip(), "swift": swift,
            "raw": data, "error": None}


# ---------------------------------------------------------------------------
# Cyrillic → Latin transliteration
# ---------------------------------------------------------------------------
#
# bik-info.ru returns bank names in Cyrillic (e.g. "ПАО СБЕРБАНК"), but
# OpenSanctions stores most Russian-bank canonicals in English form
# (e.g. "Joint Stock Company Sberbank"). OpenSanctions' logic-v2 matcher
# does fuzzy comparison but performs much better when the candidate name
# is also in Latin script — so we ship both the original Cyrillic name
# AND a transliteration to /match.
#
# We use BGN/PCGN-style mapping (the scheme used in most international
# sanctions documents). It isn't perfect for every name — e.g. "Ц" → "Ts"
# whereas Center-Invest brands itself "Center-Invest" not "Tsentr-Invest" —
# but /match's fuzzy scoring handles the residual mismatch via token
# overlap on the parts that DO transliterate cleanly (BANK, SBERBANK,
# INVEST, etc.).

_CYR_TO_LAT_LOWER: dict[str, str] = {
    "а": "a",  "б": "b",  "в": "v",  "г": "g",  "д": "d",  "е": "e",
    "ё": "yo", "ж": "zh", "з": "z",  "и": "i",  "й": "y",  "к": "k",
    "л": "l",  "м": "m",  "н": "n",  "о": "o",  "п": "p",  "р": "r",
    "с": "s",  "т": "t",  "у": "u",  "ф": "f",  "х": "kh", "ц": "ts",
    "ч": "ch", "ш": "sh", "щ": "shch", "ъ": "", "ы": "y", "ь": "",
    "э": "e",  "ю": "yu", "я": "ya",
}

# Common legal-form prefixes worth stripping when we generate name
# variants — these tokens carry no entity-identifying value and only
# dilute the name match.
_LEGAL_FORMS_CYR = ("ПАО ", "АО ", "ООО ", "ОАО ", "ЗАО ", "АКБ ", "КБ ",
                    "НКО ", "РНКО ", "ИКБ ", "ПУБЛИЧНОЕ АКЦИОНЕРНОЕ ОБЩЕСТВО ")
_LEGAL_FORMS_LAT = ("PAO ", "AO ", "OOO ", "OAO ", "ZAO ", "AKB ", "KB ",
                    "NKO ", "RNKO ", "IKB ", "PJSC ", "JSC ", "LLC ",
                    "PUBLIC JOINT STOCK COMPANY ")

# Map Cyrillic legal forms to their conventional English equivalents so
# we can add anglicized variants alongside the strict transliteration.
_LEGAL_FORM_ENGLISH = {
    "ПАО":  "PJSC",
    "ОАО":  "OJSC",
    "АО":   "JSC",
    "ЗАО":  "CJSC",
    "ООО":  "LLC",
    "АКБ":  "JSCB",   # joint-stock commercial bank
    "КБ":   "CB",     # commercial bank
    "НКО":  "NCO",    # non-bank credit organization
    "РНКО": "RNCO",
}


def transliterate(text: str) -> str:
    """Transliterate Cyrillic to Latin (BGN/PCGN-style), case-preserving."""
    if not text:
        return ""
    out: list[str] = []
    for ch in text:
        lower = ch.lower()
        if lower in _CYR_TO_LAT_LOWER:
            mapped = _CYR_TO_LAT_LOWER[lower]
            if ch.isupper() and mapped:
                # Uppercase the whole digraph for ALL-CAPS context.
                # We approximate this by using upper() — gives "SHCH" for "Щ",
                # which is what international docs use.
                mapped = mapped.upper()
            out.append(mapped)
        else:
            out.append(ch)
    return "".join(out)


def name_variants(name: str) -> list[str]:
    """Generate a deduplicated list of name forms for OpenSanctions /match.

    Includes:
    * The original name (verbatim).
    * BGN/PCGN transliteration of the original.
    * The original with leading legal-form prefix stripped (e.g. drops "ПАО ").
    * Transliteration of the stripped form.
    * Anglicized abbreviation variants (e.g. "ПАО СБЕРБАНК" → "PJSC SBERBANK").

    Order is preserved (best-match-first), with the original name first.
    """
    if not name:
        return []
    seen: set[str] = set()
    variants: list[str] = []

    def add(v: str) -> None:
        v = v.strip()
        if not v:
            return
        key = v.upper()
        if key in seen:
            return
        seen.add(key)
        variants.append(v)

    add(name)
    add(transliterate(name))

    upper = name.upper()
    # Strip a leading Cyrillic legal-form prefix
    for prefix in _LEGAL_FORMS_CYR:
        if upper.startswith(prefix):
            stripped = name[len(prefix):].strip()
            add(stripped)
            add(transliterate(stripped))
            # Anglicized legal-form prefix
            cyr_form = prefix.strip()
            english = _LEGAL_FORM_ENGLISH.get(cyr_form)
            if english:
                add(f"{english} {transliterate(stripped)}")
            break

    # Also handle the case where it starts with the Latin abbreviation
    # already (some banks return mixed-script names)
    for prefix in _LEGAL_FORMS_LAT:
        if upper.startswith(prefix):
            stripped = name[len(prefix):].strip()
            add(stripped)
            add(transliterate(stripped))
            break

    # Find a legal-form marker ANYWHERE in the name (not just the start).
    # Russian branch names commonly have the pattern
    # "<region descriptor> <legal form> <parent bank>", e.g.
    # "СЕВЕРО-ЗАПАДНЫЙ БАНК ПАО СБЕРБАНК" — the part after ПАО is the
    # parent we want to surface for matching.
    tokens = name.split()
    upper_tokens = [t.upper().strip(",.") for t in tokens]
    marker_set = {p.strip() for p in _LEGAL_FORMS_CYR} | \
                 {p.strip() for p in _LEGAL_FORMS_LAT}
    for i, tok in enumerate(upper_tokens):
        if tok in marker_set and i + 1 < len(tokens):
            tail = " ".join(tokens[i + 1:]).strip()
            if tail and tail.upper() != name.upper():
                add(tail)
                add(transliterate(tail))

    return variants


# ---------------------------------------------------------------------------
# Step 2 — OpenSanctions search by name
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=3600)
def search_opensanctions_by_bik_code(
    bik: str, name: str | None = None, api_key: str | None = None
) -> dict[str, Any]:
    """Search OpenSanctions by ``bikCode`` Company property.

    OpenSanctions stores Russian bank BIKs in the ``bikCode`` Company
    property (see e.g.
    https://www.opensanctions.org/statements/NK-eyGem2AhvVFkniDghv3Vm6/?prop=bikCode
    for T-Bank's ``044525974``). When the BIK is registered against a
    sanctioned entity, this gives us a near-1.0 exact match.

    The ``/match`` endpoint requires a ``name`` property — identifier-only
    queries return empty. We pass the bank name (and its variants) so
    logic-v2 has something to score, and the identifier match boosts the
    score on top of name similarity.
    """
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"ApiKey {api_key}"

    # /match requires name; if we don't have one we still try with a
    # placeholder so the request doesn't get rejected.
    names = name_variants(name) if name else ["Russian Bank"]

    payload = {
        "queries": {
            "q1": {
                "schema": "Company",
                "properties": {
                    "name":    names,
                    "bikCode": [bik],
                    "country": ["ru"],
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
        return {"ok": False, "error": f"OpenSanctions bikCode lookup failed: {exc}",
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
def search_opensanctions_by_swift_bic(
    swift: str, name: str | None = None, api_key: str | None = None
) -> dict[str, Any]:
    """Search OpenSanctions by ``swiftBic`` Organization property.

    OpenSanctions stores SWIFT/BIC codes (8-char primary form) on the
    ``swiftBic`` property. We strip the optional ``XXX`` branch suffix
    so ``TICSRUMMXXX`` and ``TICSRUMM`` compare equal.

    Like the bikCode lookup, we also include the bank name because
    ``/match`` requires it.
    """
    if not swift:
        return {"ok": True, "error": None, "results": [], "swift_query": None}

    # Strip optional branch suffix (XXX or X-padded)
    normalized = re.sub(r"X{1,3}$", "", swift.upper().strip())
    if not normalized:
        return {"ok": True, "error": None, "results": [], "swift_query": None}

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"ApiKey {api_key}"

    names = name_variants(name) if name else ["Russian Bank"]

    payload = {
        "queries": {
            "q1": {
                "schema": "Organization",
                "properties": {
                    "name":     names,
                    "swiftBic": [normalized],
                    "country":  ["ru"],
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
        return {"ok": False, "error": f"OpenSanctions swiftBic lookup failed: {exc}",
                "results": [], "swift_query": normalized}

    try:
        data = resp.json()
    except ValueError:
        return {"ok": False, "error": "OpenSanctions returned non-JSON",
                "results": [], "swift_query": normalized}

    results = (
        data.get("responses", {})
        .get("q1", {})
        .get("results", [])
    ) or []
    return {"ok": True, "error": None, "results": results,
            "swift_query": normalized}


@st.cache_data(show_spinner=False, ttl=3600)
def search_opensanctions_by_name(
    name: str, api_key: str | None = None
) -> dict[str, Any]:
    """Search OpenSanctions for a bank by name (multi-variant fuzzy match).

    We ship the original Cyrillic name AND its Latin transliteration AND
    legal-form-stripped variants to ``POST /match/sanctions``. This gives
    the logic-v2 matcher enough script + abbreviation coverage to find
    Russian-bank canonicals that are stored in English form (e.g.
    "Joint Stock Company Sberbank" for "ПАО СБЕРБАНК").
    """
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"ApiKey {api_key}"

    variants = name_variants(name)
    payload = {
        "queries": {
            "q1": {
                "schema": "Company",
                "properties": {
                    "name":         variants,
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
                "results": [], "variants_used": variants}

    try:
        data = resp.json()
    except ValueError:
        return {"ok": False, "error": "OpenSanctions returned non-JSON",
                "results": [], "variants_used": variants}

    results = (
        data.get("responses", {})
        .get("q1", {})
        .get("results", [])
    ) or []
    return {"ok": True, "error": None, "results": results,
            "variants_used": variants}


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


def _has_sanction_signal(candidate: dict) -> bool:
    """Cheap pre-check: does this /match candidate look sanctioned?

    Checks two signals exposed by /match results:
    * ``properties.topics`` (the standard location) — OpenSanctions uses
      ``sanction``, ``sanction.linked``, ``role.pep``, ``crime``, etc.
    * ``datasets`` — top-level array of dataset names. We treat the
      candidate as sanctioned if any of its datasets is itself a
      sanctioning list (us_ofac_sdn, eu_fsf, gb_hmt_sanctions, …).

    Falls back to top-level ``topics`` for shape variance, though /match
    always nests it under ``properties``.
    """
    topics = (
        candidate.get("properties", {}).get("topics")
        or candidate.get("topics")
        or []
    )
    if any(isinstance(t, str) and t.startswith("sanction")
           and t != "sanction.linked" for t in topics):
        return True
    for ds in candidate.get("datasets") or []:
        if _country_from_dataset(ds):
            return True
    return False


# Score thresholds for the verdict ladder. logic-v2 returns scores in
# [0, 1]; the `match` flag is set when score clears its internal
# threshold (≈ 0.7). We use a generous review band because cross-script
# names rarely clear it cleanly — better to surface borderline hits for
# human review than silently dismiss them:
#   ≥ 0.70 → strong match (SANCTIONED — high confidence)
#   ≥ 0.20 → possible name match (REVIEW — verify on OpenSanctions)
#   <  0.20 → too weak to call
MATCH_STRONG = 0.70
MATCH_POSSIBLE = 0.20
# SWIFT codes are structured identifiers, so even a partial match
# (e.g. same bank prefix, different regional office) is signal:
#   ≥ 0.95 → exact (or near-exact) SWIFT → SANCTIONED
#   ≥ 0.50 → significant SWIFT overlap (same bank family) → REVIEW
SWIFT_MATCH_STRONG = 0.95
SWIFT_MATCH_REVIEW = 0.50


def _pick_best_candidate(results: list[dict]) -> dict | None:
    """From a list of /match candidates, return the one that best
    answers "is this bank sanctioned?".

    Strategy:
    1. Prefer the highest-scoring SANCTIONED candidate above
       MATCH_POSSIBLE — even if the matcher flagged it as match=False,
       a 0.6-score Sberbank hit on "СБЕРБАНК" beats a 0.95-score clean
       bank with a coincidentally similar name.
    2. If no sanctioned candidate clears MATCH_POSSIBLE, fall back to
       the highest-scoring overall candidate so the UI can still show
       what was looked at.
    """
    if not results:
        return None
    sanctioned = [r for r in results
                  if _has_sanction_signal(r)
                  and r.get("score", 0) >= MATCH_POSSIBLE]
    sanctioned.sort(key=lambda r: r.get("score", 0), reverse=True)
    if sanctioned:
        return sanctioned[0]
    # No usable sanctioned hit — return the top-overall for display
    return max(results, key=lambda r: r.get("score", 0))


def _pick_best_swift_candidate(results: list[dict]) -> dict | None:
    """Like ``_pick_best_candidate`` but using ``SWIFT_MATCH_REVIEW`` (0.50)
    as the floor, because SWIFT prefix-matching is more meaningful than
    cross-script name fuzziness."""
    if not results:
        return None
    sanctioned = [r for r in results
                  if _has_sanction_signal(r)
                  and r.get("score", 0) >= SWIFT_MATCH_REVIEW]
    sanctioned.sort(key=lambda r: r.get("score", 0), reverse=True)
    return sanctioned[0] if sanctioned else None


def screen(bik: str, api_key: str | None) -> dict[str, Any]:
    """Full pipeline.

    1. ``bik-info.ru`` → bank name + SWIFT (for display + downstream matching).
    2. ``OpenSanctions /match`` by ``bikCode`` property — exact ID match.
    3. ``OpenSanctions /match`` by ``swiftBic`` property — exact or
       prefix match against the SWIFT/BIC from step 1.
    4. ``OpenSanctions /match`` by name — Cyrillic→Latin transliteration
       and multi-variant fuzzy search.

    The strongest signal wins:
    * Exact bikCode hit on a sanctioned entity → ❌ SANCTIONED
    * SWIFT match ≥ 0.95 on sanctioned → ❌ SANCTIONED
    * SWIFT match ≥ 0.50 on sanctioned → ⚠️ REVIEW (verify on OpenSanctions)
    * Name match ≥ 0.70 on sanctioned → ❌ SANCTIONED
    * Name match ≥ 0.20 on sanctioned → ⚠️ REVIEW
    * Otherwise → ✅ CLEAR
    """
    info = lookup_bik(bik)
    if not info["ok"]:
        return {"step": "lookup", "bik": bik, **info}

    name = info["name"]
    swift = info.get("swift")
    api_errors: list[str] = []

    # ---- Step 2: bikCode lookup (exact identifier) -------------------
    bik_match = search_opensanctions_by_bik_code(bik, name=name, api_key=api_key)
    if not bik_match.get("ok"):
        api_errors.append(bik_match.get("error", "bikCode lookup failed"))
    bik_results = bik_match.get("results") or []
    bik_candidate = _pick_best_candidate(bik_results) if bik_results else None

    if bik_candidate and _has_sanction_signal(bik_candidate):
        # Exact BIK match against a sanctioned entity — definitive.
        enriched = (
            fetch_full_entity(bik_candidate["id"], api_key=api_key)
            if bik_candidate.get("id") else None
        )
        return {
            "step": "ok", "ok": True,
            "bik": bik, "name": name, "swift": swift,
            "raw_bik_info": info["raw"],
            "results": bik_results,
            "candidate": bik_candidate,
            "enriched": enriched,
            "matched_via": "bikCode",
            "variants_used": [],
            "swift_query": None,
            "api_errors": api_errors,
            "bik_results": bik_results,
            "swift_results": [],
        }

    # ---- Step 3: SWIFT-based search via swiftBic property ------------
    swift_results: list[dict] = []
    swift_query: str | None = None
    swift_candidate: dict | None = None
    if swift:
        swift_match = search_opensanctions_by_swift_bic(
            swift, name=name, api_key=api_key
        )
        if not swift_match.get("ok"):
            api_errors.append(swift_match.get("error", "swiftBic lookup failed"))
        swift_results = swift_match.get("results") or []
        swift_query = swift_match.get("swift_query")
        swift_candidate = _pick_best_swift_candidate(swift_results)

    # ---- Step 4: name-based fuzzy search -----------------------------
    name_match = search_opensanctions_by_name(name, api_key=api_key)
    if not name_match["ok"]:
        return {"step": "match", "bik": bik, "name": name,
                "swift": swift, "raw_bik_info": info["raw"],
                "api_errors": api_errors + [name_match.get("error", "")],
                **name_match}
    name_results = name_match["results"]
    variants_used = name_match.get("variants_used", [])
    name_candidate = _pick_best_candidate(name_results) if name_results else None

    # Pick the strongest signal across SWIFT and name paths.
    # SWIFT wins ties because it's a structured identifier — not subject
    # to translation/transliteration noise.
    candidate = None
    matched_via = None
    results_to_show = name_results
    if swift_candidate and _has_sanction_signal(swift_candidate):
        candidate = swift_candidate
        matched_via = "swiftBic"
        results_to_show = swift_results
    elif name_candidate and _has_sanction_signal(name_candidate):
        candidate = name_candidate
        matched_via = "name"
    elif name_candidate:
        # Best overall candidate even if not sanctioned (for display)
        candidate = name_candidate
        matched_via = "name"
    elif bik_candidate:
        # Clean bikCode hit (the bank exists but isn't sanctioned)
        candidate = bik_candidate
        matched_via = "bikCode"
        results_to_show = bik_results

    enriched = None
    if candidate and candidate.get("id"):
        enriched = fetch_full_entity(candidate["id"], api_key=api_key)

    return {
        "step":          "ok",
        "ok":            True,
        "bik":           bik,
        "name":          name,
        "swift":         swift,
        "raw_bik_info":  info["raw"],
        "results":       results_to_show,
        "candidate":     candidate,
        "enriched":      enriched,
        "matched_via":   matched_via,
        "variants_used": variants_used,
        "swift_query":   swift_query,
        "swift_results": swift_results,
        "bik_results":   bik_results,
        "api_errors":    api_errors,
    }


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def get_api_key() -> str | None:
    """Read the OpenSanctions API key from Streamlit secrets.

    Returns ``None`` when no secrets file is configured (local dev without
    ``.streamlit/secrets.toml``). The key is optional — anonymous access
    works at a lower rate limit.
    """
    try:
        return st.secrets.get("OPENSANCTIONS_API_KEY") or None
    except (FileNotFoundError, KeyError, AttributeError):
        # No secrets.toml on disk → anonymous mode
        return None


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

api_key = get_api_key()

with st.sidebar:
    st.header("Pipeline")
    st.markdown(
        "1. `bik-info.ru` → bank name + SWIFT\n"
        "2. OpenSanctions `/match` by `bikCode` (exact ID)\n"
        "3. OpenSanctions `/match` by `swiftBic` (≥ 50% → review)\n"
        "4. OpenSanctions `/match` by name (transliterated, ≥ 20% → review)"
    )
    st.markdown("---")
    st.caption(
        "🔑 Authenticated to OpenSanctions"
        if api_key else
        "🔓 Anonymous mode — set `OPENSANCTIONS_API_KEY` in Streamlit "
        "secrets for higher rate limits"
    )

bik_input = st.text_input("BIK", placeholder="e.g. 044525225", max_chars=20)
run = st.button("Screen", type="primary")

if run:
    bik = normalize_bik(bik_input)
    if not is_valid_bik(bik):
        st.error("BIK must be 9 digits (numbers only).")
        st.stop()

    with st.spinner(f"Looking up BIK {bik}..."):
        result = screen(bik, api_key=api_key)

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
            "No entity on any sanctions list matched this bank by BIK or name. "
            "OpenSanctions covers US OFAC, EU, UK, Canada, Switzerland, "
            "Japan, Australia, Ukraine, and more."
        )
    else:
        target_entity = enriched or candidate
        sanctioned = is_sanctioned(target_entity)
        score = candidate.get("score", 0.0)
        matched_via = result.get("matched_via")
        entity_id = target_entity.get("id") or candidate.get("id")
        os_url = (
            f"https://www.opensanctions.org/entities/{entity_id}/"
            if entity_id else None
        )

        # Exact bikCode hit on a sanctioned entity → definitive SANCTIONED
        # regardless of name-match score.
        if sanctioned and matched_via == "bikCode":
            st.markdown(
                '<div style="background:#dc2626;color:white;padding:12px 16px;'
                'border-radius:8px;font-weight:600;font-size:1.1rem;'
                'display:inline-block;">❌ SANCTIONED</div>',
                unsafe_allow_html=True,
            )
            st.caption(
                "Exact identifier match: this bank's BIK is recorded "
                "directly on a sanctioned entity in OpenSanctions."
            )
        elif sanctioned and matched_via == "swiftBic" and score >= SWIFT_MATCH_STRONG:
            st.markdown(
                '<div style="background:#dc2626;color:white;padding:12px 16px;'
                'border-radius:8px;font-weight:600;font-size:1.1rem;'
                'display:inline-block;">❌ SANCTIONED</div>',
                unsafe_allow_html=True,
            )
            st.caption(
                "Exact SWIFT/BIC match: this bank's SWIFT code is recorded "
                "directly on a sanctioned entity in OpenSanctions."
            )
        elif sanctioned and matched_via == "swiftBic" and score >= SWIFT_MATCH_REVIEW:
            st.markdown(
                '<div style="background:#f59e0b;color:white;padding:12px 16px;'
                'border-radius:8px;font-weight:600;font-size:1.1rem;'
                'display:inline-block;">⚠️ REVIEW NEEDED</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f"A sanctioned entity matched **{bank_name}** by SWIFT/BIC "
                f"at **{score:.0%}** — the bank code prefix overlaps with a "
                f"sanctioned bank's SWIFT (likely a regional office or "
                f"branch of the same banking group). Please verify."
            )
            if os_url:
                st.markdown(
                    f'<a href="{os_url}" target="_blank" '
                    f'style="display:inline-block;background:#2563eb;'
                    f'color:white;padding:10px 16px;border-radius:6px;'
                    f'text-decoration:none;font-weight:600;margin-top:8px;">'
                    f'🔍 Check the match on OpenSanctions ↗</a>',
                    unsafe_allow_html=True,
                )
        elif sanctioned and matched_via == "name" and score >= MATCH_STRONG:
            st.markdown(
                '<div style="background:#dc2626;color:white;padding:12px 16px;'
                'border-radius:8px;font-weight:600;font-size:1.1rem;'
                'display:inline-block;">❌ SANCTIONED</div>',
                unsafe_allow_html=True,
            )
        elif sanctioned and matched_via == "name" and score >= MATCH_POSSIBLE:
            st.markdown(
                '<div style="background:#f59e0b;color:white;padding:12px 16px;'
                'border-radius:8px;font-weight:600;font-size:1.1rem;'
                'display:inline-block;">⚠️ REVIEW NEEDED</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f"A sanctioned entity matched **{bank_name}** with a "
                f"**{score:.0%}** name similarity. Please verify the "
                f"match below before treating this bank as sanctioned."
            )
            if os_url:
                st.markdown(
                    f'<a href="{os_url}" target="_blank" '
                    f'style="display:inline-block;background:#2563eb;'
                    f'color:white;padding:10px 16px;border-radius:6px;'
                    f'text-decoration:none;font-weight:600;margin-top:8px;">'
                    f'🔍 Check the match on OpenSanctions ↗</a>',
                    unsafe_allow_html=True,
                )
        else:
            st.success("✅ **CLEAR** — best match isn't on a sanctions list")

        st.markdown("")  # spacer
        st.markdown(f"**Bank (from bik-info.ru):** {bank_name}")
        if result.get("swift"):
            st.markdown(f"**SWIFT / BIC:** `{result['swift']}`")
        st.markdown(
            f"**Matched OpenSanctions entity:** "
            f"{target_entity.get('caption', '—')}"
        )
        if matched_via == "bikCode":
            st.caption(
                f"Resolved via exact `bikCode` property match · "
                f"score: {score:.2f}"
            )
        elif matched_via == "swiftBic":
            st.caption(
                f"Resolved via `swiftBic` property match · "
                f"score: {score:.2f}  ({score:.0%})"
            )
        elif matched_via == "name":
            st.caption(
                f"Resolved via fuzzy name match (cross-script) · "
                f"score: {score:.2f}  ({score:.0%})"
            )

        if sanctioned and (
            matched_via == "bikCode"
            or (matched_via == "swiftBic" and score >= SWIFT_MATCH_REVIEW)
            or (matched_via == "name" and score >= MATCH_POSSIBLE)
        ):
            countries = extract_sanction_countries(target_entity)
            if countries:
                st.markdown("### Sanctioning jurisdictions")
                for c in countries:
                    st.markdown(f"- {c}")

        if os_url:
            st.markdown(
                f"### Full OpenSanctions record\n"
                f"[Open canonical entity page ↗]({os_url})"
            )

    # --- Surface any API errors prominently --------------------------------
    api_errors = result.get("api_errors") or []
    if api_errors:
        st.error(
            "⚠️ Some OpenSanctions lookups failed — verdict may be incomplete:"
        )
        for err in api_errors:
            st.markdown(f"- {err}")

    # --- Diagnostic details -------------------------------------------------
    with st.expander("Raw bik-info.ru response"):
        st.json(result["raw_bik_info"])

    if result.get("swift_query"):
        with st.expander(
            f"`swiftBic` lookup → {len(result.get('swift_results') or [])} "
            f"candidates  (query: `{result['swift_query']}`)"
        ):
            sresults = result.get("swift_results") or []
            if not sresults:
                st.markdown("_No candidates returned._")
            for r in sresults:
                topics = r.get("properties", {}).get("topics") or r.get("topics") or []
                st.markdown(
                    f"**{r.get('caption', '?')}** — score "
                    f"`{r.get('score', 0):.2f}` "
                    f"{'✅' if r.get('match') else '—'}  "
                    f"{'❌ sanctioned' if _has_sanction_signal(r) else '✓ clean'}"
                )
                st.caption(
                    f"ID: `{r.get('id')}` · topics: {topics} · "
                    f"datasets: {(r.get('datasets') or [])[:5]}"
                )

    if result.get("bik_results") is not None:
        bik_results = result.get("bik_results") or []
        with st.expander(
            f"`bikCode` lookup → {len(bik_results)} candidates"
        ):
            if not bik_results:
                st.markdown("_No candidates returned._")
            for r in bik_results:
                topics = r.get("properties", {}).get("topics") or r.get("topics") or []
                st.markdown(
                    f"**{r.get('caption', '?')}** — score "
                    f"`{r.get('score', 0):.2f}` "
                    f"{'✅' if r.get('match') else '—'}  "
                    f"{'❌ sanctioned' if _has_sanction_signal(r) else '✓ clean'}"
                )
                st.caption(
                    f"ID: `{r.get('id')}` · topics: {topics} · "
                    f"datasets: {(r.get('datasets') or [])[:5]}"
                )

    variants = result.get("variants_used") or []
    if variants:
        with st.expander(f"Name variants sent to OpenSanctions ({len(variants)})"):
            for v in variants:
                st.markdown(f"- `{v}`")

    if result.get("results"):
        with st.expander(f"All OpenSanctions candidates ({len(result['results'])})"):
            rows = []
            for r in result["results"]:
                rows.append({
                    "Match": "✅" if r.get("match") else "—",
                    "Score": f"{r.get('score', 0):.2f}",
                    "Sanctioned": "❌" if _has_sanction_signal(r) else "✓",
                    "Entity": r.get("caption", ""),
                    "ID": r.get("id", ""),
                })
            st.table(rows)
