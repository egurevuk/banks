"""
BIK Sanctions Screener
======================
Streamlit app that takes a Russian BIK (Bank Identification Code / БИК),
resolves the bank's official name via OpenSanctions' `ru_cbr_banks` dataset
(sourced from the Central Bank of Russia), and screens it against global
sanctions lists via the OpenSanctions /match API.

Author: built for Stape Online Ltd
"""

from __future__ import annotations

import os
import re
import json
import time
from datetime import datetime
from typing import Any

import pandas as pd
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OS_BASE = "https://api.opensanctions.org"
BANKS_DATASET = "ru_cbr_banks"          # CBR-sourced Russian banking registry
SANCTIONS_SCOPE = "sanctions"           # /match endpoint scope (sanctions only)
DEFAULT_SCOPE = "default"               # /match endpoint scope (everything)
ALGORITHM = "logic-v2"                  # current recommended scoring algo
REQUEST_TIMEOUT = 30
SANCTION_SCORE_THRESHOLD = 0.70         # below this we don't alert
STRONG_SCORE_THRESHOLD = 0.85           # above this we treat as confirmed

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_bik(raw: str) -> str:
    """Strip non-digits and left-pad to 9 digits.

    Russian BIKs are formally 9 digits, but operators frequently drop the
    leading zero (e.g. `44525700` -> `044525700`).
    """
    digits = re.sub(r"\D", "", str(raw or ""))
    if not digits:
        return ""
    if len(digits) == 8:
        digits = "0" + digits
    return digits


def is_valid_bik(bik: str) -> bool:
    return bool(re.fullmatch(r"\d{9}", bik))


def os_headers(api_key: str | None) -> dict:
    """Build Authorization header for OpenSanctions if a key is set."""
    if api_key:
        return {"Authorization": f"ApiKey {api_key}"}
    return {}


def get_api_key() -> str:
    """Return the OpenSanctions API key.

    Resolution order:
      1. ``st.secrets["OPENSANCTIONS_API_KEY"]`` — recommended.
         Stored in ``.streamlit/secrets.toml`` locally, or in the Streamlit
         Cloud "Secrets" panel in deployment. Never committed to git.
      2. ``OPENSANCTIONS_API_KEY`` environment variable — useful for
         Docker / CI / quick local runs.

    Returns an empty string if neither is set, so the caller can show a
    friendly message rather than crash.
    """
    # 1. Streamlit secrets. Accessing st.secrets when no secrets.toml exists
    #    raises StreamlitSecretNotFoundError; accessing a missing key raises
    #    KeyError. Both are non-fatal — we fall through to the env var.
    try:
        val = st.secrets.get("OPENSANCTIONS_API_KEY", "")
        if val:
            return str(val).strip()
    except Exception:
        pass
    # 2. Environment variable
    return os.environ.get("OPENSANCTIONS_API_KEY", "").strip()


# ---------------------------------------------------------------------------
# OpenSanctions API calls
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=3600)
def lookup_bank_in_cbr_registry(bik: str, api_key: str) -> dict | None:
    """Resolve a BIK to a bank entity in OpenSanctions' CBR banking registry.

    Uses the /search endpoint scoped to the `ru_cbr_banks` dataset. We pass
    the BIK as the query text — the search index covers identifiers,
    so a precise match is returned for any valid BIK.
    """
    url = f"{OS_BASE}/search/{BANKS_DATASET}"
    params = {"q": bik, "limit": 10}
    resp = requests.get(
        url, params=params, headers=os_headers(api_key), timeout=REQUEST_TIMEOUT
    )
    resp.raise_for_status()
    data = resp.json()
    results = data.get("results", [])

    # Prefer results where the BIK appears literally in identifiers/registrationNumber
    for r in results:
        props = r.get("properties", {}) or {}
        haystack = (
            (props.get("registrationNumber") or [])
            + (props.get("bikCode") or [])
            + (props.get("ogrnCode") or [])
            + (props.get("innCode") or [])
        )
        haystack = [str(x) for x in haystack]
        if any(bik in v or v in bik for v in haystack):
            return r

    # Fallback: highest-ranked result if anything was returned
    return results[0] if results else None


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_full_entity(entity_id: str, api_key: str) -> dict | None:
    """Get a nested entity record (with sanctions relationships) by ID."""
    url = f"{OS_BASE}/entities/{entity_id}"
    resp = requests.get(
        url, headers=os_headers(api_key), timeout=REQUEST_TIMEOUT
    )
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


@st.cache_data(show_spinner=False, ttl=3600)
def match_against_sanctions(
    bank_entity: dict, api_key: str, scope: str = SANCTIONS_SCOPE
) -> dict:
    """Run OpenSanctions /match using the bank entity as query-by-example.

    We pass every known identifier (BIK, INN, OGRN, etc.) plus every name
    variant, so the scoring algorithm has maximum signal. The endpoint
    returns up to 5 candidate matches each with a score.
    """
    url = f"{OS_BASE}/match/{scope}"
    props = (bank_entity or {}).get("properties", {}) or {}

    # Pick out the properties that make sense to forward
    query_props: dict[str, list[str]] = {}
    for key in (
        "name", "alias", "previousName", "weakAlias",
        "innCode", "ogrnCode", "registrationNumber",
        "taxNumber", "swiftBic", "bikCode",
        "address", "mainCountry", "jurisdiction",
        "website", "email", "phone",
    ):
        vals = props.get(key)
        if vals:
            query_props[key] = [str(v) for v in vals]

    # Always force jurisdiction=Russia if not present (helps disambiguate)
    query_props.setdefault("jurisdiction", ["Russia"])

    payload = {"queries": {"q1": {"schema": "Company", "properties": query_props}}}
    params = {"algorithm": ALGORITHM}

    resp = requests.post(
        url,
        params=params,
        headers={**os_headers(api_key), "Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------

def is_sanctioned_topic(topics: list[str] | None) -> bool:
    """Topics list contains 'sanction' if the entity is on a sanctions list.

    OpenSanctions topics include things like: sanction, sanction.linked,
    sanction.counter, role.pep, crime, debarment, export.control. We treat
    a direct `sanction` tag as the strong signal. `sanction.linked` (an
    entity linked to but not itself sanctioned) is flagged for review.
    """
    if not topics:
        return False
    t = [str(x).lower() for x in topics]
    return "sanction" in t


# Datasets that are clear sanctions sources — used as a fallback signal
# when topics aren't populated on a /search response.
SANCTIONS_DATASETS_HINTS = {
    "sanctions", "default",
    "us_ofac_sdn", "us_ofac_cons",
    "eu_fsf", "eu_sanctions_map", "eu_travel_bans",
    "gb_hmt_sanctions", "gb_fcdo_sanctions",
    "ca_dfatd_sema_sanctions",
    "ch_seco_sanctions",
    "au_dfat_sanctions",
    "ja_mof_sanctions",
    "nz_russia_sanctions",
    "ua_nsdc_sanctions", "ua_war_sanctions",
    "fr_ga_sanctions", "be_fod_sanctions",
    "sg_terrorists",
    "mc_freezes",
}


def classify(bank_entity: dict, match_response: dict) -> dict:
    """Combine entity-level evidence and /match evidence into a verdict.

    Returns dict with:
        status:    "sanctioned" | "clear" | "review"
        reasons:   list[str] human-readable evidence
        top_hits:  list[dict] sanctions candidates with scores
    """
    reasons: list[str] = []
    top_hits: list[dict] = []
    status = "clear"

    bank_entity = bank_entity or {}
    props = bank_entity.get("properties", {}) or {}

    # 1) The bank registry entity is itself marked as a sanctions target.
    #    OpenSanctions sets `target=True` on entities that are designated
    #    on any watchlist. Because the CBR registry is deduplicated against
    #    sanctions lists, this is the most reliable signal.
    if bank_entity.get("target") is True:
        status = "sanctioned"
        reasons.append("Bank entity is flagged as a sanctions `target` in OpenSanctions.")

    # 2) Topic-based check (belt-and-braces; covers entities where the
    #    search response includes topics but not target=True).
    topics = props.get("topics") or bank_entity.get("topics") or []
    if is_sanctioned_topic(topics) and status != "sanctioned":
        status = "sanctioned"
        reasons.append(f"Bank entity carries sanctions topic ({', '.join(topics)}).")

    # 3) Dataset-membership check — if the entity belongs to any dataset
    #    other than `ru_cbr_banks` AND that dataset looks sanctions-related,
    #    flag as sanctioned. (Pure debarment/PEP datasets won't trigger.)
    datasets = bank_entity.get("datasets") or []
    other_ds = [d for d in datasets if d != BANKS_DATASET]
    sanction_ds_hits = [d for d in other_ds if d in SANCTIONS_DATASETS_HINTS
                        or "sanction" in d.lower() or "ofac" in d.lower()]
    if sanction_ds_hits and status != "sanctioned":
        status = "sanctioned"
        reasons.append(f"Bank appears in sanctions dataset(s): {', '.join(sanction_ds_hits)}.")

    # 4) Independent verification via /match
    q1 = (match_response or {}).get("responses", {}).get("q1", {})
    for hit in q1.get("results", []) or []:
        score = float(hit.get("score") or 0)
        hit_topics = (hit.get("properties", {}) or {}).get("topics") or \
                     hit.get("topics") or []
        top_hits.append({
            "id": hit.get("id"),
            "caption": hit.get("caption"),
            "score": score,
            "match": bool(hit.get("match")),
            "target": bool(hit.get("target")),
            "topics": hit_topics,
            "datasets": hit.get("datasets", []),
            "schema": hit.get("schema"),
        })
        if hit.get("match") and (is_sanctioned_topic(hit_topics) or hit.get("target")):
            if status != "sanctioned":
                status = "sanctioned"
            reasons.append(
                f"/match returned a confirmed match: "
                f"\"{hit.get('caption')}\" score={score:.2f}"
                + (f" topics=[{', '.join(hit_topics)}]" if hit_topics else "")
            )

    # 5) High-score but unconfirmed → REVIEW
    if status == "clear" and top_hits:
        best = max(top_hits, key=lambda x: x["score"])
        if best["score"] >= SANCTION_SCORE_THRESHOLD:
            status = "review"
            reasons.append(
                f"Top candidate \"{best['caption']}\" scored {best['score']:.2f} "
                f"(≥ {SANCTION_SCORE_THRESHOLD:.2f} threshold). Manual review recommended."
            )

    if status == "clear" and not reasons:
        reasons.append("No matches above the alert threshold across sanctions lists.")

    return {"status": status, "reasons": reasons, "top_hits": top_hits}


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def render_status_badge(status: str) -> None:
    colors = {
        "sanctioned": ("#b00020", "❌  SANCTIONED"),
        "review":     ("#b07a00", "⚠️  REVIEW REQUIRED"),
        "clear":      ("#0a7a2f", "✅  CLEAR"),
        "error":      ("#444444", "⚠  LOOKUP ERROR"),
        "unknown":    ("#444444", "❔ UNKNOWN BIK"),
    }
    color, label = colors.get(status, ("#444444", status.upper()))
    st.markdown(
        f"<div style='display:inline-block;padding:6px 14px;"
        f"background:{color};color:white;border-radius:6px;"
        f"font-weight:600;font-size:0.95rem'>{label}</div>",
        unsafe_allow_html=True,
    )


def first(props: dict, key: str, default: str = "—") -> str:
    vals = props.get(key) or []
    return str(vals[0]) if vals else default


def render_bank_card(bank: dict) -> None:
    props = bank.get("properties", {}) or {}
    name = first(props, "name", "(no name)")
    inn = first(props, "innCode")
    ogrn = first(props, "ogrnCode")
    address = first(props, "address")
    status = first(props, "status")
    incorp = first(props, "incorporationDate")
    aliases = props.get("alias") or []

    st.markdown(f"### {name}")
    cols = st.columns([1, 1, 1])
    cols[0].metric("INN", inn)
    cols[1].metric("OGRN", ogrn)
    cols[2].metric("Status", status)

    with st.expander("More bank details", expanded=False):
        if address != "—":
            st.write(f"**Address:** {address}")
        if incorp != "—":
            st.write(f"**Incorporated:** {incorp}")
        if aliases:
            st.write("**Aliases:** " + " · ".join(aliases[:10]))
        st.write(f"**OpenSanctions entity:** [`{bank.get('id')}`](https://www.opensanctions.org/entities/{bank.get('id')}/)")


def render_match_table(top_hits: list[dict]) -> None:
    if not top_hits:
        st.info("No candidate matches returned by /match.")
        return
    rows = []
    for h in top_hits:
        rows.append({
            "Score": round(h["score"], 3),
            "Confirmed match": "✅" if h["match"] else "—",
            "Entity": h["caption"],
            "Topics": ", ".join(h["topics"] or []),
            "Datasets": ", ".join(h["datasets"] or []),
            "OpenSanctions link": f"https://www.opensanctions.org/entities/{h['id']}/",
        })
    df = pd.DataFrame(rows).sort_values("Score", ascending=False)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "OpenSanctions link": st.column_config.LinkColumn(
                "OpenSanctions link", display_text="open ↗"
            ),
        },
    )


# ---------------------------------------------------------------------------
# Per-BIK pipeline
# ---------------------------------------------------------------------------

def screen_bik(bik_raw: str, api_key: str, scope: str) -> dict:
    """Run the full pipeline for a single BIK and return a result dict."""
    bik = normalize_bik(bik_raw)
    result: dict[str, Any] = {
        "bik_input": bik_raw,
        "bik": bik,
        "status": "error",
        "bank": None,
        "match": None,
        "verdict": None,
        "error": None,
    }
    if not is_valid_bik(bik):
        result["status"] = "error"
        result["error"] = "BIK must be 8 or 9 digits."
        return result

    try:
        bank = lookup_bank_in_cbr_registry(bik, api_key)
    except requests.HTTPError as exc:
        result["error"] = f"CBR lookup failed: {exc.response.status_code} {exc.response.text[:200]}"
        return result
    except Exception as exc:
        result["error"] = f"CBR lookup failed: {exc}"
        return result

    if not bank:
        result["status"] = "unknown"
        result["error"] = "No bank found in CBR registry for this BIK."
        return result
    result["bank"] = bank

    # Pull the fully-hydrated entity (includes datasets + sanction relationships)
    full = None
    if bank.get("id"):
        try:
            full = fetch_full_entity(bank["id"], api_key)
        except Exception:
            pass
    result["bank_full"] = full or bank

    try:
        match_response = match_against_sanctions(bank, api_key, scope=scope)
    except requests.HTTPError as exc:
        result["error"] = f"/match failed: {exc.response.status_code} {exc.response.text[:200]}"
        return result
    except Exception as exc:
        result["error"] = f"/match failed: {exc}"
        return result
    result["match"] = match_response

    verdict = classify(result["bank_full"], match_response)
    result["verdict"] = verdict
    result["status"] = verdict["status"]
    return result


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="BIK Sanctions Screener",
    page_icon="🏦",
    layout="wide",
)

st.title("🏦 BIK Sanctions Screener")
st.caption(
    "Resolve a Russian Bank Identification Code (БИК) to its issuer and screen "
    "it against global sanctions lists via the OpenSanctions API."
)

# --- Sidebar config -------------------------------------------------------
api_key = get_api_key()

with st.sidebar:
    st.header("Configuration")

    # API key status (no input — the key lives in st.secrets / env)
    if api_key:
        st.success("OpenSanctions API key loaded ✓", icon="🔑")
        with st.expander("Where is the key loaded from?"):
            try:
                in_secrets = bool(st.secrets.get("OPENSANCTIONS_API_KEY"))
            except Exception:
                in_secrets = False
            in_env = bool(os.environ.get("OPENSANCTIONS_API_KEY"))
            st.markdown(
                f"* `st.secrets`: {'✅ set' if in_secrets else '— not set'}\n"
                f"* `OPENSANCTIONS_API_KEY` env var: {'✅ set' if in_env else '— not set'}"
            )
    else:
        st.error("OpenSanctions API key not configured", icon="🔒")
        with st.expander("How to add the key", expanded=True):
            st.markdown(
                "Create **`.streamlit/secrets.toml`** in the project root:\n\n"
                "```toml\n"
                'OPENSANCTIONS_API_KEY = "your_key_here"\n'
                "```\n\n"
                "…or export it as an env var before launching Streamlit:\n\n"
                "```bash\n"
                'export OPENSANCTIONS_API_KEY="your_key_here"\n'
                "streamlit run app.py\n"
                "```\n\n"
                "Get a free key at "
                "[opensanctions.org/account](https://www.opensanctions.org/account/)."
            )

    scope = st.radio(
        "Screening scope",
        options=[SANCTIONS_SCOPE, DEFAULT_SCOPE],
        index=0,
        help=(
            "`sanctions` — only checks against sanctions lists (recommended). "
            "`default` — broader scope incl. PEPs, crime, debarment."
        ),
        format_func=lambda s: {
            SANCTIONS_SCOPE: "Sanctions only (recommended)",
            DEFAULT_SCOPE: "All (sanctions + PEPs + crime)",
        }[s],
    )
    st.divider()
    st.markdown(
        "**About**\n\n"
        "* BIK resolved via OpenSanctions `ru_cbr_banks` dataset "
        "(Central Bank of Russia registry, refreshed daily).\n"
        "* Sanctions match via `/match` endpoint with `logic-v2` scoring.\n"
        "* The CBR dataset and many sanctions lists are deduplicated by "
        "OpenSanctions, so a sanctioned bank shows up as a single entity "
        "tagged with the `sanction` topic."
    )

# --- Main pane ------------------------------------------------------------
tab_single, tab_batch = st.tabs(["Single BIK", "Batch screening"])

# ---------- Tab 1: Single BIK --------------------------------------------
with tab_single:
    col_in, col_btn = st.columns([3, 1])
    with col_in:
        bik_input = st.text_input(
            "Enter a BIK (8 or 9 digits)",
            placeholder="e.g. 044525700",
            label_visibility="collapsed",
        )
    with col_btn:
        run = st.button("Screen", type="primary", use_container_width=True)

    # Sample chips for convenience
    st.caption("Try a sample BIK:")
    sample_cols = st.columns(5)
    samples = [
        ("040813713", "sanctioned"),
        ("044030653", "sanctioned"),
        ("046577904", "clear"),
        ("046015762", "clear"),
        ("044525700", "clear"),
    ]
    for col, (s, label) in zip(sample_cols, samples):
        with col:
            icon = "❌" if label == "sanctioned" else "✅"
            if st.button(f"{icon} {s}", key=f"sample_{s}", use_container_width=True,
                         help=f"expected: {label}"):
                st.session_state["_prefill"] = s
                st.rerun()

    if "_prefill" in st.session_state and not bik_input:
        bik_input = st.session_state.pop("_prefill")
        run = True

    if run:
        if not api_key:
            st.warning(
                "OpenSanctions API key is not configured. Add it to "
                "`.streamlit/secrets.toml` as "
                "`OPENSANCTIONS_API_KEY = \"...\"` (see sidebar) and reload.",
                icon="🔒",
            )
        elif not bik_input.strip():
            st.warning("Please enter a BIK.")
        else:
            with st.spinner(f"Screening BIK {normalize_bik(bik_input)}…"):
                res = screen_bik(bik_input, api_key, scope)

            st.divider()
            top = st.columns([1, 3])
            with top[0]:
                render_status_badge(res["status"])
                st.write(f"**BIK:** `{res['bik']}`")
            with top[1]:
                if res["status"] == "error":
                    st.error(res["error"])
                elif res["status"] == "unknown":
                    st.warning(res["error"])
                else:
                    for r in res["verdict"]["reasons"]:
                        st.write(f"• {r}")

            if res.get("bank"):
                st.divider()
                render_bank_card(res["bank_full"] or res["bank"])

            if res.get("verdict"):
                st.divider()
                st.subheader("Sanctions match candidates")
                render_match_table(res["verdict"]["top_hits"])

            with st.expander("Raw API responses (debug)"):
                st.write("**Bank (search):**")
                st.json(res.get("bank") or {})
                st.write("**Bank (full entity):**")
                st.json(res.get("bank_full") or {})
                st.write("**/match response:**")
                st.json(res.get("match") or {})

# ---------- Tab 2: Batch screening ---------------------------------------
with tab_batch:
    st.write(
        "Paste one BIK per line (or comma/space-separated). "
        "Up to ~50 at a time is comfortable."
    )
    bulk = st.text_area(
        "BIKs",
        height=140,
        placeholder="040813713\n044030653\n046577904\n046015762\n044525700",
        label_visibility="collapsed",
    )
    run_batch = st.button("Screen all", type="primary")

    if run_batch:
        if not api_key:
            st.warning(
                "OpenSanctions API key is not configured. Add it to "
                "`.streamlit/secrets.toml` and reload.",
                icon="🔒",
            )
        else:
            raw_biks = re.split(r"[\s,;]+", bulk.strip())
            biks = [b for b in (normalize_bik(x) for x in raw_biks) if b]
            biks = list(dict.fromkeys(biks))  # de-dupe, preserve order
            if not biks:
                st.warning("No valid BIKs found.")
            else:
                progress = st.progress(0.0)
                status_line = st.empty()
                results = []
                for i, b in enumerate(biks, 1):
                    status_line.write(f"Screening {b} ({i}/{len(biks)})…")
                    results.append(screen_bik(b, api_key, scope))
                    progress.progress(i / len(biks))
                    # gentle rate-limit; OpenSanctions free tier ~60 rpm
                    time.sleep(0.2)
                status_line.empty()
                progress.empty()

                # Build a results dataframe
                rows = []
                for r in results:
                    props = (r.get("bank") or {}).get("properties", {}) or {}
                    rows.append({
                        "BIK": r["bik"],
                        "Status": r["status"],
                        "Bank": first(props, "name"),
                        "INN": first(props, "innCode"),
                        "OGRN": first(props, "ogrnCode"),
                        "Top hit": (
                            r["verdict"]["top_hits"][0]["caption"]
                            if r.get("verdict") and r["verdict"]["top_hits"] else "—"
                        ),
                        "Top score": (
                            round(r["verdict"]["top_hits"][0]["score"], 3)
                            if r.get("verdict") and r["verdict"]["top_hits"] else None
                        ),
                        "Notes": (
                            r["verdict"]["reasons"][0]
                            if r.get("verdict") and r["verdict"]["reasons"]
                            else (r.get("error") or "")
                        ),
                    })
                df = pd.DataFrame(rows)

                # Color status — Styler.map (pandas ≥ 2.1; Styler.applymap is deprecated)
                def _color(s):
                    return {
                        "sanctioned": "background-color:#fdecea;color:#b00020;font-weight:600",
                        "review":     "background-color:#fff4e0;color:#b07a00;font-weight:600",
                        "clear":      "background-color:#e9f7ee;color:#0a7a2f;font-weight:600",
                    }.get(s, "")
                styled = df.style.map(_color, subset=["Status"]) \
                    if hasattr(df.style, "map") else df.style.applymap(_color, subset=["Status"])
                st.divider()
                st.dataframe(
                    styled,
                    use_container_width=True,
                    hide_index=True,
                )

                # Export
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download results as CSV",
                    csv,
                    file_name=f"bik_screening_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )

                # Per-BIK detail expanders
                st.divider()
                st.subheader("Details")
                for r in results:
                    label = f"{r['bik']} — {first((r.get('bank') or {}).get('properties', {}) or {}, 'name')} — {r['status'].upper()}"
                    with st.expander(label):
                        if r["status"] in ("error", "unknown"):
                            st.error(r.get("error") or "")
                            continue
                        for line in r["verdict"]["reasons"]:
                            st.write(f"• {line}")
                        render_match_table(r["verdict"]["top_hits"])

st.divider()
st.caption(
    f"Data: OpenSanctions ({BANKS_DATASET} + sanctions collections). "
    "Not legal advice — confirm matches with your compliance officer."
)
