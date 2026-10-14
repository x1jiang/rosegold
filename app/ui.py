import sys
import os

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import html
import re
import time
import json
import datetime
import requests
import pandas as pd
import streamlit as st

from app.engine import AdjudicationEngine
from app.omop_loader import load_omop_data
from app.mimic_ext_notes import looks_like_mimic_ext_notes_path
from app.concordance import calculate_concordance_metrics
from app.omop_export import export_to_omop_observation
from app.config_loader import resolve_criteria
from app.model_selector import detect_hardware, resolve_model_and_engine
from app.paths import UnsafePathError, data_dir, resolve_data_file
from app.storage import (
    append_audit,
    load_batch_results,
    load_criteria,
    read_audit,
    save_batch_results,
    save_criteria,
    storage_status,
)

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Rose Gold Clinical Adjudication Suite",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1a237e;
        margin-bottom: 2px;
        letter-spacing: -0.02em;
    }
    .sub-header {
        font-size: 0.98rem;
        color: #546e7a;
        margin-bottom: 16px;
    }
    .status-positive {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        border: 2px solid #e53935;
        color: #b71c1c;
        padding: 12px 20px;
        border-radius: 10px;
        font-weight: 700;
        font-size: 1.12rem;
        text-align: center;
        margin-bottom: 14px;
    }
    .status-negative {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border: 2px solid #43a047;
        color: #1b5e20;
        padding: 12px 20px;
        border-radius: 10px;
        font-weight: 700;
        font-size: 1.12rem;
        text-align: center;
        margin-bottom: 14px;
    }
    .status-suspected {
        background: linear-gradient(135deg, #fff8e1 0%, #ffecb3 100%);
        border: 2px solid #ffa000;
        color: #e65100;
        padding: 12px 20px;
        border-radius: 10px;
        font-weight: 700;
        font-size: 1.12rem;
        text-align: center;
        margin-bottom: 14px;
    }
    .evidence-card {
        background-color: #f1f8ff;
        border-left: 5px solid #0366d6;
        padding: 12px 14px;
        border-radius: 6px;
        margin-bottom: 10px;
        font-size: 0.94rem;
    }
    .note-card {
        background-color: #ffffff;
        border: 1px solid #e1e4e8;
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .note-card-matched {
        background-color: #ffffff;
        border: 1.5px solid #90caf9;
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 12px;
        box-shadow: 0 2px 6px rgba(3,102,214,0.08);
    }
    .metric-badge {
        background-color: #f0f4f8;
        padding: 5px 11px;
        border-radius: 14px;
        font-size: 0.84rem;
        font-weight: 600;
        color: #24292e;
        display: inline-block;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    .metric-badge-accent {
        background-color: #e3f2fd;
        border: 1px solid #bbdefb;
        padding: 5px 11px;
        border-radius: 14px;
        font-size: 0.84rem;
        font-weight: 600;
        color: #0d47a1;
        display: inline-block;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    mark.ev-highlight {
        background-color: #fff3cd;
        border-bottom: 2px solid #f59f00;
        color: #212529;
        padding: 1px 3px;
        border-radius: 3px;
    }
    mark.kw-highlight {
        background-color: #d0ebff;
        border-bottom: 2px solid #1c7ed6;
        color: #0b7285;
        padding: 1px 3px;
        border-radius: 3px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Persistent HTTP Session & Cached Helpers
# ---------------------------------------------------------
API_BASE_URL = os.getenv("ROSEGOLD_API_URL", "http://localhost:8000")


@st.cache_resource
def _http_session() -> requests.Session:
    """Reuse TCP connections to the local FastAPI server across reruns."""
    return requests.Session()


def _api_headers():
    """Shared-secret header when the API is protected by ROSEGOLD_API_KEY."""
    key = os.getenv("ROSEGOLD_API_KEY", "").strip()
    return {"X-API-Key": key} if key else {}


# On Cloud Run (K_SERVICE) or when ROSEGOLD_API_URL is configured explicitly, the
# API process is authoritative: falling back to an in-process engine would load a
# second 2 GB model next to the one the API already holds. Local Mode is only for
# laptops where no API is running at all (connection refused).
_LOCAL_ENGINE_ALLOWED = not os.getenv("K_SERVICE") and not os.getenv("ROSEGOLD_API_URL")
_LAST_HEALTH = {}


@st.cache_data(ttl=5)
def check_api():
    try:
        r = _http_session().get(f"{API_BASE_URL}/health", headers=_api_headers(), timeout=(2, 8))
        if r.status_code == 200:
            payload = r.json()
            _LAST_HEALTH["data"] = payload
            return True, payload
    except requests.ConnectionError:
        if _LOCAL_ENGINE_ALLOWED:
            return False, None
    except Exception:
        pass
    if _LOCAL_ENGINE_ALLOWED:
        return False, None
    return True, _LAST_HEALTH.get("data") or {"backend": "loading", "llm_real": False}


@st.cache_data(ttl=30)
def _cached_detect_hardware():
    return detect_hardware()


@st.cache_data(ttl=60)
def _cached_looks_like_mimic(path: str) -> bool:
    return looks_like_mimic_ext_notes_path(path)


@st.cache_data(ttl=30)
def _cached_storage_status():
    return storage_status()


@st.cache_data(ttl=15)
def fetch_audit_entries(api_is_online: bool):
    entries = []
    if api_is_online:
        try:
            resp = _http_session().get(f"{API_BASE_URL}/api/audit", headers=_api_headers(), timeout=10)
            if resp.status_code == 200:
                entries = resp.json().get("entries") or []
        except Exception:
            entries = []
    if not entries:
        entries = read_audit()
    return entries


@st.cache_data(ttl=15)
def fetch_persisted_batch(api_is_online: bool):
    persisted = []
    if api_is_online:
        try:
            resp = _http_session().get(f"{API_BASE_URL}/api/batch", headers=_api_headers(), timeout=10)
            if resp.status_code == 200:
                persisted = resp.json().get("results") or []
        except Exception:
            persisted = []
    if not persisted:
        persisted = load_batch_results()
    return persisted


@st.cache_data
def build_export_payloads(results_json: str, condition: str):
    """Cache OMOP Observation table conversion and CSV/JSONL serialization."""
    results_list = json.loads(results_json)
    df_cohort = pd.DataFrame(results_list)
    df_omop_obs = export_to_omop_observation(results_list, condition)
    csv_omop = df_omop_obs.to_csv(index=False).encode("utf-8")
    csv_rosegold = df_cohort.to_csv(index=False).encode("utf-8")
    jsonl_data = "\n".join(json.dumps(r) for r in results_list).encode("utf-8")
    return csv_omop, csv_rosegold, jsonl_data


def _highlight_note_text(
    body: str,
    search_kw: str = "",
    evidence_quotes: list = None,
    enable_ev_highlight: bool = True,
) -> tuple[str, bool]:
    """Safely HTML-escape note text while highlighting verbatim evidence quotes and search keywords."""
    if not body:
        return "", False

    spans = []  # list of (start, end, css_class)
    body_lower = body.lower()

    if enable_ev_highlight and evidence_quotes:
        for raw_q in evidence_quotes:
            q = (raw_q or "").strip()
            if len(q) < 6:
                continue
            q_lower = q.lower()
            idx = body_lower.find(q_lower)
            if idx != -1:
                spans.append((idx, idx + len(q), "ev-highlight"))
            else:
                # Fallback: match whitespace-flexible regex for multi-word quote
                words = [re.escape(w) for w in q.split() if w]
                if len(words) >= 3:
                    pat = r"\s+".join(words)
                    m = re.search(pat, body, flags=re.IGNORECASE)
                    if m:
                        spans.append((m.start(), m.end(), "ev-highlight"))

    kw = (search_kw or "").strip()
    if len(kw) >= 2:
        for m in re.finditer(re.escape(kw), body, flags=re.IGNORECASE):
            spans.append((m.start(), m.end(), "kw-highlight"))

    if not spans:
        return html.escape(body), False

    # Sort spans by start index, preferring longer spans on ties
    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    merged = []
    for s_start, s_end, s_cls in spans:
        if not merged or s_start >= merged[-1][1]:
            merged.append((s_start, s_end, s_cls))

    parts = []
    cursor = 0
    for s_start, s_end, s_cls in merged:
        if s_start > cursor:
            parts.append(html.escape(body[cursor:s_start]))
        parts.append(f"<mark class='{s_cls}'>{html.escape(body[s_start:s_end])}</mark>")
        cursor = s_end
    if cursor < len(body):
        parts.append(html.escape(body[cursor:]))

    return "".join(parts), True


api_online, health_data = check_api()


def _api_error_message(resp) -> str:
    """Short, non-sensitive summary of an API error for the dashboard."""
    rid = resp.headers.get("x-request-id", "")
    suffix = f" (request id `{rid}`)" if rid else ""
    if resp.status_code == 503:
        return f"The adjudication backend is not ready yet{suffix}. Check `/ready` or the service logs."
    if resp.status_code == 401:
        return "The API rejected the dashboard's credentials. Set ROSEGOLD_API_KEY for both processes."
    if resp.status_code == 429:
        return "The API is rate-limiting requests. Wait a moment and retry."
    return f"The API returned HTTP {resp.status_code}{suffix}."


# ---------------------------------------------------------
# Sidebar Navigation & Settings
# ---------------------------------------------------------
st.sidebar.markdown("### ⚙️ Adjudication Controls")

DATA_ROOT = data_dir() if os.getenv("ROSEGOLD_DATA_DIR") else os.path.join(PROJECT_ROOT, "data")
omop_notes = os.path.join(DATA_ROOT, "synthetic_notes.csv")
omop_visits = os.path.join(DATA_ROOT, "synthetic_visits.csv")
mimic_notes = os.path.join(DATA_ROOT, "synthetic_mimic_ext_notes", "notes.csv")
mimic_visits = os.path.join(DATA_ROOT, "synthetic_mimic_ext_notes", "omop_visits.csv")

dataset = st.sidebar.selectbox(
    "Cohort",
    ["Synthetic OMOP (20 visits)", "Synthetic MIMIC-III-Ext-Notes"],
)
if dataset.startswith("Synthetic MIMIC"):
    default_notes, default_visits = mimic_notes, mimic_visits
else:
    default_notes, default_visits = omop_notes, omop_visits

with st.sidebar.expander("📂 Dataset File Paths", expanded=False):
    notes_path_input = st.text_input("Notes table", value=default_notes, key=f"notes_{dataset}")
    visits_path_input = st.text_input("Visits table", value=default_visits, key=f"visits_{dataset}")
    st.caption(f"Files must be `.csv`/`.parquet` inside `{DATA_ROOT}`.")

try:
    notes_path = resolve_data_file(notes_path_input, root=DATA_ROOT)
except UnsafePathError:
    st.sidebar.error("Notes table must be a .csv/.parquet file inside the configured data directory.")
    st.stop()

visits_path = None
if (visits_path_input or "").strip():
    try:
        visits_path = resolve_data_file(visits_path_input, root=DATA_ROOT)
    except UnsafePathError:
        visits_path = None

try:
    notes_is_mimic = _cached_looks_like_mimic(notes_path)
except Exception:
    st.sidebar.error("Notes table could not be read. Check that it is a valid CSV/Parquet file.")
    st.stop()

if notes_is_mimic:
    st.sidebar.caption("Detected MIMIC-III-Ext-Notes `notes.csv` (`hadm_id` visits).")
elif visits_path is None:
    st.sidebar.error("Visits table must be a .csv/.parquet file inside the configured data directory.")
    st.stop()

phenotype_choice = st.sidebar.selectbox(
    "Target Clinical Phenotype",
    [
        "Sepsis / Septic Shock",
        "Acute Ischemic Stroke",
        "Acute Respiratory Distress Syndrome (ARDS)",
        "Acute Kidney Injury (AKI)",
        "Custom Phenotype...",
    ]
)

if phenotype_choice == "Custom Phenotype...":
    custom_pheno = st.sidebar.text_input(
        "Custom Phenotype Name",
        value="Pulmonary Embolism",
        placeholder="e.g. Pulmonary Embolism, Decompensated Heart Failure",
    ).strip()
    target_condition = custom_pheno or "Pulmonary Embolism"
else:
    target_condition = phenotype_choice

hw_info = _cached_detect_hardware()


@st.cache_resource
def get_local_engine():
    return AdjudicationEngine(model_name="auto")


st.sidebar.markdown("### 🖥️ Runtime & LLM Status")
if hw_info["is_gpu"]:
    st.sidebar.success(f"🚀 **GPU Detected:** {hw_info['device_name']} ({hw_info['vram_gb']} GB VRAM)")
    default_model_idx = 0
else:
    st.sidebar.info(f"💻 **CPU Mode:** {hw_info['device_name']}")
    default_model_idx = 0

llm_backend = st.sidebar.selectbox(
    "Inference Engine & Model",
    [
        "Auto-Select for Hardware (Recommended)",
        "Rose Gold Hybrid (Rules + Muse GPU)",
        "Llama 3.1 8B-Instruct (GPU / vLLM)",
        "Llama 3.2 3B-Instruct (CPU Lightweight)",
        "Gemma 2 9B-Instruct (GPU)",
        "Gemma 2 2B-Instruct (CPU Lightweight)",
        "Muse Glimmer (30B - H100 GPU)"
    ],
    index=default_model_idx
)

if "Auto-Select" in llm_backend:
    resolved = resolve_model_and_engine(requested_family="llama")
    st.sidebar.caption(f"⚡ Active: **{resolved['selected_model']}** on {resolved['target_device'].upper()}")
elif "Hybrid" in llm_backend:
    resolved = resolve_model_and_engine(requested_model="muse-glimmer-30b", force_device="cuda")
    st.sidebar.caption("⚡ Active: **Rose Gold Hybrid (Rules + Muse-30B)**")
elif "3.2 3B" in llm_backend:
    resolved = resolve_model_and_engine(requested_model="meta-llama/Llama-3.2-3B-Instruct", force_device="cpu")
elif "2B" in llm_backend:
    resolved = resolve_model_and_engine(requested_model="google/gemma-2-2b-it", force_device="cpu")
elif "Glimmer" in llm_backend:
    resolved = resolve_model_and_engine(requested_model="muse-glimmer-30b", force_device="cuda")
elif "Gemma 2 9B" in llm_backend:
    resolved = resolve_model_and_engine(requested_model="google/gemma-2-9b-it", force_device="cuda")
else:
    resolved = resolve_model_and_engine(requested_model="meta-llama/Llama-3.1-8B-Instruct", force_device="cuda")

if api_online:
    backend = (health_data or {}).get("backend") or "unknown"
    model_label = (health_data or {}).get("model_name") or resolved["selected_model"]
    if (health_data or {}).get("llm_real"):
        st.sidebar.success(f"🟢 **Real LLM:** `{backend}` / `{model_label}`")
    elif backend == "loading":
        st.sidebar.warning("🟡 **Loading Llama 3.2 3B** CPU weights. First review is slow.")
    else:
        st.sidebar.error(f"🔴 **Not an LLM:** `{backend}` (keyword rules). Adjudications are canned heuristics.")
    st.sidebar.caption(f"FastAPI online · device={(health_data or {}).get('device', 'cpu')}")
else:
    st.sidebar.info("🟡 **Local Mode:** Direct Engine")

_store = _cached_storage_status()
if _store["durable"]:
    st.sidebar.success(f"💾 **Persistent GCS:** `{_store['gcs_bucket'] or _store['output_dir']}`")
else:
    st.sidebar.info(f"💾 **Local store:** `{_store['output_dir']}`")

if "custom_criteria_loaded" not in st.session_state:
    saved_criteria = None
    if api_online:
        try:
            resp = _http_session().get(f"{API_BASE_URL}/api/criteria", headers=_api_headers(), timeout=5)
            if resp.status_code == 200:
                saved_criteria = resp.json().get("text")
        except Exception:
            saved_criteria = None
    if not saved_criteria:
        saved_criteria = load_criteria()
    if saved_criteria:
        st.session_state["custom_criteria"] = saved_criteria
    st.session_state["custom_criteria_loaded"] = True


# Load Cohort
@st.cache_data
def load_cohort(n_p, v_p):
    if not os.path.exists(n_p):
        return []
    if _cached_looks_like_mimic(n_p) or (v_p and os.path.exists(v_p)):
        return load_omop_data(n_p, v_p)
    return []


try:
    records = load_cohort(notes_path, visits_path)
except Exception:
    st.error("The selected dataset could not be parsed. Check that it follows the OMOP NOTE / VISIT_OCCURRENCE schema.")
    st.stop()

if not records:
    st.error("No OMOP data found. Please check data file paths.")
    st.stop()

criteria_state_key = f"criteria_{target_condition}"
active_criteria = (
    st.session_state.get(criteria_state_key)
    or st.session_state.get("custom_criteria")
    or resolve_criteria(target_condition)
)

# Hydrate persisted batch results once into session state
if "cohort_results_hydrated" not in st.session_state:
    persisted_batch = fetch_persisted_batch(api_online)
    if persisted_batch:
        st.session_state["cohort_results"] = persisted_batch
        for item in persisted_batch:
            vid = item.get("visit_occurrence_id")
            if vid is not None:
                s_key = f"adj_{vid}_{target_condition}"
                if s_key not in st.session_state:
                    st.session_state[s_key] = item
    st.session_state["cohort_results_hydrated"] = True


def adjudicate_record(record, condition, criteria):
    """Adjudicate one visit. Returns None (after showing an error) instead of raising."""
    t0 = time.monotonic()
    if api_online:
        try:
            resp = _http_session().post(
                f"{API_BASE_URL}/api/adjudicate/single",
                headers=_api_headers(),
                json={
                    "visit_occurrence_id": record.get("visit_occurrence_id"),
                    "person_id": record.get("person_id"),
                    "notes_formatted_text": record.get("notes_formatted_text"),
                    "target_condition": condition,
                    "clinical_criteria": criteria,
                },
                timeout=600,
            )
        except requests.RequestException:
            st.error("Could not reach the adjudication API. Retry once it is back online.")
            return None
        if resp.status_code != 200:
            st.error(_api_error_message(resp))
            return None
        res = resp.json()
        res["_elapsed_sec"] = round(time.monotonic() - t0, 2)
        return res
    try:
        res = get_local_engine().adjudicate_single(record, condition, criteria)
        res["_elapsed_sec"] = round(time.monotonic() - t0, 2)
        return res
    except RuntimeError:
        st.error("The configured LLM backend is not available in this process. Start the API service or fix the backend configuration.")
        return None


def adjudicate_cohort(recs, condition, criteria):
    """Batch adjudicate. Returns None (after showing an error) instead of raising."""
    if api_online:
        try:
            resp = _http_session().post(
                f"{API_BASE_URL}/api/adjudicate/batch",
                headers=_api_headers(),
                json={
                    "target_condition": condition,
                    "clinical_criteria": criteria,
                    "visit_occurrence_ids": [item["visit_occurrence_id"] for item in recs],
                },
                timeout=600,
            )
        except requests.RequestException:
            st.error("Could not reach the adjudication API. Retry once it is back online.")
            return None
        if resp.status_code != 200:
            st.error(_api_error_message(resp))
            return None
        results = resp.json()
        fetch_persisted_batch.clear()
        return results
    try:
        results = get_local_engine().adjudicate_batch(recs, condition, criteria)
    except RuntimeError:
        st.error("The configured LLM backend is not available in this process. Start the API service or fix the backend configuration.")
        return None
    save_batch_results(results, condition)
    fetch_persisted_batch.clear()
    return results


# ---------------------------------------------------------
# Top Header & KPI Summary
# ---------------------------------------------------------
st.markdown("<div class='main-header'>🩺 Rose Gold: Clinical LLM Chart Adjudication Suite</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>High-Throughput Clinical Phenotyping & Evidence-Grounded Chart Review for OMOP CDM</div>", unsafe_allow_html=True)

audit_entries = fetch_audit_entries(api_online)
signed_visit_ids = {e.get("visit_occurrence_id") for e in audit_entries if e.get("visit_occurrence_id") is not None}
adjudicated_count = sum(
    1 for r in records if f"adj_{r['visit_occurrence_id']}_{target_condition}" in st.session_state
)

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("Encounter Cohort", f"{len(records)} Visits", f"{adjudicated_count} scored")
total_notes_count = sum(r["note_count"] for r in records)
kpi2.metric("Total Notes", f"{total_notes_count} Notes", f"{len(signed_visit_ids)} signed off")
kpi3.metric("Target Phenotype", target_condition.split("/")[0].strip())
kpi4.metric("LLM Engine", resolved.get("engine_type", "Auto"))
if api_online and (health_data or {}).get("backend") == "llamacpp":
    kpi5.metric("LLM Path", "Llama 3.2 3B CPU")
elif api_online and (health_data or {}).get("backend") == "vertex":
    kpi5.metric("LLM Path", "Vertex AI Gemini")
elif api_online and (health_data or {}).get("llm_real"):
    kpi5.metric("PHI Boundary", "Site-Approved LLM")
else:
    kpi5.metric("PHI Boundary", "100% On-Premises")

st.divider()

# ---------------------------------------------------------
# Tabs: Review | Calibration | Batch | Definitions | Tutorial
# ---------------------------------------------------------
tab_review, tab_calib, tab_batch, tab_rules, tab_tutorial = st.tabs([
    "🔍 Interactive Chart Review",
    "📊 Calibration & Concordance Matrix",
    "⚡ Batch Adjudication & OMOP Export",
    "🛠️ Phenotype Rules & Prompts",
    "📖 CLI & FastAPI Tutorial"
])

# ---------------------------------------------------------
# TAB 1: INTERACTIVE CHART REVIEW
# ---------------------------------------------------------
with tab_review:
    # Queue Filter & Prev/Next Navigation
    nav_c1, nav_c2, nav_c3, nav_c4, nav_c5 = st.columns([1.2, 2.6, 0.55, 0.55, 1.3])

    with nav_c1:
        queue_filter = st.selectbox(
            "Filter Queue:",
            ["All Encounters", "Unreviewed Only", "Adjudicated Positive", "Adjudicated Negative"],
            key="review_queue_filter",
        )

    filtered_records = []
    for r in records:
        vid = r["visit_occurrence_id"]
        s_key = f"adj_{vid}_{target_condition}"
        adj_cached = st.session_state.get(s_key)
        if queue_filter == "Unreviewed Only" and vid in signed_visit_ids:
            continue
        if queue_filter == "Adjudicated Positive" and not (adj_cached and adj_cached.get("condition_present")):
            continue
        if queue_filter == "Adjudicated Negative" and not (adj_cached and not adj_cached.get("condition_present")):
            continue
        filtered_records.append(r)

    if not filtered_records:
        st.info(f"No encounters match filter **{queue_filter}**. Showing all encounters.")
        filtered_records = records

    def _visit_label(rec):
        vid = rec["visit_occurrence_id"]
        s_key = f"adj_{vid}_{target_condition}"
        status_icon = "⚪"
        if s_key in st.session_state:
            st_val = st.session_state[s_key].get("phenotype_status", "")
            status_icon = "🔴" if "POSITIVE" in st_val else ("🟢" if "NEGATIVE" in st_val else "🟡")
        sign_tag = " ✓ Signed" if vid in signed_visit_ids else ""
        return f"{status_icon} Visit {vid} | Person {rec['person_id']} ({rec['note_count']} notes: {rec['visit_start_date']}){sign_tag}"

    visit_dict = {r["visit_occurrence_id"]: _visit_label(r) for r in filtered_records}
    visit_ids_list = list(visit_dict.keys())

    if "active_visit_id" not in st.session_state or st.session_state["active_visit_id"] not in visit_ids_list:
        st.session_state["active_visit_id"] = visit_ids_list[0]

    curr_idx = visit_ids_list.index(st.session_state["active_visit_id"])

    with nav_c3:
        st.write("")
        st.write("")
        if st.button("⬅️ Prev", use_container_width=True, disabled=(curr_idx == 0)):
            st.session_state["active_visit_id"] = visit_ids_list[max(0, curr_idx - 1)]
            st.rerun()

    with nav_c4:
        st.write("")
        st.write("")
        if st.button("Next ➡️", use_container_width=True, disabled=(curr_idx >= len(visit_ids_list) - 1)):
            st.session_state["active_visit_id"] = visit_ids_list[min(len(visit_ids_list) - 1, curr_idx + 1)]
            st.rerun()

    with nav_c2:
        selected_vid = st.selectbox(
            "Select Patient Visit Encounter to Review:",
            visit_ids_list,
            index=visit_ids_list.index(st.session_state["active_visit_id"]),
            format_func=lambda x: visit_dict[x],
        )
        st.session_state["active_visit_id"] = selected_vid

    selected_record = next(r for r in records if r["visit_occurrence_id"] == selected_vid)

    with nav_c5:
        st.write("")
        st.write("")
        run_btn = st.button("🚀 Adjudicate this Encounter", type="primary", use_container_width=True)

    session_key = f"adj_{selected_vid}_{target_condition}"
    if run_btn:
        with st.spinner("Running adjudication..."):
            adjudication = adjudicate_record(selected_record, target_condition, active_criteria)
        if adjudication is not None:
            st.session_state[session_key] = adjudication
            st.rerun()

    current_adj = st.session_state.get(session_key)
    evidence_quotes = [
        ev.get("evidence_quote", "")
        for ev in (current_adj.get("key_evidence") or [])
        if isinstance(ev, dict) and ev.get("evidence_quote")
    ] if current_adj else []

    col_notes_view, col_adj_view = st.columns([1.15, 0.85])

    # Left: Medical Record Timeline
    with col_notes_view:
        st.markdown(f"### 📄 Medical Record Timeline (Visit `{selected_vid}`)")
        sign_badge = (
            "<span class='metric-badge-accent'>✅ Physician Signed</span>"
            if selected_vid in signed_visit_ids
            else "<span class='metric-badge'>⏳ Pending Sign-off</span>"
        )
        st.markdown(
            f"<span class='metric-badge'>👤 Person ID: {html.escape(str(selected_record['person_id']))}</span>"
            f"<span class='metric-badge'>📅 {html.escape(str(selected_record['visit_start_date']))} to {html.escape(str(selected_record['visit_end_date']))}</span>"
            f"<span class='metric-badge'>📝 {html.escape(str(selected_record['note_count']))} Notes</span>"
            f"{sign_badge}",
            unsafe_allow_html=True
        )

        f_col1, f_col2 = st.columns([2.2, 1.0])
        with f_col1:
            filter_kw = st.text_input(
                "🔎 Search / Highlight keywords in patient notes:",
                "",
                placeholder="e.g. lactate, norepinephrine, stroke, intubated",
            )
        with f_col2:
            st.write("")
            st.write("")
            hl_evidence = st.checkbox("Highlight LLM Evidence", value=True, disabled=not bool(evidence_quotes))

        raw_text = selected_record["notes_formatted_text"]

        note_chunks = raw_text.split("--- [Note ID:")
        shown_notes = 0
        for chunk in note_chunks:
            if not chunk.strip():
                continue
            chunk_full = "--- [Note ID:" + chunk if not chunk.startswith("---") else chunk
            lines = chunk_full.strip().split("\n")
            header = lines[0] if lines else "Note"
            body = "\n".join(lines[1:]) if len(lines) > 1 else ""

            if filter_kw and filter_kw.lower() not in chunk_full.lower():
                continue

            shown_notes += 1
            highlighted_body, has_match = _highlight_note_text(
                body,
                search_kw=filter_kw,
                evidence_quotes=evidence_quotes,
                enable_ev_highlight=hl_evidence,
            )
            card_cls = "note-card-matched" if has_match else "note-card"
            match_pill = (
                " <span style='background:#fff3cd;color:#856404;padding:2px 8px;border-radius:10px;font-size:0.78rem;float:right;'>📌 Evidence / Keyword Match</span>"
                if has_match else ""
            )

            st.markdown(f"""
            <div class='{card_cls}'>
                <strong>{html.escape(header)}</strong>{match_pill}
                <hr style='margin: 6px 0;'>
                <div style='white-space: pre-wrap; font-family: -apple-system, BlinkMacSystemFont, sans-serif; font-size: 0.93rem; line-height: 1.5; color: #2c3e50;'>{highlighted_body}</div>
            </div>
            """, unsafe_allow_html=True)

        if shown_notes == 0 and filter_kw:
            st.warning(f"No notes in Visit `{selected_vid}` matched keyword `{filter_kw}`.")

    # Right: LLM Adjudication Panel
    with col_adj_view:
        st.markdown("### 🤖 Rose Gold Adjudication")

        if session_key not in st.session_state:
            st.info("Click **🚀 Adjudicate this Encounter** above to run the engine. Charts are not scored automatically on page load.")
        else:
            adj = st.session_state[session_key]
            status = adj["phenotype_status"]
            conf = float(adj.get("confidence_score", 0.0))

            if "CONFIRMED_POSITIVE" in status:
                st.markdown(f"<div class='status-positive'>⚠️ {html.escape(status)}<br><small>Confidence: {conf*100:.1f}%</small></div>", unsafe_allow_html=True)
            elif "CONFIRMED_NEGATIVE" in status:
                st.markdown(f"<div class='status-negative'>✅ {html.escape(status)}<br><small>Confidence: {conf*100:.1f}%</small></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='status-suspected'>❓ {html.escape(status)}<br><small>Confidence: {conf*100:.1f}%</small></div>", unsafe_allow_html=True)

            st.progress(min(1.0, max(0.0, conf)), text=f"Model Diagnostic Confidence: {conf*100:.1f}%")

            backend_used = adj.get("inference_backend") or (health_data or {}).get("backend") or "unknown"
            elapsed_s = adj.get("_elapsed_sec")
            ev_count = len(adj.get("key_evidence") or [])
            meta_badges = f"<span class='metric-badge'>⚙️ Backend: {html.escape(str(backend_used))}</span>"
            meta_badges += f"<span class='metric-badge'>📌 {ev_count} Grounded Quote{'s' if ev_count != 1 else ''}</span>"
            if elapsed_s is not None:
                meta_badges += f"<span class='metric-badge'>⏱️ {elapsed_s}s</span>"
            st.markdown(meta_badges, unsafe_allow_html=True)

            if str(backend_used).startswith("keyword"):
                st.caption("⚠️ _Generated via deterministic clinical rules (no LLM backend active)._")

            st.markdown("#### 📋 Clinical Criteria Met:")
            if adj.get("primary_criteria_met"):
                for crit in adj["primary_criteria_met"]:
                    st.markdown(f"- ✅ **{crit}**")
            else:
                st.markdown("_No positive consensus criteria met._")

            st.markdown("#### 💬 Verbatim Clinical Evidence:")
            if adj.get("key_evidence"):
                for ev in adj["key_evidence"]:
                    quote = html.escape(str(ev.get("evidence_quote", "")))
                    interpretation = html.escape(str(ev.get("interpretation", "")))
                    st.markdown(f"""
                    <div class='evidence-card'>
                        "{quote}"<br>
                        <small style='color: #546e7a;'>📅 Date: {html.escape(str(ev.get('note_date', 'N/A')))} | Note ID: {html.escape(str(ev.get('note_id', 'N/A')))} | <i>{interpretation}</i></small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("_No positive text evidence excerpts found._")

            st.markdown("#### 🧠 Clinical Chain-of-Thought Rationale:")
            st.info(adj["clinical_rationale"])

            st.divider()
            st.markdown("#### 👨‍⚕️ Physician Verification Sign-off")

            col_r1, col_r2 = st.columns(2)
            with col_r1:
                rev_name = st.text_input("Reviewer Name", value="Dr. Gilles Clermont / Dr. Eric Rosenthal", key=f"rev_{selected_vid}")
            with col_r2:
                sign_action = st.selectbox(
                    "Adjudication Decision",
                    ["Agree with LLM", "Override: Mark Positive", "Override: Mark Negative", "Flag for Consensual Review"],
                    key=f"dec_{selected_vid}"
                )

            sign_comments = st.text_input(
                "Clinical Review Comments",
                placeholder="e.g. Verified urosepsis meeting Sepsis-3 criteria",
                key=f"comm_{selected_vid}"
            )

            if st.button("💾 Sign & Save Verification", type="primary", use_container_width=True):
                log_entry = {
                    "visit_occurrence_id": selected_vid,
                    "person_id": selected_record["person_id"],
                    "reviewer_id": rev_name,
                    "adjudication_status": status,
                    "llm_status": status,
                    "llm_confidence": conf,
                    "human_decision": sign_action,
                    "human_positive": "Positive" in sign_action if "Override" in sign_action else ("POSITIVE" in status),
                    "llm_positive": "POSITIVE" in status,
                    "reviewer_agreement": "Agree" in sign_action,
                    "override_reason": sign_comments if "Override" in sign_action else None,
                    "comments": sign_comments,
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                }
                saved_via = "local store"
                if api_online:
                    try:
                        resp = _http_session().post(f"{API_BASE_URL}/api/feedback", headers=_api_headers(), json=log_entry, timeout=20)
                        if resp.status_code == 200:
                            payload = resp.json()
                            saved_via = "GCS" if payload.get("durable") else payload.get("path", "API")
                        else:
                            append_audit(log_entry)
                    except Exception:
                        append_audit(log_entry)
                else:
                    append_audit(log_entry)
                fetch_audit_entries.clear()
                st.success(f"Verification saved for Visit {selected_vid} ({saved_via}).")

# ---------------------------------------------------------
# TAB 2: CALIBRATION & CONCORDANCE MATRIX
# ---------------------------------------------------------
with tab_calib:
    hdr_c1, hdr_c2 = st.columns([3, 1])
    with hdr_c1:
        st.markdown("### 📊 Dual-Review Calibration & Inter-Annotator Concordance")
        st.markdown("Evaluates agreement between **Human Gold-Standard Reviews** and **LLM Rose Gold Labels** across the multi-site calibration cohort.")
    with hdr_c2:
        st.write("")
        if st.button("🔄 Refresh Audit Log", use_container_width=True):
            fetch_audit_entries.clear()
            st.rerun()

    sample_eval_data = fetch_audit_entries(api_online)

    if not sample_eval_data:
        st.info("No human audit entries yet. Sign off encounters in Interactive Chart Review to populate concordance.")
        sample_eval_data = []

    df_eval = pd.DataFrame(sample_eval_data)
    if df_eval.empty:
        metrics = calculate_concordance_metrics(pd.DataFrame({"human_positive": [], "llm_positive": []}))
    else:
        if "human_positive" not in df_eval.columns:
            df_eval["human_positive"] = df_eval["reviewer_agreement"] if "reviewer_agreement" in df_eval.columns else False
        if "llm_positive" not in df_eval.columns:
            if "llm_status" in df_eval.columns:
                df_eval["llm_positive"] = df_eval["llm_status"].astype(str).str.contains("POSITIVE")
            else:
                df_eval["llm_positive"] = False
        metrics = calculate_concordance_metrics(df_eval)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cohen's Kappa (κ)", f"{metrics['cohens_kappa']:.3f}", f"N = {metrics['total_evaluated']} reviewed")
    c2.metric("Overall Concordance", f"{metrics['overall_agreement_pct']}%")
    c3.metric("Sensitivity", f"{metrics['sensitivity']}%")
    c4.metric("Specificity", f"{metrics['specificity']}%")

    st.write("")
    col_mat, col_log = st.columns([1, 1.3])

    with col_mat:
        st.markdown("#### 2x2 Diagnostic Confusion Matrix")
        matrix_df = pd.DataFrame({
            "LLM Positive": [f"TP: {metrics['tp']}", f"FP: {metrics['fp']}"],
            "LLM Negative": [f"FN: {metrics['fn']}", f"TN: {metrics['tn']}"]
        }, index=["Human Positive (True)", "Human Negative (True)"])
        st.table(matrix_df)
        st.caption(f"PPV (Precision): **{metrics['ppv']}%** | NPV: **{metrics['npv']}%**")

    with col_log:
        st.markdown("#### Recent Human Adjudication Logs")
        if not df_eval.empty:
            display_cols = [
                c for c in ["visit_occurrence_id", "reviewer_id", "human_decision", "reviewer_agreement", "comments", "timestamp"]
                if c in df_eval.columns
            ]
            st.dataframe(df_eval[display_cols].tail(12), use_container_width=True)
            audit_csv = df_eval.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Calibration Audit Log (.csv)",
                data=audit_csv,
                file_name="rosegold_calibration_audit_log.csv",
                mime="text/csv",
                use_container_width=True,
            )

# ---------------------------------------------------------
# TAB 3: BATCH ADJUDICATION & OMOP EXPORT
# ---------------------------------------------------------
with tab_batch:
    st.markdown(f"### ⚡ High-Throughput Cohort Adjudication ({len(records)} Encounters)")
    st.write(f"Processes clinical encounter note trajectories in batch for **{target_condition}**.")

    b_ctrl1, b_ctrl2 = st.columns([1.6, 1.4])
    with b_ctrl1:
        batch_scope = st.radio(
            "Batch Scope:",
            [
                f"Full Cohort ({len(records)} visits)",
                f"Quick Sample (First {min(5, len(records))} visits)",
            ],
            horizontal=True,
        )
    target_recs = records[:5] if batch_scope.startswith("Quick Sample") else records

    with b_ctrl2:
        st.write("")
        run_batch_btn = st.button(
            "▶️ Execute Full Cohort Batch Adjudication",
            type="primary",
            use_container_width=True,
        )

    if run_batch_btn:
        with st.spinner(f"Running batch adjudication on {len(target_recs)} visits..."):
            cohort_results = adjudicate_cohort(target_recs, target_condition, active_criteria)
        if cohort_results:
            st.session_state["cohort_results"] = cohort_results
            for item in cohort_results:
                vid = item.get("visit_occurrence_id")
                if vid is not None:
                    st.session_state[f"adj_{vid}_{target_condition}"] = item
            st.rerun()

    if "cohort_results" in st.session_state and st.session_state["cohort_results"]:
        results = st.session_state["cohort_results"]
        df_cohort = pd.DataFrame(results)

        st.success(f"Batch adjudication available for {len(df_cohort)} encounters.")

        b1, b2, b3, b4 = st.columns(4)
        pos_cnt = int((df_cohort["condition_present"] == True).sum()) if "condition_present" in df_cohort else 0
        neg_cnt = int((df_cohort["condition_present"] == False).sum()) if "condition_present" in df_cohort else 0
        avg_conf = float(df_cohort["confidence_score"].mean()) * 100 if "confidence_score" in df_cohort else 0.0
        b1.metric("Total Adjudicated", len(df_cohort))
        b2.metric("Confirmed Positive", f"{pos_cnt} ({pos_cnt/max(1, len(df_cohort))*100:.1f}%)")
        b3.metric("Confirmed Negative", f"{neg_cnt} ({neg_cnt/max(1, len(df_cohort))*100:.1f}%)")
        b4.metric("Mean Confidence", f"{avg_conf:.1f}%")

        bf1, bf2 = st.columns([1.2, 1.8])
        with bf1:
            status_filter = st.selectbox(
                "Filter Batch Results by Status:",
                ["All", "CONFIRMED_POSITIVE", "CONFIRMED_NEGATIVE", "SUSPECTED_PROBABLE", "INDETERMINATE_INSUFFICIENT_DATA"],
            )
        with bf2:
            batch_search = st.text_input("Search in Batch Rationales / Visit IDs:", "")

        df_view = df_cohort.copy()
        if status_filter != "All" and "phenotype_status" in df_view.columns:
            df_view = df_view[df_view["phenotype_status"] == status_filter]
        if batch_search.strip():
            q_str = batch_search.strip().lower()
            mask = (
                df_view["visit_occurrence_id"].astype(str).str.lower().str.contains(q_str, na=False)
                | df_view["clinical_rationale"].astype(str).str.lower().str.contains(q_str, na=False)
            )
            df_view = df_view[mask]

        cols_to_show = [
            c for c in ["visit_occurrence_id", "person_id", "phenotype_status", "confidence_score", "clinical_rationale", "inference_backend"]
            if c in df_view.columns
        ]
        st.dataframe(df_view[cols_to_show], use_container_width=True)

        st.markdown("#### 📥 Export Formats for Consortia & OMOP Ingestion:")
        col_e1, col_e2, col_e3 = st.columns(3)

        csv_omop, csv_rosegold, jsonl_data = build_export_payloads(
            json.dumps(results, sort_keys=True, default=str),
            target_condition,
        )
        col_e1.download_button(
            "📦 Download OMOP OBSERVATION Table (.csv)",
            data=csv_omop,
            file_name=f"omop_observation_rosegold_{target_condition.split('/')[0].strip().lower().replace(' ', '_')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        col_e2.download_button(
            "📄 Download Rose Gold Labels (.csv)",
            data=csv_rosegold,
            file_name="rose_gold_adjudications.csv",
            mime="text/csv",
            use_container_width=True
        )
        col_e3.download_button(
            "📑 Download Evidence JSONL (.jsonl)",
            data=jsonl_data,
            file_name="rose_gold_adjudications.jsonl",
            mime="application/jsonl",
            use_container_width=True
        )

# ---------------------------------------------------------
# TAB 4: PHENOTYPE RULES & PROMPTS
# ---------------------------------------------------------
with tab_rules:
    st.markdown("### 🛠️ Phenotype Criteria & Adjudication Guidelines")
    st.markdown("Define or customize the clinical rules and consensus guidelines for model evaluation.")

    default_for_cond = resolve_criteria(target_condition)
    rule_text = st.text_area(
        f"Consensus Clinical Definition for: {target_condition}",
        value=active_criteria,
        height=240
    )

    r_btn1, r_btn2 = st.columns([1.5, 1.0])
    with r_btn1:
        if st.button("💾 Save Updated Phenotype Criteria", type="primary", use_container_width=True):
            st.session_state["custom_criteria"] = rule_text
            st.session_state[criteria_state_key] = rule_text
            saved_via = "local store"
            path = None
            if api_online:
                try:
                    resp = _http_session().post(
                        f"{API_BASE_URL}/api/criteria",
                        headers=_api_headers(),
                        json={"text": rule_text},
                        timeout=20
                    )
                    if resp.status_code == 200:
                        payload = resp.json()
                        path = payload.get("path")
                        saved_via = "GCS" if payload.get("durable") else path or "API"
                    else:
                        path = save_criteria(rule_text)
                except Exception:
                    path = save_criteria(rule_text)
            else:
                path = save_criteria(rule_text)
            st.success(f"Criteria saved ({saved_via}{': ' + path if path else ''}). Subsequent adjudications will use this definition.")

    with r_btn2:
        if st.button("🔄 Reset to Default Consensus Criteria", use_container_width=True):
            st.session_state.pop("custom_criteria", None)
            st.session_state.pop(criteria_state_key, None)
            st.rerun()

# ---------------------------------------------------------
# TAB 5: CLI & FASTAPI TUTORIAL
# ---------------------------------------------------------
with tab_tutorial:
    st.markdown("### 📖 Developer & Site Deployment Tutorial")
    st.markdown("Complete guide for institutional IT teams and clinical data scientists to run Rose Gold via **CLI Batch Mode**, **FastAPI REST Service**, **Docker / Singularity**, or **Site-Approved Hosted LLMs**.")

    st.markdown("---")
    st.markdown("#### 1️⃣ CLI Mode: High-Throughput Offline Batch Processing")
    st.write("Best for overnight or automated processing of 20,000+ OMOP clinical notes directly on GPU or CPU servers with automatic checkpoint resumption.")

    st.code("""# Run offline batch adjudication directly against OMOP CSV or Parquet files
python -m app.adjudicator \\
  --notes_path data/synthetic_notes.csv \\
  --visits_path data/synthetic_visits.csv \\
  --output_path outputs/rose_gold_adjudications.csv \\
  --backend auto \\
  --target_condition "Sepsis / Septic Shock" \\
  --chunk_size 50""", language="bash")

    st.markdown("---")
    st.markdown("#### 2️⃣ FastAPI Mode: Programmatic Integration")
    st.write("Best for integrating Rose Gold adjudication into local EHR data pipelines, REDCap, or hospital research platforms.")

    st.markdown("**Start the FastAPI Server:**")
    st.code("uvicorn app.api:app --host 0.0.0.0 --port 8000", language="bash")
    st.caption("Interactive Swagger API Documentation available at: `http://localhost:8000/docs`")

    col_curl, col_py = st.columns(2)

    with col_curl:
        st.markdown("**Example A: cURL Single Adjudication Request**")
        st.code("""curl -X POST "http://localhost:8000/api/adjudicate/single" \\
  -H "Content-Type: application/json" \\
  -d '{
    "visit_occurrence_id": 20001,
    "target_condition": "Sepsis / Septic Shock"
  }'""", language="bash")

    with col_py:
        st.markdown("**Example B: Python API Client**")
        st.code("""import requests

url = "http://localhost:8000/api/adjudicate/single"
payload = {
    "visit_occurrence_id": 20001,
    "target_condition": "Sepsis / Septic Shock"
}
response = requests.post(url, json=payload).json()
print("Status:", response["phenotype_status"])
print("Confidence:", response["confidence_score"])
print("Evidence:", response["key_evidence"])""", language="python")

    st.markdown("---")
    st.markdown("#### 3️⃣ Docker & Singularity / Apptainer Containers (GPU & CPU)")
    st.write("Standardized rootless-compatible containers for hospital servers and HPC SLURM clusters.")

    col_dock, col_sing = st.columns(2)
    with col_dock:
        st.markdown("**Docker (GPU or CPU):**")
        st.code("""# CPU / Cloud Run lightweight image (baked Llama 3.2 3B GGUF)
docker build -f Dockerfile.cpu -t rosegold-cpu:latest .
docker run -p 8000:8000 -p 8501:8501 rosegold-cpu:latest""", language="bash")

    with col_sing:
        st.markdown("**Singularity / Apptainer (HPC Clusters):**")
        st.code("""# Build & run rootless SIF on HPC (with or without --nv GPU flag)
singularity build rosegold_cpu.sif singularity/rosegold_cpu.def
singularity run -B $PWD/data:/workspace/data rosegold_cpu.sif""", language="bash")

    st.markdown("---")
    st.markdown("#### 4️⃣ Site-Hosted Cloud LLMs (Bedrock Mantle, Databricks, Vertex AI)")
    st.code("""# AWS Bedrock Mantle (OpenAI-compatible endpoint)
export ROSEGOLD_LLM_BACKEND=mantle
export AWS_REGION=us-west-2
export AWS_BEARER_TOKEN_BEDROCK="your-site-bedrock-key"
export ROSEGOLD_OPENAI_MODEL="openai.gpt-oss-120b"

# Databricks Model Serving
export ROSEGOLD_LLM_BACKEND=databricks
export DATABRICKS_HOST="https://adb-xxxx.azuredatabricks.net"
export DATABRICKS_TOKEN="dapi..."
export ROSEGOLD_OPENAI_MODEL="site-llama-endpoint" """, language="bash")
