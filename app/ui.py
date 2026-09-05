import sys
import os

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import html
import streamlit as st
import pandas as pd
import requests
import json
import datetime
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
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a237e;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #546e7a;
        margin-bottom: 20px;
    }
    .status-positive {
        background-color: #ffebee;
        border: 2px solid #e53935;
        color: #b71c1c;
        padding: 12px 20px;
        border-radius: 10px;
        font-weight: 700;
        font-size: 1.15rem;
        text-align: center;
        margin-bottom: 15px;
    }
    .status-negative {
        background-color: #e8f5e9;
        border: 2px solid #43a047;
        color: #1b5e20;
        padding: 12px 20px;
        border-radius: 10px;
        font-weight: 700;
        font-size: 1.15rem;
        text-align: center;
        margin-bottom: 15px;
    }
    .status-suspected {
        background-color: #fff8e1;
        border: 2px solid #ffa000;
        color: #e65100;
        padding: 12px 20px;
        border-radius: 10px;
        font-weight: 700;
        font-size: 1.15rem;
        text-align: center;
        margin-bottom: 15px;
    }
    .evidence-card {
        background-color: #f1f8ff;
        border-left: 5px solid #0366d6;
        padding: 14px;
        border-radius: 6px;
        margin-bottom: 12px;
        font-size: 0.95rem;
    }
    .note-card {
        background-color: #ffffff;
        border: 1px solid #e1e4e8;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 14px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-badge {
        background-color: #f0f4f8;
        padding: 6px 12px;
        border-radius: 15px;
        font-size: 0.85rem;
        font-weight: 600;
        color: #24292e;
        display: inline-block;
        margin-right: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Environment & Backend Status
# ---------------------------------------------------------
API_BASE_URL = os.getenv("ROSEGOLD_API_URL", "http://localhost:8000")


def _api_headers():
    """Shared-secret header when the API is protected by ROSEGOLD_API_KEY."""
    key = os.getenv("ROSEGOLD_API_KEY", "").strip()
    return {"X-API-Key": key} if key else {}

@st.cache_data(ttl=5)
def check_api():
    try:
        r = requests.get(f"{API_BASE_URL}/health", headers=_api_headers(), timeout=1.2)
        if r.status_code == 200:
            return True, r.json()
    except Exception:
        pass
    return False, None

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

# Dataset paths typed into the sidebar are validated against ROSEGOLD_DATA_DIR
# exactly like the API's notes_path/visits_path: the dashboard is the public
# surface on Cloud Run and must not be able to read arbitrary files.
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

notes_path_input = st.sidebar.text_input("Notes table", value=default_notes, key=f"notes_{dataset}")
visits_path_input = st.sidebar.text_input("Visits table", value=default_visits, key=f"visits_{dataset}")
st.sidebar.caption(f"Files must be `.csv`/`.parquet` inside `{DATA_ROOT}`.")

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
    notes_is_mimic = looks_like_mimic_ext_notes_path(notes_path)
except Exception:
    st.sidebar.error("Notes table could not be read. Check that it is a valid CSV/Parquet file.")
    st.stop()

if notes_is_mimic:
    st.sidebar.caption("Detected MIMIC-III-Ext-Notes notes.csv. Visits are derived from hadm_id.")
elif visits_path is None:
    st.sidebar.error("Visits table must be a .csv/.parquet file inside the configured data directory.")
    st.stop()

target_condition = st.sidebar.selectbox(
    "Target Clinical Phenotype",
    [
        "Sepsis / Septic Shock",
        "Acute Ischemic Stroke",
        "Acute Respiratory Distress Syndrome (ARDS)",
        "Acute Kidney Injury (AKI)"
    ]
)

hw_info = detect_hardware()

@st.cache_resource
def get_local_engine():
    return AdjudicationEngine(model_name="auto")

st.sidebar.markdown("### 🖥️ Hardware Auto-Detection")
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

# Resolve selected configuration
if "Auto-Select" in llm_backend:
    resolved = resolve_model_and_engine(requested_family="llama")
    st.sidebar.caption(f"⚡ Active: **{resolved['selected_model']}** on {resolved['target_device'].upper()}")
elif "Hybrid" in llm_backend:
    resolved = resolve_model_and_engine(requested_model="muse-glimmer-30b", force_device="cuda")
    st.sidebar.caption(f"⚡ Active: **Rose Gold Hybrid (Rules + Muse-30B)**")
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

_store = storage_status()
if _store["durable"]:
    st.sidebar.success(f"💾 **Persistent GCS:** `{_store['gcs_bucket'] or _store['output_dir']}`")
else:
    st.sidebar.info(f"💾 **Local store:** `{_store['output_dir']}`")

if "custom_criteria" not in st.session_state:
    saved_criteria = None
    if api_online:
        try:
            resp = requests.get(f"{API_BASE_URL}/api/criteria", headers=_api_headers(), timeout=10)
            if resp.status_code == 200:
                saved_criteria = resp.json().get("text")
        except Exception:
            saved_criteria = None
    if not saved_criteria:
        saved_criteria = load_criteria()
    if saved_criteria:
        st.session_state["custom_criteria"] = saved_criteria

# Load Cohort
@st.cache_data
def load_cohort(n_p, v_p):
    if not os.path.exists(n_p):
        return []
    if looks_like_mimic_ext_notes_path(n_p) or (v_p and os.path.exists(v_p)):
        return load_omop_data(n_p, v_p)
    return []

try:
    records = load_cohort(notes_path, visits_path)
except Exception:
    # Malformed CSV/Parquet: keep the traceback in the server log, not the browser.
    st.error("The selected dataset could not be parsed. Check that it follows the OMOP NOTE / VISIT_OCCURRENCE schema.")
    st.stop()

if not records:
    st.error("No OMOP data found. Please check data file paths.")
    st.stop()

active_criteria = st.session_state.get("custom_criteria") or resolve_criteria(target_condition)


def adjudicate_record(record, condition, criteria):
    """Adjudicate one visit. Returns None (after showing an error) instead of raising.

    When the API is online its answer is authoritative; the dashboard never
    silently substitutes an in-process engine for a failed API call, because
    that could turn a configured LLM deployment into keyword-rule labels.
    """
    if api_online:
        try:
            resp = requests.post(
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
        return resp.json()
    try:
        return get_local_engine().adjudicate_single(record, condition, criteria)
    except RuntimeError:
        st.error("The configured LLM backend is not available in this process. Start the API service or fix the backend configuration.")
        return None


def adjudicate_cohort(recs, condition, criteria):
    """Batch adjudicate. Returns None (after showing an error) instead of raising."""
    if api_online:
        try:
            resp = requests.post(
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
        return resp.json()
    try:
        results = get_local_engine().adjudicate_batch(recs, condition, criteria)
    except RuntimeError:
        st.error("The configured LLM backend is not available in this process. Start the API service or fix the backend configuration.")
        return None
    save_batch_results(results, condition)
    return results

# ---------------------------------------------------------
# Top Header & KPI Summary
# ---------------------------------------------------------
st.markdown("<div class='main-header'>🩺 Rose Gold: Clinical LLM Chart Adjudication Suite</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>High-Throughput On-Premises Clinical Phenotyping for OMOP CDM (MGH • Pitt • UTHealth • Duke • Emory • Tufts • UVA)</div>", unsafe_allow_html=True)

# Top KPI Summary Cards
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("Encounter Cohort", f"{len(records)} Visits")
total_notes_count = sum(r['note_count'] for r in records)
kpi2.metric("Total Notes", f"{total_notes_count} Notes")
kpi3.metric("Target Phenotype", target_condition.split('/')[0].strip())
kpi4.metric("LLM Engine", resolved.get("engine_type", "Auto"))
if api_online and (health_data or {}).get("backend") == "llamacpp":
    kpi5.metric("LLM Path", "Llama 3.2 3B CPU")
elif api_online and (health_data or {}).get("backend") == "vertex":
    kpi5.metric("LLM Path", "Hosted API (not Llama)")
elif api_online and (health_data or {}).get("llm_real"):
    kpi5.metric("PHI Boundary", "On-prem LLM")
else:
    kpi5.metric("PHI Boundary", "100% On-Premises")

st.divider()

# ---------------------------------------------------------
# Tabs: Review | Calibration | Batch | Definitions
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
    # Patient Encounter Selector
    visit_dict = {
        r['visit_occurrence_id']: f"Visit {r['visit_occurrence_id']} | Person {r['person_id']} ({r['note_count']} notes: {r['visit_start_date']})"
        for r in records
    }
    
    col_sel1, col_sel2 = st.columns([2.5, 1.5])
    with col_sel1:
        selected_vid = st.selectbox(
            "Select Patient Visit Encounter to Review:",
            list(visit_dict.keys()),
            format_func=lambda x: visit_dict[x]
        )
    selected_record = next(r for r in records if r['visit_occurrence_id'] == selected_vid)

    with col_sel2:
        st.write("")
        st.write("")
        run_btn = st.button("🚀 Adjudicate this Encounter", type="primary", use_container_width=True)

    col_notes_view, col_adj_view = st.columns([1.15, 0.85])

    # Left: Medical Record Timeline
    with col_notes_view:
        st.markdown(f"### 📄 Medical Record Timeline (Visit `{selected_vid}`)")
        st.markdown(
            f"<span class='metric-badge'>👤 Person ID: {html.escape(str(selected_record['person_id']))}</span>"
            f"<span class='metric-badge'>📅 {html.escape(str(selected_record['visit_start_date']))} to {html.escape(str(selected_record['visit_end_date']))}</span>"
            f"<span class='metric-badge'>📝 {html.escape(str(selected_record['note_count']))} Notes</span>",
            unsafe_allow_html=True
        )

        filter_kw = st.text_input("🔎 Search / Filter keywords in patient notes (e.g. 'lactate', 'norepinephrine', 'stroke'):", "")
        raw_text = selected_record['notes_formatted_text']
        
        # Display notes as distinct chronological cards
        note_chunks = raw_text.split("--- [Note ID:")
        for chunk in note_chunks:
            if not chunk.strip():
                continue
            chunk_full = "--- [Note ID:" + chunk if not chunk.startswith("---") else chunk
            lines = chunk_full.strip().split("\n")
            header = lines[0] if lines else "Note"
            body = "\n".join(lines[1:]) if len(lines) > 1 else ""

            if filter_kw and filter_kw.lower() not in chunk_full.lower():
                continue

            st.markdown(f"""
            <div class='note-card'>
                <strong>{html.escape(header)}</strong>
                <hr style='margin: 6px 0;'>
                <div style='white-space: pre-wrap; font-family: -apple-system, sans-serif; font-size: 0.92rem; color: #2c3e50;'>{html.escape(body)}</div>
            </div>
            """, unsafe_allow_html=True)

    # Right: LLM Adjudication Panel
    with col_adj_view:
        st.markdown("### 🤖 Rose Gold Adjudication")

        session_key = f"adj_{selected_vid}_{target_condition}"
        if run_btn:
            with st.spinner("Running adjudication..."):
                adjudication = adjudicate_record(selected_record, target_condition, active_criteria)
            if adjudication is not None:
                st.session_state[session_key] = adjudication

        if session_key not in st.session_state:
            st.info("Click **Adjudicate this Encounter** to run the engine. Charts are not scored on page load.")
        else:
            adj = st.session_state[session_key]
            status = adj["phenotype_status"]
            conf = adj["confidence_score"]

            if "CONFIRMED_POSITIVE" in status:
                st.markdown(f"<div class='status-positive'>⚠️ {html.escape(status)}<br><small>Confidence: {conf*100:.1f}%</small></div>", unsafe_allow_html=True)
            elif "CONFIRMED_NEGATIVE" in status:
                st.markdown(f"<div class='status-negative'>✅ {html.escape(status)}<br><small>Confidence: {conf*100:.1f}%</small></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='status-suspected'>❓ {html.escape(status)}<br><small>Confidence: {conf*100:.1f}%</small></div>", unsafe_allow_html=True)

            st.progress(conf, text=f"Model Diagnostic Confidence: {conf*100:.1f}%")

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
            backend_used = adj.get("inference_backend") or (health_data or {}).get("backend")
            if backend_used:
                if str(backend_used).startswith("keyword"):
                    st.error("This label came from keyword rules, not a language model.")
                else:
                    st.caption(f"Inference backend: `{backend_used}`")

            st.divider()
            st.markdown("#### 👨‍⚕️ Physician Verification Sign-off")

            col_r1, col_r2 = st.columns(2)
            with col_r1:
                rev_name = st.text_input("Reviewer Name", value="Dr. Gilles Clermont / Dr. Eric Rosenthal", key=f"rev_{selected_vid}")
            with col_r2:
                sign_action = st.selectbox("Adjudication Decision", ["Agree with LLM", "Override: Mark Positive", "Override: Mark Negative", "Flag for Consensual Review"], key=f"dec_{selected_vid}")

            sign_comments = st.text_input("Clinical Review Comments", placeholder="e.g. Verified urosepsis meeting Sepsis-3 criteria", key=f"comm_{selected_vid}")

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
                        resp = requests.post(f"{API_BASE_URL}/api/feedback", headers=_api_headers(), json=log_entry, timeout=20)
                        if resp.status_code == 200:
                            payload = resp.json()
                            saved_via = "GCS" if payload.get("durable") else payload.get("path", "API")
                        else:
                            append_audit(log_entry)
                    except Exception:
                        append_audit(log_entry)
                else:
                    append_audit(log_entry)
                st.success(f"Verification saved for Visit {selected_vid} ({saved_via}).")

# ---------------------------------------------------------
# TAB 2: CALIBRATION & CONCORDANCE MATRIX
# ---------------------------------------------------------
with tab_calib:
    st.markdown("### 📊 Dual-Review Calibration & Inter-Annotator Concordance")
    st.markdown("Evaluates agreement between **Human Gold-Standard Reviews** and **LLM Rose Gold Labels** across the multi-site calibration cohort.")

    sample_eval_data = []
    if api_online:
        try:
            resp = requests.get(f"{API_BASE_URL}/api/audit", headers=_api_headers(), timeout=10)
            if resp.status_code == 200:
                sample_eval_data = resp.json().get("entries") or []
        except Exception:
            sample_eval_data = []
    if not sample_eval_data:
        sample_eval_data = read_audit()

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
    c1.metric("Cohen's Kappa (κ)", f"{metrics['cohens_kappa']:.3f}", "Strong Agreement")
    c2.metric("Overall Concordance", f"{metrics['overall_agreement_pct']}%")
    c3.metric("Sensitivity", f"{metrics['sensitivity']}%")
    c4.metric("Specificity", f"{metrics['specificity']}%")

    st.write("")
    col_mat, col_log = st.columns([1, 1.2])

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
            display_cols = [c for c in ['visit_occurrence_id', 'reviewer_id', 'human_decision', 'reviewer_agreement', 'comments'] if c in df_eval.columns]
            st.dataframe(df_eval[display_cols].tail(8), use_container_width=True)

# ---------------------------------------------------------
# TAB 3: BATCH ADJUDICATION & OMOP EXPORT
# ---------------------------------------------------------
with tab_batch:
    st.markdown(f"### ⚡ High-Throughput Cohort Adjudication ({len(records)} Encounters)")
    st.write(f"Processes all {len(records)} clinical encounter note trajectories in parallel for **{target_condition}**.")

    if st.button("▶️ Execute Full Cohort Batch Adjudication", type="primary"):
        with st.spinner(f"Running high-throughput batch adjudication on {len(records)} visits..."):
            cohort_results = adjudicate_cohort(records, target_condition, active_criteria)
        if cohort_results:
            st.session_state["cohort_results"] = cohort_results

    if "cohort_results" not in st.session_state:
        persisted = []
        if api_online:
            try:
                resp = requests.get(f"{API_BASE_URL}/api/batch", headers=_api_headers(), timeout=10)
                if resp.status_code == 200:
                    persisted = resp.json().get("results") or []
            except Exception:
                persisted = []
        if not persisted:
            persisted = load_batch_results()
        if persisted:
            st.session_state["cohort_results"] = persisted
            st.caption("Loaded last persisted cohort from the durable store.")

    if "cohort_results" in st.session_state:
        results = st.session_state["cohort_results"]
        df_cohort = pd.DataFrame(results)
        
        st.success(f"Batch adjudication complete for all {len(df_cohort)} encounters!")

        b1, b2, b3 = st.columns(3)
        pos_cnt = len(df_cohort[df_cohort['condition_present'] == True])
        neg_cnt = len(df_cohort[df_cohort['condition_present'] == False])
        b1.metric("Total Adjudicated", len(df_cohort))
        b2.metric("Confirmed Positive", f"{pos_cnt} ({pos_cnt/len(df_cohort)*100:.1f}%)")
        b3.metric("Confirmed Negative", f"{neg_cnt} ({neg_cnt/len(df_cohort)*100:.1f}%)")

        st.dataframe(
            df_cohort[['visit_occurrence_id', 'person_id', 'phenotype_status', 'confidence_score', 'clinical_rationale']],
            use_container_width=True
        )

        st.markdown("#### 📥 Export Formats for Consortia & OMOP Ingestion:")
        col_e1, col_e2, col_e3 = st.columns(3)

        # 1. OMOP CDM Observation Table
        df_omop_obs = export_to_omop_observation(results, target_condition)
        csv_omop = df_omop_obs.to_csv(index=False).encode('utf-8')
        col_e1.download_button(
            "📦 Download OMOP OBSERVATION Table (.csv)",
            data=csv_omop,
            file_name=f"omop_observation_rosegold_{target_condition.split('/')[0].strip().lower()}.csv",
            mime="text/csv",
            use_container_width=True
        )

        # 2. Detailed Rose Gold CSV
        csv_rosegold = df_cohort.to_csv(index=False).encode('utf-8')
        col_e2.download_button(
            "📄 Download Rose Gold Labels (.csv)",
            data=csv_rosegold,
            file_name="rose_gold_adjudications.csv",
            mime="text/csv",
            use_container_width=True
        )

        # 3. JSON Lines (with full quotes)
        jsonl_data = "\n".join([json.dumps(r) for r in results]).encode('utf-8')
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

    rule_text = st.text_area(
        f"Consensus Clinical Definition for: {target_condition}",
        value=st.session_state.get("custom_criteria") or resolve_criteria(target_condition),
        height=240
    )

    if st.button("💾 Save Updated Phenotype Criteria"):
        st.session_state["custom_criteria"] = rule_text
        saved_via = "local store"
        path = None
        if api_online:
            try:
                resp = requests.post(f"{API_BASE_URL}/api/criteria", headers=_api_headers(), json={"text": rule_text}, timeout=20)
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

# ---------------------------------------------------------
# TAB 5: CLI & FASTAPI TUTORIAL
# ---------------------------------------------------------
with tab_tutorial:
    st.markdown("### 📖 Developer & Site Deployment Tutorial")
    st.markdown("Complete guide for institutional IT teams and clinical data scientists to run Rose Gold via **CLI Batch Mode** or **FastAPI REST Service**.")

    st.markdown("---")
    st.markdown("#### 1️⃣ CLI Mode: High-Throughput Offline Batch Processing")
    st.write("Best for overnight or automated processing of 20,000+ OMOP clinical notes directly on GPU servers with zero HTTP overhead.")
    
    st.code("""# Run offline batch adjudication directly against OMOP CSV or Parquet files
python -m app.adjudicator \\
  --notes_path /path/to/OMOP_NOTE.parquet \\
  --visits_path /path/to/OMOP_VISIT_OCCURRENCE.parquet \\
  --output_path outputs/rose_gold_adjudications.csv \\
  --model_name meta-llama/Llama-3.1-8B-Instruct \\
  --target_condition "Sepsis / Septic Shock" \\
  --tensor_parallel_size 1 \\
  --max_model_len 32768""", language="bash")

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
    st.markdown("#### 3️⃣ Docker Container Deployment (Multi-Site Consortium)")
    st.write("Standardized container ensures identical model weights, prompts, and schemas across all participating sites.")

    st.code("""# 1. Build or pull the container
docker build -t rosegold-suite:latest .

# 2. Run with GPU support and mount local hospital OMOP data
docker run --gpus all -d \\
  -p 8000:8000 -p 8501:8501 \\
  -v /hospital_data/omop:/workspace/data \\
  -v /hospital_data/outputs:/workspace/outputs \\
  -e HUGGING_FACE_HUB_TOKEN="your_hf_token" \\
  rosegold-suite:latest""", language="bash")

    st.markdown("---")
    st.markdown("#### 4️⃣ OMOP CDM Data Ingestion & Export Schema")
    st.markdown("""
- **Inputs required:** Standard OMOP `NOTE` (`note_id`, `person_id`, `visit_occurrence_id`, `note_date`, `note_text`) and `VISIT_OCCURRENCE` (`visit_occurrence_id`, `person_id`, `visit_start_date`, `visit_end_date`).
- **Output:** Standard OMOP CDM `OBSERVATION` table with concept IDs:
  - Sepsis Concept: `132797`
  - Acute Ischemic Stroke Concept: `443454`
  - Value Present Concept: `4181412`
  - Value Absent Concept: `4188540`
    """)
