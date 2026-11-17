# CHORUS Chart-to-REDCap Abstraction & Clinician Concordance

Rose Gold provides two complementary modules for running and evaluating CHORUS-aligned chart abstraction at a participating hospital site using only approved on-premises or hospital-hosted LLM endpoints:

1. **`app.chorus_abstractor`**: Ingests raw OMOP/EHR notes, visits, and optional structured measurements, medications, and microbiology cultures; enforces staged clinician-parity screening (`Stage 1` discharge summary 3-way gate -> `Stage 2` uncertain H&P/labs/micro/meds follow-up -> `Stage 3` full panel + deterministic calculators); and exports REDCap-importable coded CSVs plus a grounded evidence sidecar.
2. **`app.chorus_concordance`**: Compares clinician REDCap exports against Rose Gold REDCap outputs across 3-way gates, post-uncertain resolved presence (including stratification by clinician `conf1`), per-item checkboxes and culture organism overlap, quantitative metrics (`sep6_v3` lactate, `sep7_v3` daily SOFA), and onset timing (`sep_day_v3`, `sep_days`, `sep_dt_v3`).

---

## 1. Input Data Contract

All input files stay strictly inside the local site environment and accept `.csv` or `.parquet`:

| Input | Flag | Required Columns | Notes |
| --- | --- | --- | --- |
| **Clinical Notes** | `--notes` | `visit_occurrence_id`, `note_text`, `note_date` or `note_datetime` | Optional `note_id`, `note_title` (used to classify panels: `discharge_summary`, `hp`, `procedure_operative`, `nursing`, `microbiology`, `progress_consult`). |
| **Visits** | `--visits` | `visit_occurrence_id`, `person_id` | Optional `record_id`, `institution_v3`, `icu_admit_datetime`, `visit_start_datetime`, `visit_start_date`, `visit_end_date`. |
| **REDCap Dictionary** | `--dictionary` | Official REDCap columns (`Variable / Field Name`, `Field Type`, `Field Label`, `Choices, Calculations, OR Slider Labels`, `Branching Logic (Show field only if...)`) | Choice codes and branching logic are driven directly from the canonical dictionary. |
| **Measurements (optional)** | `--measurements` | `visit_occurrence_id`, `measurement_datetime`, `measurement_name`, `value_as_number` or `value_as_string`, `unit` | Covers Temp (`°C` or `°F`), HR, RR, `PaCO2`, WBC, `% bands`, Lactate, `PaO2`, `FiO2`, ventilation/CPAP/BiPAP, platelets, bilirubin, MAP, GCS, creatinine. |
| **Medications (optional)** | `--medications` | `visit_occurrence_id`, `drug_start_datetime`, `drug_end_datetime`, `drug_name` | Covers systemic antibiotics (supporting >24h span merging with antibiotic narrowing and up to 48h vancomycin/aminoglycoside trough pauses) and vasopressors (`norepinephrine`, `epinephrine`, `dopamine`, `dobutamine`, `vasopressin`, with optional `dose_rate`). |
| **Cultures (optional)** | `--cultures` | `visit_occurrence_id`, `collection_datetime`, `source`, `organism`, `result` | Used to deterministically build `sep3_v3` tuples (`Source, Pathogen, (+)`) or `NO` when negative cultures are documented. |

---

## 2. Staged Clinician Parity & Deterministic Calculations

To maintain parity with clinician chart review and avoid silent false negatives:

- **Stage 1 (`sepsis_ync_v3`)**: Exposes **only** the discharge summary to decide `1` (YES), `2` (NO), or `3` (I am not sure). Missing discharge summaries automatically route to `3`.
- **Stage 2 (`sepsis_uncertain_v3`, only when Stage 1 = `3`)**: Exposes initial H&P plus structured admission labs, microbiology, and medications (`1` = YES, `2` = NO, or `UNKNOWN`).
- **Stage 3 (when Sepsis branch is open)**: Runs deterministic calculators on structured EHR tables (`sep1_v3___1..4`, `sep3_v3`, `sep4_v3`, `sep6_v3`, `sep7_v3`, `sep_day_v3`, `sep_dt_v3`, `sep_days`, `sep_time`) combined with a single structured JSON LLM call for `sep5_v3` unstructured clinical evidence, `sep_surg_v3` surgical/vasoplegia/burn/trauma confounders (`1,2,3,4,9` with mutual-exclusivity normalization), and note-based fallback when structured tables are omitted.
- **Missingness & Privacy Guards**:
  - Unmeasured labs/criteria remain blank (`""`) rather than defaulting to `NO` (`"0"`).
  - Explicit calendar dates are scrubbed from CHoRUS-facing text (`[DATE REDACTED]`) and replaced by relative `Day +d HH:MM` offsets during inference.
  - Human-only metadata (`reviewer_id`, `conf1`–`conf8`, `t1art_end`, survey timers) remain empty (`""`) for actual clinician reviewers.
  - Transport or JSON schema failures preserve `INDETERMINATE_INSUFFICIENT_DATA` and never coerce an encounter to `NO`.

---

## 3. Running Locally (Site-Approved Backends)

### Step A: Chart-to-REDCap Abstraction

With a hospital-approved OpenAI-compatible, AWS Bedrock Mantle, Databricks, Vertex Gemini, or Hybrid Muse endpoint:

```sh
export ROSEGOLD_LLM_BACKEND="mantle"
export AWS_REGION="us-west-2"
export AWS_BEARER_TOKEN_BEDROCK="your-approved-key"
export ROSEGOLD_OPENAI_MODEL="your-approved-model-id"

python -m app.chorus_abstractor \
  --notes /secure/path/notes.csv \
  --visits /secure/path/visits.csv \
  --measurements /secure/path/measurements.csv \
  --medications /secure/path/medications.csv \
  --cultures /secure/path/cultures.csv \
  --dictionary /secure/path/instrument.csv \
  --institution-code 9 \
  --output-dir outputs/chorus_sepsis_run
```

For synthetic offline testing (no PHI, no remote API calls):

```sh
ROSEGOLD_LLM_BACKEND=keyword_rules ROSEGOLD_ALLOW_MOCK=1 \
python -m app.chorus_abstractor \
  --notes data/synthetic_notes.csv \
  --visits data/synthetic_visits.csv \
  --dictionary /path/to/instrument.csv \
  --output-dir outputs/synthetic_chorus_demo
```

Outputs in `--output-dir`:
- `redcap_import.csv`: Coded REDCap-ready columns with formula-safe escaping and human-only fields left blank.
- `rosegold_adjudications.csv`: Encounter-level `CONFIRMED_POSITIVE` / `SUSPECTED_PROBABLE` / `CONFIRMED_NEGATIVE` / `INDETERMINATE_INSUFFICIENT_DATA` status, confidence, rationale, and review flags (including Sepsis-3 $\Delta\text{SOFA} = 2$ vs. `> 2` alerts and midnight boundary notices).
- `field_evidence.csv`: Per-field verbatim grounded quotes and structured row references.
- `run_manifest.json`: SHA-256 dictionary digest, backend configuration, and encounter counts.

### Step B: Clinician-vs-LLM Concordance

Compare clinician REDCap exports against Rose Gold's `redcap_import.csv`:

```sh
python -m app.chorus_concordance \
  --human-csv /secure/path/clinician_redcap_export.csv \
  --llm-csv outputs/chorus_sepsis_run/redcap_import.csv \
  --dictionary /secure/path/instrument.csv \
  --llm-evidence outputs/chorus_sepsis_run/field_evidence.csv \
  --output-dir outputs/chorus_concordance_report
```

Outputs in `--output-dir`:
- `concordance_report.md`: Summary tables for the 3-way `sepsis_ync_v3` matrix (exact agreement, unweighted $\kappa$, linear-weighted $\kappa$), resolved Sepsis sensitivity/specificity/$\kappa$ (plus stratification by `conf1`), per-item checkbox agreement (`sep1_v3___1..4`, `sep4_v3`, `sep_surg_v3___1..9`, `sep3_v3_positive`, organism token Jaccard), and MAE/accuracy for lactate, daily SOFA, and onset timing.
- `concordance_summary.json`: Machine-readable metrics dictionary.
- `discrepancies.csv`: Row-by-row mismatch list paired with clinician `conf1` and Rose Gold's grounded `evidence_quote`.
