# CHORUS survey review with required Muse adjudication

The `app.chorus_review` command runs the actual Rosegold pipeline:
`AdjudicationEngine` to `HybridAdjudicationEngine`, with Muse required for every
phenotype. Rules supply a candidate for Muse to review. Failed or unsupported
responses remain indeterminate, and unresolved source checks produce a quality hold.

This adapter assesses support in submitted reviewer questionnaires. It does not
replace original chart review or establish independently validated diagnoses.
The default hybrid cascade remains available for other workflows.

## Local inputs

Supply two local CSV files:

1. A raw-coded REDCap data export, with `institution_v3`, `record_id`, `vid`, and
   checkbox columns named `field___code`. Export codes, rather than label text.
2. The matching ordered REDCap data dictionary, including `Variable / Field Name`,
   `Field Type`, `Field Label`, and `Choices, Calculations, OR Slider Labels`.

The mapping targets the adult instrument fields listed in `PHENOTYPES`. The
cardiac-arrest section runs from `did_the_patient_visit_have` to `mortality_ca`.
Validate a different instrument version before inference. Unknown radio codes,
missing linkage identifiers, duplicate dictionary fields and missing section
boundaries are rejected. A CSV row number includes its header as row one.

`record_id` is assumed to identify a patient within an institution. Multiple rows
for the same composite patient key are supplied jointly, with current and linked
stays marked separately. Reused visit IDs never merge different patients.
Chronology and prediction cutoffs are unverified, so linked stays provide context
only. Obtain a confirmed crosswalk and dated visits before longitudinal validation.
The numeric IDs returned by the engine are local surrogate IDs, not source IDs.
The review CSV retains the original identifiers for local review.

## Prepare and run

Use the existing Python environment installed from `requirements.txt`. The
adapter adds no dependencies and requires no GPU on the client machine.

First inspect the prepared inputs without contacting a model:

```sh
python -m app.chorus_review \
  --data /secure/path/data.csv \
  --dictionary /secure/path/dictionary.csv \
  --output-dir outputs/chorus/prepared \
  --prepare-only
```

Use a new empty directory for inference. Set the endpoint for the approved Muse
service explicitly. It must serve the configured model at `/v1/chat/completions`.
The existing URL validator requires HTTPS except for loopback, unless a private
network HTTP override is explicitly configured.

```sh
python -m app.chorus_review \
  --data /secure/path/data.csv \
  --dictionary /secure/path/dictionary.csv \
  --output-dir outputs/chorus/reviewed \
  --endpoint https://muse.example.org/v1/chat/completions \
  --model Muse-Glimmer-30B
```

`ROSEGOLD_MUSE_URL` can supply the endpoint instead. Existing
`ROSEGOLD_MUSE_API_KEY` and `ROSEGOLD_MUSE_TIMEOUT` settings are supported. The
adapter defaults to a 240-second timeout, a request budget of 8192 output tokens,
temperature zero, and sequential inference. Model response names must match the
requested model. A quality hold yields exit code 2 while preserving the results;
exit code 0 indicates that all cases passed the defined source checks.

## Outputs and review

The output directory contains:

| File | Purpose |
| --- | --- |
| `review.csv` | Decisions, short summaries, extracted evidence, flags and three blank reviewer fields |
| `evidence.csv` | Complete model quotes, original raw observations and source CSV references |
| `model_inputs.json` | Prepared observations, question definitions and patient linkage |
| `model_results.jsonl` | Every proposal, cumulative correction and final quality status |
| `run_manifest.json` | Policy, source and code hashes, model, request counts and quality holds |
| `POLICY.txt` | Exact provisional policy supplied in this run |

Open `review.csv` in Excel and filter by target, suggestion or Review ID. Read the
full quote and its raw observation in `evidence.csv`, verify the source chart,
then complete the reviewer check, final label and notes. Source references name
the input CSV, row and field. They are not fabricated chart-note IDs or dates.
CSV values that could become spreadsheet formulas are escaped for safe viewing.

The CLI exports CSV and JSON using existing dependencies. Historical Excel
formatting and comparison workbooks remain separate local artifacts. This command
consolidates the reviewed policy into one pass, rather than replaying historical
staged prompts. New model outputs can differ from a previous workbook.

## Evidence policy and limits

The versioned policy expands compound supporting choices from the dictionary,
allows reported diagnoses as an alternative to complete physiological checklists,
and includes source renal/shock context uniformly in sepsis inputs. Numeric SOFA
or acute renal findings with unresolved infection attribution support at most a
qualified probable pathway unless a separate diagnostic pathway is present.
The arrest audit considers substantive resuscitation reports without requiring
the deliberately withheld primary label. Ambiguous screens, nonspecific ADHF
findings, unknown arrest details and missing data retain their evidence limits.

Final phenotype labels, confirmation answers, uncertainty follow-ups and all
outcome answers are excluded from model inputs. Inference does not load reference
labels or calculate accuracy. Conditional questionnaire completion still creates
selection bias, and targets within a patient are correlated. Independent charts,
measurements, dates and authoritative adjudication are needed for validation.
Categorical outcomes are not run. Private model reasoning is not exported.

The policy is a provisional research contract informed by the
[Sepsis-3 consensus](https://jamanetwork.com/journals/jama/fullarticle/2492881),
the [ESICM shock consensus](https://pmc.ncbi.nlm.nih.gov/articles/PMC4239778/), and
the [2025 ESICM update](https://www.esicm.org/esicm-guideline-circulatory-shock-haemodynamic-monitoring/).
It requires study-specific review before clinical or prospective use.

Patient exports, local request packets, credentials and generated reports remain
outside Git. Keep runs under the ignored `outputs/` directory or another approved
local data location. The repository tests use synthetic records and simulated
service responses.
