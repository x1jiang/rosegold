"""Streamlit demo and interactive UI helpers for CHORUS REDCap survey abstraction and concordance."""

from __future__ import annotations

import csv
import io
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from app.chorus_abstractor import (
    _mock_stage_caller,
    _read_csv_flexible,
    _write_safe_csv,
    abstract_sepsis_cohort,
    audit_prompt_dictionary,
    load_chorus_instrument,
)
from app.chorus_concordance import evaluate_chorus_concordance, format_concordance_markdown


def build_demo_chorus_sepsis_bundle(work_dir: Path) -> Dict[str, Path]:
    """Create a 100% synthetic 6-encounter multi-modal ICU Sepsis bundle and REDCap dictionary."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    inst_cols = [
        "Variable / Field Name",
        "Form Name",
        "Section Header",
        "Field Type",
        "Field Label",
        "Choices, Calculations, OR Slider Labels",
        "Branching Logic (Show field only if...)",
    ]
    inst_rows = [
        {"Variable / Field Name": "record_id", "Form Name": "chorus", "Section Header": "", "Field Type": "text", "Field Label": "Person ID", "Choices, Calculations, OR Slider Labels": "", "Branching Logic (Show field only if...)": ""},
        {"Variable / Field Name": "institution_v3", "Form Name": "chorus", "Section Header": "", "Field Type": "dropdown", "Field Label": "Participating Site", "Choices, Calculations, OR Slider Labels": "99, SYNTHETIC-SITE", "Branching Logic (Show field only if...)": ""},
        {"Variable / Field Name": "reviewer_id", "Form Name": "chorus", "Section Header": "", "Field Type": "text", "Field Label": "Reviewer ID", "Choices, Calculations, OR Slider Labels": "", "Branching Logic (Show field only if...)": ""},
        {"Variable / Field Name": "vid", "Form Name": "chorus", "Section Header": "", "Field Type": "text", "Field Label": "Visit Occurrence ID", "Choices, Calculations, OR Slider Labels": "", "Branching Logic (Show field only if...)": ""},
        {"Variable / Field Name": "t0", "Form Name": "chorus", "Section Header": "", "Field Type": "text", "Field Label": "ICU Admission Time", "Choices, Calculations, OR Slider Labels": "", "Branching Logic (Show field only if...)": ""},
        {"Variable / Field Name": "concept_1_v3", "Form Name": "chorus", "Section Header": "", "Field Type": "checkbox", "Field Label": "Discharge concepts", "Choices, Calculations, OR Slider Labels": "1, Sepsis | 10, None of the above", "Branching Logic (Show field only if...)": ""},
        {"Variable / Field Name": "sepsis_ync_v3", "Form Name": "chorus", "Section Header": "", "Field Type": "radio", "Field Label": "Supports SEPSIS?", "Choices, Calculations, OR Slider Labels": "1, YES | 2, NO | 3, I am not sure", "Branching Logic (Show field only if...)": ""},
        {"Variable / Field Name": "sepsis_uncertain_v3", "Form Name": "chorus", "Section Header": "", "Field Type": "radio", "Field Label": "Evidence in H&P, Labs, Microbiology, Meds?", "Choices, Calculations, OR Slider Labels": "1, YES | 2, NO", "Branching Logic (Show field only if...)": "[sepsis_ync_v3] = 3"},
        {"Variable / Field Name": "sep1_v3", "Form Name": "chorus", "Section Header": "", "Field Type": "checkbox", "Field Label": "Suspicion of systemic infection (1)", "Choices, Calculations, OR Slider Labels": "1, Temperature: Greater than 38°C (100.4°F) or less than 36°C (96.8°F). | 2, Heart Rate: Greater than 90 beats per minute. | 3, Respiratory Rate: Greater than 20 breaths per minute or PaCO2 less than 32 mm Hg. | 4, White Blood Cell Count: Greater than 12,000/µL, less than 4,000/µL, or greater than 10% bands (immature white blood cells)", "Branching Logic (Show field only if...)": "[sepsis_ync_v3] = 1 or [sepsis_uncertain_v3] = 1"},
        {"Variable / Field Name": "sep3_v3", "Form Name": "chorus", "Section Header": "", "Field Type": "text", "Field Label": "Culture evidence (Source, Pathogen, Result)", "Choices, Calculations, OR Slider Labels": "", "Branching Logic (Show field only if...)": "[sepsis_ync_v3] = 1 or [sepsis_uncertain_v3] = 1"},
        {"Variable / Field Name": "sep4_v3", "Form Name": "chorus", "Section Header": "", "Field Type": "yesno", "Field Label": "Continued systemic antibiotics >24 hours", "Choices, Calculations, OR Slider Labels": "", "Branching Logic (Show field only if...)": "[sepsis_ync_v3] = 1 or [sepsis_uncertain_v3] = 1"},
        {"Variable / Field Name": "sep5_v3", "Form Name": "chorus", "Section Header": "", "Field Type": "text", "Field Label": "Other unstructured EHR evidence", "Choices, Calculations, OR Slider Labels": "", "Branching Logic (Show field only if...)": "[sepsis_ync_v3] = 1 or [sepsis_uncertain_v3] = 1"},
        {"Variable / Field Name": "sep6_v3", "Form Name": "chorus", "Section Header": "", "Field Type": "text", "Field Label": "Highest lactate (mmol/L)", "Choices, Calculations, OR Slider Labels": "", "Branching Logic (Show field only if...)": "[sepsis_ync_v3] = 1 or [sepsis_uncertain_v3] = 1"},
        {"Variable / Field Name": "sep_surg_v3", "Form Name": "chorus", "Section Header": "", "Field Type": "checkbox", "Field Label": "Surgical / vasoplegia / burn / trauma confounders", "Choices, Calculations, OR Slider Labels": "1, Yes - major surgery and the patient is <48 hours from the surgery | 2, Yes - there is a comment on vasoplegia | 3, Yes - active burn care / trauma care with <72 hours from the admission | 4, No | 9, Unclear to me given notes", "Branching Logic (Show field only if...)": "[sepsis_ync_v3] = 1 or [sepsis_uncertain_v3] = 1"},
        {"Variable / Field Name": "sep7_v3", "Form Name": "chorus", "Section Header": "", "Field Type": "text", "Field Label": "Daily SOFA scores", "Choices, Calculations, OR Slider Labels": "", "Branching Logic (Show field only if...)": "[sepsis_ync_v3] = 1 or [sepsis_uncertain_v3] = 1"},
        {"Variable / Field Name": "sep_day_v3", "Form Name": "chorus", "Section Header": "", "Field Type": "text", "Field Label": "ICU day SOFA rose >2 on antibiotics", "Choices, Calculations, OR Slider Labels": "", "Branching Logic (Show field only if...)": "[sepsis_ync_v3] = 1 or [sepsis_uncertain_v3] = 1"},
        {"Variable / Field Name": "sep_dt_v3", "Form Name": "chorus", "Section Header": "", "Field Type": "text", "Field Label": "Earliest sepsis identification time (HH:MM)", "Choices, Calculations, OR Slider Labels": "", "Branching Logic (Show field only if...)": "[sepsis_ync_v3] = 1 or [sepsis_uncertain_v3] = 1"},
        {"Variable / Field Name": "sep_days", "Form Name": "chorus", "Section Header": "", "Field Type": "text", "Field Label": "Calendar days since ICU admission", "Choices, Calculations, OR Slider Labels": "", "Branching Logic (Show field only if...)": "[sepsis_ync_v3] = 1 or [sepsis_uncertain_v3] = 1"},
        {"Variable / Field Name": "sep_time", "Form Name": "chorus", "Section Header": "", "Field Type": "calc", "Field Label": "Interval between t0 and sep_dt_v3", "Choices, Calculations, OR Slider Labels": 'datediff([t0],[sep_dt_v3],"m","mdy",true)', "Branching Logic (Show field only if...)": ""},
        {"Variable / Field Name": "conf1", "Form Name": "chorus", "Section Header": "", "Field Type": "radio", "Field Label": "Reviewer confidence 1-5", "Choices, Calculations, OR Slider Labels": "1, 1 - not confident | 2, 2 | 3, 3 | 4, 4 | 5, 5 - very confident", "Branching Logic (Show field only if...)": "[sepsis_ync_v3] = 1 or [sepsis_uncertain_v3] = 1"},
    ]

    sample_draft_rows = [
        {"QID": "Q1", "Code": "sepsis_ync_v3", "Type": "Concept gate", "UI": "radio", "Question": "Do you think the current chart review supports that this patient had SEPSIS?", "Prompt": 'Do you think the "[DATA]" supports that this patient had SEPSIS, respond with a YES or NO'},
        {"QID": "Q2", "Code": "sepsis_uncertain_v3", "Type": "Uncertain follow-up gate", "UI": "radio", "Question": "H&P, Labs, Micro, Meds evidence of Sepsis?", "Prompt": 'In the "[DATA]" is there evidence of Sepsis, respond with a YES or NO'},
        {"QID": "Q3.1", "Code": "sep1_v3___1", "Type": "Criterion / evidence", "UI": "checkbox", "Question": "Temperature: >38?C (100.4?F) or <36?C (96.8?F).", "Prompt": 'In the "[DATA]" was the temperature greater than 100.4F or less than 96.8F, respond with a YES or NO'},
        {"QID": "Q3.4", "Code": "sep1_v3___4", "Type": "Criterion / evidence", "UI": "checkbox", "Question": "White Blood Cell Count: >12,000/‘L, <4,000/‘L, or >10% bands", "Prompt": 'In the "[DATA]" was WBC > 12,000/‘L, respond with a YES or NO'},
        {"QID": "Q5", "Code": "sep4_v3", "Type": "Criterion / evidence", "UI": "yesno", "Question": "Antibiotics >24 hours", "Prompt": 'In the "[DATA]" confirm antibiotics >24 hours, respond with a CONFIRMED or CANNOT CONFIRM'},
        {"QID": "Q8.3", "Code": "sep_surg_v3___3", "Type": "Criterion / evidence", "UI": "checkbox", "Question": "major surgery related heavy sedation", "Prompt": ""},
        {"QID": "Q8.4", "Code": "sep_surg_v3___4", "Type": "Criterion / evidence", "UI": "checkbox", "Question": "major surgery related burn care", "Prompt": ""},
        {"QID": "Q8.9", "Code": "sep_surg_v3___9", "Type": "Criterion / evidence", "UI": "checkbox", "Question": "major surgery related trauma care", "Prompt": ""},
        {"QID": "Q9", "Code": "sep7_v3", "Type": "Criterion / evidence", "UI": "text", "Question": "Daily SOFA scores", "Prompt": ""},
        {"QID": "Q10", "Code": "sep_day_v3", "Type": "Timing ? days", "UI": "text", "Question": "SOFA rose >2 points day", "Prompt": ""},
        {"QID": "Q12", "Code": "sep_days", "Type": "Timing ? days", "UI": "text", "Question": "Calendar days since ICU admission", "Prompt": ""},
        {"QID": "Q14", "Code": "conf1", "Type": "Confidence 1?5", "UI": "radio", "Question": "Reviewer confidence 1-5", "Prompt": ""},
    ]

    visits_rows = [
        {"visit_occurrence_id": 9001, "person_id": 1001, "record_id": "SYN-ICU-01", "institution_v3": "99", "icu_admit_datetime": "2026-03-01 23:00:00", "visit_start_date": "2026-03-01", "visit_end_date": "2026-03-06"},
        {"visit_occurrence_id": 9002, "person_id": 1002, "record_id": "SYN-ICU-02", "institution_v3": "99", "icu_admit_datetime": "2026-03-05 10:00:00", "visit_start_date": "2026-03-05", "visit_end_date": "2026-03-11"},
        {"visit_occurrence_id": 9003, "person_id": 1003, "record_id": "SYN-ICU-03", "institution_v3": "99", "icu_admit_datetime": "2026-03-10 08:00:00", "visit_start_date": "2026-03-10", "visit_end_date": "2026-03-17"},
        {"visit_occurrence_id": 9004, "person_id": 1004, "record_id": "SYN-ICU-04", "institution_v3": "99", "icu_admit_datetime": "2026-03-14 14:00:00", "visit_start_date": "2026-03-14", "visit_end_date": "2026-03-16"},
        {"visit_occurrence_id": 9005, "person_id": 1005, "record_id": "SYN-ICU-05", "institution_v3": "99", "icu_admit_datetime": "2026-03-18 11:00:00", "visit_start_date": "2026-03-18", "visit_end_date": "2026-03-22"},
        {"visit_occurrence_id": 9006, "person_id": 1006, "record_id": "SYN-ICU-06", "institution_v3": "99", "icu_admit_datetime": "2026-03-22 09:00:00", "visit_start_date": "2026-03-22", "visit_end_date": "2026-03-27"},
    ]

    notes_rows = [
        {"note_id": 101, "visit_occurrence_id": 9001, "note_datetime": "2026-03-01 23:15:00", "note_title": "ED Note", "note_text": "74yo F nursing home resident with dysuria and rigors. Initial BP 102/64."},
        {"note_id": 102, "visit_occurrence_id": 9001, "note_datetime": "2026-03-02 02:30:00", "note_title": "MICU Progress Note", "note_text": "Developed septic shock with MAP 54 requiring norepinephrine. Blood culture growing ESBL Klebsiella pneumoniae."},
        {"note_id": 103, "visit_occurrence_id": 9001, "note_datetime": "2026-03-06 11:00:00", "note_title": "Discharge Summary", "note_text": "Principal Diagnosis: Severe sepsis and septic shock due to Klebsiella urosepsis, resolved on meropenem."},
        {"note_id": 201, "visit_occurrence_id": 9002, "note_datetime": "2026-03-05 10:30:00", "note_title": "Admission H&P", "note_text": "63yo M with multifocal pneumonia and hypoxemia. Cultures sent prior to IV vancomycin and cefepime."},
        {"note_id": 202, "visit_occurrence_id": 9002, "note_datetime": "2026-03-11 09:00:00", "note_title": "Discharge Summary", "note_text": "Culture-negative severe sepsis secondary to multifocal pneumonia, successfully treated with prolonged IV antibiotics."},
        {"note_id": 301, "visit_occurrence_id": 9003, "note_datetime": "2026-03-10 08:30:00", "note_title": "Operative Report", "note_text": "Sigmoid colectomy completed on 2026-03-09. POD 1 anastomotic leak with intra-abdominal sepsis."},
        {"note_id": 302, "visit_occurrence_id": 9003, "note_datetime": "2026-03-10 14:00:00", "note_title": "Surgery ICU Progress Note", "note_text": "Post-op septic shock with vasoplegia requiring norepinephrine after exploratory laparotomy."},
        {"note_id": 303, "visit_occurrence_id": 9003, "note_datetime": "2026-03-17 15:00:00", "note_title": "Discharge Summary", "note_text": "Post-operative intra-abdominal sepsis and septic shock resolved following source control."},
        {"note_id": 401, "visit_occurrence_id": 9004, "note_datetime": "2026-03-14 14:30:00", "note_title": "Admission H&P", "note_text": "33yo F with flank tenderness and uncomplicated pyelonephritis. Normal lactate 1.1, normotensive."},
        {"note_id": 402, "visit_occurrence_id": 9004, "note_datetime": "2026-03-16 12:00:00", "note_title": "Discharge Summary", "note_text": "Uncomplicated pyelonephritis treated with ceftriaxone. No sepsis or acute organ dysfunction."},
        {"note_id": 501, "visit_occurrence_id": 9005, "note_datetime": "2026-03-18 11:30:00", "note_title": "Cath Lab Procedure Note", "note_text": "Anterior STEMI with 100% LAD occlusion, primary PCI performed. Cardiogenic shock on dobutamine."},
        {"note_id": 502, "visit_occurrence_id": 9005, "note_datetime": "2026-03-22 10:00:00", "note_title": "Discharge Summary", "note_text": "Acute STEMI complicated by cardiogenic shock. Afebrile throughout stay, no infection or sepsis."},
        {"note_id": 601, "visit_occurrence_id": 9006, "note_datetime": "2026-03-22 09:30:00", "note_title": "Initial H&P", "note_text": "59yo M with severe bacteremic sepsis from lower extremity cellulitis, fever 39.2C, HR 122, lactate 3.4."},
        {"note_id": 602, "visit_occurrence_id": 9006, "note_datetime": "2026-03-23 11:00:00", "note_title": "ID Consult Note", "note_text": "MSSA bacteremia and sepsis responding to IV cefazolin."},
    ]

    measurements_rows = [
        {"measurement_id": "M1", "visit_occurrence_id": 9001, "measurement_datetime": "2026-03-01 23:20:00", "measurement_name": "platelets", "value_as_number": 190, "value_as_string": "", "unit": "10^3/uL"},
        {"measurement_id": "M2", "visit_occurrence_id": 9001, "measurement_datetime": "2026-03-01 23:20:00", "measurement_name": "creatinine", "value_as_number": 0.9, "value_as_string": "", "unit": "mg/dL"},
        {"measurement_id": "M3", "visit_occurrence_id": 9001, "measurement_datetime": "2026-03-02 02:00:00", "measurement_name": "temperature", "value_as_number": 102.0, "value_as_string": "", "unit": "F"},
        {"measurement_id": "M4", "visit_occurrence_id": 9001, "measurement_datetime": "2026-03-02 02:00:00", "measurement_name": "heart_rate", "value_as_number": 128, "value_as_string": "", "unit": "bpm"},
        {"measurement_id": "M5", "visit_occurrence_id": 9001, "measurement_datetime": "2026-03-02 02:00:00", "measurement_name": "respiratory_rate", "value_as_number": 26, "value_as_string": "", "unit": "bpm"},
        {"measurement_id": "M6", "visit_occurrence_id": 9001, "measurement_datetime": "2026-03-02 02:15:00", "measurement_name": "wbc", "value_as_number": 18.4, "value_as_string": "", "unit": "10^3/uL"},
        {"measurement_id": "M7", "visit_occurrence_id": 9001, "measurement_datetime": "2026-03-02 02:30:00", "measurement_name": "lactate", "value_as_number": 4.2, "value_as_string": "", "unit": "mmol/L"},
        {"measurement_id": "M8", "visit_occurrence_id": 9001, "measurement_datetime": "2026-03-02 08:00:00", "measurement_name": "platelets", "value_as_number": 90, "value_as_string": "", "unit": "10^3/uL"},
        {"measurement_id": "M9", "visit_occurrence_id": 9001, "measurement_datetime": "2026-03-02 08:00:00", "measurement_name": "creatinine", "value_as_number": 2.2, "value_as_string": "", "unit": "mg/dL"},
        {"measurement_id": "M10", "visit_occurrence_id": 9002, "measurement_datetime": "2026-03-05 10:30:00", "measurement_name": "temperature", "value_as_number": 38.7, "value_as_string": "", "unit": "C"},
        {"measurement_id": "M11", "visit_occurrence_id": 9002, "measurement_datetime": "2026-03-05 10:30:00", "measurement_name": "heart_rate", "value_as_number": 108, "value_as_string": "", "unit": "bpm"},
        {"measurement_id": "M12", "visit_occurrence_id": 9002, "measurement_datetime": "2026-03-05 10:30:00", "measurement_name": "respiratory_rate", "value_as_number": 28, "value_as_string": "", "unit": "bpm"},
        {"measurement_id": "M13", "visit_occurrence_id": 9002, "measurement_datetime": "2026-03-05 10:45:00", "measurement_name": "wbc", "value_as_number": 15.6, "value_as_string": "", "unit": "10^3/uL"},
        {"measurement_id": "M14", "visit_occurrence_id": 9002, "measurement_datetime": "2026-03-05 11:00:00", "measurement_name": "lactate", "value_as_number": 2.6, "value_as_string": "", "unit": "mmol/L"},
        {"measurement_id": "M15", "visit_occurrence_id": 9002, "measurement_datetime": "2026-03-05 11:00:00", "measurement_name": "pao2", "value_as_number": 140, "value_as_string": "", "unit": "mmHg"},
        {"measurement_id": "M16", "visit_occurrence_id": 9002, "measurement_datetime": "2026-03-05 11:00:00", "measurement_name": "fio2", "value_as_number": 50, "value_as_string": "", "unit": "%"},
        {"measurement_id": "M17", "visit_occurrence_id": 9002, "measurement_datetime": "2026-03-06 09:00:00", "measurement_name": "pao2", "value_as_number": 95, "value_as_string": "", "unit": "mmHg"},
        {"measurement_id": "M18", "visit_occurrence_id": 9002, "measurement_datetime": "2026-03-06 09:00:00", "measurement_name": "fio2", "value_as_number": 60, "value_as_string": "", "unit": "%"},
        {"measurement_id": "M19", "visit_occurrence_id": 9002, "measurement_datetime": "2026-03-06 09:00:00", "measurement_name": "mechanical_ventilation", "value_as_number": 1, "value_as_string": "intubated", "unit": ""},
        {"measurement_id": "M20", "visit_occurrence_id": 9002, "measurement_datetime": "2026-03-06 09:00:00", "measurement_name": "creatinine", "value_as_number": 2.3, "value_as_string": "", "unit": "mg/dL"},
        {"measurement_id": "M21", "visit_occurrence_id": 9003, "measurement_datetime": "2026-03-10 08:30:00", "measurement_name": "temperature", "value_as_number": 39.1, "value_as_string": "", "unit": "C"},
        {"measurement_id": "M22", "visit_occurrence_id": 9003, "measurement_datetime": "2026-03-10 08:30:00", "measurement_name": "heart_rate", "value_as_number": 134, "value_as_string": "", "unit": "bpm"},
        {"measurement_id": "M23", "visit_occurrence_id": 9003, "measurement_datetime": "2026-03-10 08:30:00", "measurement_name": "wbc", "value_as_number": 21.0, "value_as_string": "", "unit": "10^3/uL"},
        {"measurement_id": "M24", "visit_occurrence_id": 9003, "measurement_datetime": "2026-03-10 09:00:00", "measurement_name": "lactate", "value_as_number": 3.5, "value_as_string": "", "unit": "mmol/L"},
        {"measurement_id": "M25", "visit_occurrence_id": 9003, "measurement_datetime": "2026-03-10 12:00:00", "measurement_name": "creatinine", "value_as_number": 1.4, "value_as_string": "", "unit": "mg/dL"},
        {"measurement_id": "M26", "visit_occurrence_id": 9003, "measurement_datetime": "2026-03-11 08:00:00", "measurement_name": "creatinine", "value_as_number": 2.9, "value_as_string": "", "unit": "mg/dL"},
        {"measurement_id": "M27", "visit_occurrence_id": 9006, "measurement_datetime": "2026-03-22 09:30:00", "measurement_name": "temperature", "value_as_number": 39.2, "value_as_string": "", "unit": "C"},
        {"measurement_id": "M28", "visit_occurrence_id": 9006, "measurement_datetime": "2026-03-22 09:30:00", "measurement_name": "heart_rate", "value_as_number": 122, "value_as_string": "", "unit": "bpm"},
        {"measurement_id": "M29", "visit_occurrence_id": 9006, "measurement_datetime": "2026-03-22 09:30:00", "measurement_name": "respiratory_rate", "value_as_number": 24, "value_as_string": "", "unit": "bpm"},
        {"measurement_id": "M30", "visit_occurrence_id": 9006, "measurement_datetime": "2026-03-22 09:45:00", "measurement_name": "wbc", "value_as_number": 17.5, "value_as_string": "", "unit": "10^3/uL"},
        {"measurement_id": "M31", "visit_occurrence_id": 9006, "measurement_datetime": "2026-03-22 10:00:00", "measurement_name": "lactate", "value_as_number": 3.4, "value_as_string": "", "unit": "mmol/L"},
        {"measurement_id": "M32", "visit_occurrence_id": 9006, "measurement_datetime": "2026-03-22 10:00:00", "measurement_name": "creatinine", "value_as_number": 1.3, "value_as_string": "", "unit": "mg/dL"},
        {"measurement_id": "M33", "visit_occurrence_id": 9006, "measurement_datetime": "2026-03-23 08:00:00", "measurement_name": "creatinine", "value_as_number": 2.7, "value_as_string": "", "unit": "mg/dL"},
        {"measurement_id": "M34", "visit_occurrence_id": 9006, "measurement_datetime": "2026-03-23 08:00:00", "measurement_name": "platelets", "value_as_number": 85, "value_as_string": "", "unit": "10^3/uL"},
    ]

    medications_rows = [
        {"drug_exposure_id": "D1", "visit_occurrence_id": 9001, "drug_start_datetime": "2026-03-02 02:30:00", "drug_end_datetime": "2026-03-05 12:00:00", "drug_name": "Meropenem", "route": "IV", "dose_rate": ""},
        {"drug_exposure_id": "D2", "visit_occurrence_id": 9001, "drug_start_datetime": "2026-03-02 02:45:00", "drug_end_datetime": "2026-03-03 18:00:00", "drug_name": "Norepinephrine", "route": "IV", "dose_rate": "0.14"},
        {"drug_exposure_id": "D3", "visit_occurrence_id": 9002, "drug_start_datetime": "2026-03-05 11:00:00", "drug_end_datetime": "2026-03-05 15:00:00", "drug_name": "Vancomycin", "route": "IV", "dose_rate": ""},
        {"drug_exposure_id": "D4", "visit_occurrence_id": 9002, "drug_start_datetime": "2026-03-07 03:00:00", "drug_end_datetime": "2026-03-09 12:00:00", "drug_name": "Cefepime", "route": "IV", "dose_rate": ""},
        {"drug_exposure_id": "D5", "visit_occurrence_id": 9003, "drug_start_datetime": "2026-03-10 09:30:00", "drug_end_datetime": "2026-03-15 10:00:00", "drug_name": "Piperacillin-Tazobactam", "route": "IV", "dose_rate": ""},
        {"drug_exposure_id": "D6", "visit_occurrence_id": 9003, "drug_start_datetime": "2026-03-10 14:00:00", "drug_end_datetime": "2026-03-12 08:00:00", "drug_name": "Norepinephrine", "route": "IV", "dose_rate": "0.08"},
        {"drug_exposure_id": "D7", "visit_occurrence_id": 9005, "drug_start_datetime": "2026-03-18 12:00:00", "drug_end_datetime": "2026-03-20 08:00:00", "drug_name": "Dobutamine", "route": "IV", "dose_rate": "5.0"},
        {"drug_exposure_id": "D8", "visit_occurrence_id": 9006, "drug_start_datetime": "2026-03-22 10:30:00", "drug_end_datetime": "2026-03-25 10:00:00", "drug_name": "Cefazolin", "route": "IV", "dose_rate": ""},
    ]

    cultures_rows = [
        {"culture_id": "C1", "visit_occurrence_id": 9001, "collection_datetime": "2026-03-02 02:15:00", "source": "Blood", "organism": "ESBL Klebsiella pneumoniae", "result": "(+)"},
        {"culture_id": "C2", "visit_occurrence_id": 9002, "collection_datetime": "2026-03-05 10:45:00", "source": "Blood", "organism": "No growth", "result": "Negative"},
        {"culture_id": "C3", "visit_occurrence_id": 9003, "collection_datetime": "2026-03-10 10:00:00", "source": "Peritoneal fluid", "organism": "Escherichia coli", "result": "(+)"},
        {"culture_id": "C4", "visit_occurrence_id": 9004, "collection_datetime": "2026-03-14 15:00:00", "source": "Urine", "organism": "Escherichia coli", "result": "(+)"},
        {"culture_id": "C5", "visit_occurrence_id": 9006, "collection_datetime": "2026-03-22 10:00:00", "source": "Blood", "organism": "MSSA (Staphylococcus aureus)", "result": "(+)"},
    ]

    clinician_rows = [
        {"institution_v3": "99", "record_id": "SYN-ICU-01", "vid": "9001", "sepsis_ync_v3": "1", "sepsis_uncertain_v3": "", "sep1_v3___1": "1", "sep1_v3___2": "1", "sep1_v3___3": "1", "sep1_v3___4": "1", "sep3_v3": "Blood, ESBL Klebsiella pneumoniae, (+)", "sep4_v3": "1", "sep6_v3": "4.2", "sep_surg_v3___1": "0", "sep_surg_v3___2": "0", "sep_surg_v3___3": "0", "sep_surg_v3___4": "1", "sep_surg_v3___9": "0", "sep7_v3": "0, 8", "sep_day_v3": "1", "sep_dt_v3": "02:00", "sep_days": "1", "conf1": "5"},
        {"institution_v3": "99", "record_id": "SYN-ICU-02", "vid": "9002", "sepsis_ync_v3": "1", "sepsis_uncertain_v3": "", "sep1_v3___1": "1", "sep1_v3___2": "1", "sep1_v3___3": "1", "sep1_v3___4": "1", "sep3_v3": "NO", "sep4_v3": "1", "sep6_v3": "2.6", "sep_surg_v3___1": "0", "sep_surg_v3___2": "0", "sep_surg_v3___3": "0", "sep_surg_v3___4": "1", "sep_surg_v3___9": "0", "sep7_v3": "2, 5", "sep_day_v3": "1", "sep_dt_v3": "10:30", "sep_days": "0", "conf1": "4"},
        {"institution_v3": "99", "record_id": "SYN-ICU-03", "vid": "9003", "sepsis_ync_v3": "1", "sepsis_uncertain_v3": "", "sep1_v3___1": "1", "sep1_v3___2": "1", "sep1_v3___3": "", "sep1_v3___4": "1", "sep3_v3": "Peritoneal fluid, Escherichia coli, (+)", "sep4_v3": "1", "sep6_v3": "3.5", "sep_surg_v3___1": "1", "sep_surg_v3___2": "1", "sep_surg_v3___3": "0", "sep_surg_v3___4": "0", "sep_surg_v3___9": "0", "sep7_v3": "4, 2", "sep_day_v3": "", "sep_dt_v3": "08:30", "sep_days": "0", "conf1": "3"},
        {"institution_v3": "99", "record_id": "SYN-ICU-04", "vid": "9004", "sepsis_ync_v3": "2", "sepsis_uncertain_v3": "", "sep1_v3___1": "", "sep1_v3___2": "", "sep1_v3___3": "", "sep1_v3___4": "", "sep3_v3": "", "sep4_v3": "", "sep6_v3": "", "sep_surg_v3___1": "", "sep_surg_v3___2": "", "sep_surg_v3___3": "", "sep_surg_v3___4": "", "sep_surg_v3___9": "", "sep7_v3": "", "sep_day_v3": "", "sep_dt_v3": "", "sep_days": "", "conf1": "5"},
        {"institution_v3": "99", "record_id": "SYN-ICU-05", "vid": "9005", "sepsis_ync_v3": "2", "sepsis_uncertain_v3": "", "sep1_v3___1": "", "sep1_v3___2": "", "sep1_v3___3": "", "sep1_v3___4": "", "sep3_v3": "", "sep4_v3": "", "sep6_v3": "", "sep_surg_v3___1": "", "sep_surg_v3___2": "", "sep_surg_v3___3": "", "sep_surg_v3___4": "", "sep_surg_v3___9": "", "sep7_v3": "", "sep_day_v3": "", "sep_dt_v3": "", "sep_days": "", "conf1": "5"},
        {"institution_v3": "99", "record_id": "SYN-ICU-06", "vid": "9006", "sepsis_ync_v3": "3", "sepsis_uncertain_v3": "1", "sep1_v3___1": "1", "sep1_v3___2": "1", "sep1_v3___3": "1", "sep1_v3___4": "1", "sep3_v3": "Blood, MSSA (Staphylococcus aureus), (+)", "sep4_v3": "1", "sep6_v3": "3.4", "sep_surg_v3___1": "0", "sep_surg_v3___2": "0", "sep_surg_v3___3": "0", "sep_surg_v3___4": "1", "sep_surg_v3___9": "0", "sep7_v3": "1, 4", "sep_day_v3": "1", "sep_dt_v3": "09:30", "sep_days": "0", "conf1": "2"},
    ]

    paths: Dict[str, Path] = {
        "dictionary": work_dir / "instrument.csv",
        "sample_draft_prompts": work_dir / "sample_draft_prompts.csv",
        "visits": work_dir / "visits.csv",
        "notes": work_dir / "notes.csv",
        "measurements": work_dir / "measurements.csv",
        "medications": work_dir / "medications.csv",
        "cultures": work_dir / "cultures.csv",
        "clinician_redcap": work_dir / "simulated_clinician_redcap.csv",
    }
    _write_safe_csv(paths["dictionary"], inst_rows, inst_cols)
    _write_safe_csv(paths["sample_draft_prompts"], sample_draft_rows, list(sample_draft_rows[0]))
    _write_safe_csv(paths["visits"], visits_rows, list(visits_rows[0]))
    _write_safe_csv(paths["notes"], notes_rows, list(notes_rows[0]))
    _write_safe_csv(paths["measurements"], measurements_rows, list(measurements_rows[0]))
    _write_safe_csv(paths["medications"], medications_rows, list(medications_rows[0]))
    _write_safe_csv(paths["cultures"], cultures_rows, list(cultures_rows[0]))
    _write_safe_csv(paths["clinician_redcap"], clinician_rows, list(clinician_rows[0]))
    return paths


def run_demo_chorus_sepsis_pipeline() -> Dict[str, Any]:
    """Execute the staged REDCap pipeline on synthetic data without modifying os.environ."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        bundle = build_demo_chorus_sepsis_bundle(root / "inputs")
        out_dir = root / "outputs"
        abs_res = abstract_sepsis_cohort(
            notes_path=bundle["notes"],
            visits_path=bundle["visits"],
            dictionary_path=bundle["dictionary"],
            output_dir=out_dir,
            measurements_path=bundle["measurements"],
            medications_path=bundle["medications"],
            cultures_path=bundle["cultures"],
            institution_code="99",
            backend="keyword_rules",
            stage_caller=_mock_stage_caller,
        )
        specs = load_chorus_instrument(bundle["dictionary"])
        _, human_rows = _read_csv_flexible(bundle["clinician_redcap"])
        _, llm_rows = _read_csv_flexible(out_dir / "redcap_import.csv")
        _, ev_rows = _read_csv_flexible(out_dir / "field_evidence.csv")
        concordance = evaluate_chorus_concordance(human_rows, llm_rows, specs, llm_evidence_rows=ev_rows)
        concordance_md = format_concordance_markdown(concordance)
        audit_findings = audit_prompt_dictionary(bundle["sample_draft_prompts"], bundle["dictionary"])

        return {
            "manifest": abs_res["manifest"],
            "redcap_rows": llm_rows,
            "adjudication_rows": _read_csv_flexible(out_dir / "rosegold_adjudications.csv")[1],
            "evidence_rows": ev_rows,
            "concordance": concordance,
            "concordance_md": concordance_md,
            "audit_findings": audit_findings,
            "redcap_csv": (out_dir / "redcap_import.csv").read_text(encoding="utf-8-sig"),
            "evidence_csv": (out_dir / "field_evidence.csv").read_text(encoding="utf-8-sig"),
            "adjudications_csv": (out_dir / "rosegold_adjudications.csv").read_text(encoding="utf-8-sig"),
        }


def _rows_to_csv_bytes(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return ""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def _run_uploaded_bundle(
    notes_bytes: bytes,
    visits_bytes: bytes,
    dict_bytes: bytes,
    meas_bytes: Optional[bytes] = None,
    meds_bytes: Optional[bytes] = None,
    cult_bytes: Optional[bytes] = None,
    clinician_bytes: Optional[bytes] = None,
    draft_bytes: Optional[bytes] = None,
) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        in_dir = root / "inputs"
        in_dir.mkdir(parents=True, exist_ok=True)
        out_dir = root / "outputs"

        def _save(name: str, data: Optional[bytes]) -> Optional[Path]:
            if data is None:
                return None
            p = in_dir / name
            p.write_bytes(data)
            return p

        notes_p = _save("notes.csv", notes_bytes)
        visits_p = _save("visits.csv", visits_bytes)
        dict_p = _save("dictionary.csv", dict_bytes)
        meas_p = _save("measurements.csv", meas_bytes)
        meds_p = _save("medications.csv", meds_bytes)
        cult_p = _save("cultures.csv", cult_bytes)
        clin_p = _save("clinician_redcap.csv", clinician_bytes)
        draft_p = _save("draft_prompts.csv", draft_bytes)

        abs_res = abstract_sepsis_cohort(
            notes_path=notes_p,  # type: ignore[arg-type]
            visits_path=visits_p,  # type: ignore[arg-type]
            dictionary_path=dict_p,  # type: ignore[arg-type]
            output_dir=out_dir,
            measurements_path=meas_p,
            medications_path=meds_p,
            cultures_path=cult_p,
            backend="keyword_rules",
            stage_caller=_mock_stage_caller,
        )
        specs = load_chorus_instrument(dict_p)  # type: ignore[arg-type]
        _, llm_rows = _read_csv_flexible(out_dir / "redcap_import.csv")
        _, ev_rows = _read_csv_flexible(out_dir / "field_evidence.csv")

        concordance = None
        concordance_md = ""
        if clin_p is not None:
            _, human_rows = _read_csv_flexible(clin_p)
            concordance = evaluate_chorus_concordance(human_rows, llm_rows, specs, llm_evidence_rows=ev_rows)
            concordance_md = format_concordance_markdown(concordance)

        audit_findings = audit_prompt_dictionary(draft_p, dict_p) if draft_p is not None else []  # type: ignore[arg-type]
        return {
            "manifest": abs_res["manifest"],
            "redcap_rows": llm_rows,
            "adjudication_rows": _read_csv_flexible(out_dir / "rosegold_adjudications.csv")[1],
            "evidence_rows": ev_rows,
            "concordance": concordance,
            "concordance_md": concordance_md,
            "audit_findings": audit_findings,
            "redcap_csv": (out_dir / "redcap_import.csv").read_text(encoding="utf-8-sig"),
            "evidence_csv": (out_dir / "field_evidence.csv").read_text(encoding="utf-8-sig"),
            "adjudications_csv": (out_dir / "rosegold_adjudications.csv").read_text(encoding="utf-8-sig"),
        }


def render_chorus_redcap_tab() -> None:
    """Render the interactive REDCap Survey & Concordance tab inside Streamlit."""
    import streamlit as st

    @st.cache_data(show_spinner=False)
    def _cached_demo() -> Dict[str, Any]:
        return run_demo_chorus_sepsis_pipeline()

    st.markdown("### 📋 REDCap Survey Abstraction & Clinician Concordance (Sepsis Q1–Q18)")
    st.markdown(
        "Staged, clinician-parity REDCap abstraction: **Stage 1** screens the Discharge Summary (`1=YES`, `2=NO`, "
        "`3=I am not sure`), **Stage 2** reviews Initial H&P + admission labs/micro/meds when Stage 1 is `3`, and "
        "**Stage 3** runs deterministic Python calculators for SIRS, cultures, >24h antibiotic spans (including "
        "vancomycin pauses), peak lactate, daily 6-organ SOFA, and relative `Day +d HH:MM` onset timing."
    )

    mode = st.radio(
        "Data Source Mode",
        [
            "Built-in Synthetic 6-Visit ICU Sepsis Cohort (Instant Demo)",
            "Upload De-Identified OMOP & REDCap CSVs",
        ],
        horizontal=True,
        key="chorus_redcap_mode",
    )

    result: Optional[Dict[str, Any]] = None
    if mode.startswith("Built-in"):
        result = _cached_demo()
    else:
        st.info(
            "In-browser uploads use deterministic rules + structured calculators with zero external PHI egress. "
            "For full LLM inference behind your hospital firewall, run `python -m app.chorus_abstractor` locally."
        )
        u1, u2, u3 = st.columns(3)
        notes_file = u1.file_uploader("Required: notes.csv", type=["csv"], key="chorus_up_notes")
        visits_file = u2.file_uploader("Required: visits.csv", type=["csv"], key="chorus_up_visits")
        dict_file = u3.file_uploader("Required: REDCap dictionary.csv", type=["csv"], key="chorus_up_dict")

        u4, u5, u6, u7, u8 = st.columns(5)
        meas_file = u4.file_uploader("measurements.csv", type=["csv"], key="chorus_up_meas")
        meds_file = u5.file_uploader("medications.csv", type=["csv"], key="chorus_up_meds")
        cult_file = u6.file_uploader("cultures.csv", type=["csv"], key="chorus_up_cult")
        clin_file = u7.file_uploader("clinician_redcap.csv", type=["csv"], key="chorus_up_clin")
        draft_file = u8.file_uploader("draft_prompts.csv", type=["csv"], key="chorus_up_draft")

        if st.button("▶️ Run REDCap Abstraction & Concordance", type="primary", key="chorus_up_run"):
            if not (notes_file and visits_file and dict_file):
                st.error("Please upload `notes.csv`, `visits.csv`, and the canonical REDCap `dictionary.csv`.")
            else:
                try:
                    st.session_state["chorus_uploaded_result"] = _run_uploaded_bundle(
                        notes_file.getvalue(),
                        visits_file.getvalue(),
                        dict_file.getvalue(),
                        meas_file.getvalue() if meas_file else None,
                        meds_file.getvalue() if meds_file else None,
                        cult_file.getvalue() if cult_file else None,
                        clin_file.getvalue() if clin_file else None,
                        draft_file.getvalue() if draft_file else None,
                    )
                except Exception as exc:
                    st.error(f"Could not process uploaded CSVs ({type(exc).__name__}: {exc})")
        result = st.session_state.get("chorus_uploaded_result")

    if not result:
        return

    manifest = result["manifest"]
    conc = result.get("concordance")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Encounters Abstracted", f"{manifest['encounters_processed']}")
    if conc:
        g3 = conc["gate_3way"]
        gr = conc["gate_resolved"]
        nt = conc["numeric_and_timing_metrics"]
        m2.metric("Stage-1 3-Way Gate (κw)", f"{g3['overall_agreement_pct']}%", f"κw = {g3['linear_weighted_kappa']}")
        m3.metric("Post-Branch Resolved κ", f"{gr['cohens_kappa']:.3f}", f"Sens {gr['sensitivity']}% | Spec {gr['specificity']}%")
        lac_mae = nt["sep6_v3"]["mae"]
        sofa_acc = nt["sep7_v3"]["within_1_point_pct"]
        m4.metric(
            "Lactate / Peak SOFA",
            f"MAE {lac_mae if lac_mae is not None else 'N/A'} mmol/L",
            f"SOFA ±1 pt: {sofa_acc if sofa_acc is not None else 'N/A'}%",
        )
    else:
        m2.metric("Stage-1 3-Way Gate (κw)", "Upload clinician CSV")
        m3.metric("Post-Branch Resolved κ", "Upload clinician CSV")
        m4.metric("Grounded Evidence Rows", f"{len(result['evidence_rows'])}")

    st.divider()
    st.subheader("1. Coded REDCap Survey Output & Grounded Evidence Sidecar")
    st.dataframe(pd.DataFrame(result["redcap_rows"]), use_container_width=True, hide_index=True)

    dl1, dl2, dl3 = st.columns(3)
    dl1.download_button(
        "📥 Download REDCap Import CSV",
        data=result["redcap_csv"],
        file_name="redcap_import.csv",
        mime="text/csv",
        use_container_width=True,
    )
    dl2.download_button(
        "📥 Download Field Evidence CSV",
        data=result["evidence_csv"],
        file_name="field_evidence.csv",
        mime="text/csv",
        use_container_width=True,
    )
    dl3.download_button(
        "📥 Download Adjudications & Flags CSV",
        data=result["adjudications_csv"],
        file_name="rosegold_adjudications.csv",
        mime="text/csv",
        use_container_width=True,
    )

    with st.expander("View Field-Level Grounded Evidence & Review Flags", expanded=False):
        st.markdown("**Adjudication Statuses & Clinical Review Flags**")
        st.dataframe(pd.DataFrame(result["adjudication_rows"]), use_container_width=True, hide_index=True)
        st.markdown("**Grounded Verbatim Quotes & Structured Provenance (`field_evidence.csv`)**")
        st.dataframe(pd.DataFrame(result["evidence_rows"]), use_container_width=True, hide_index=True)

    if conc:
        st.divider()
        st.subheader("2. Multi-Level Clinician-vs-LLM Concordance Suite")
        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown("**Stage-1 Three-Way Gate Confusion Matrix (`sepsis_ync_v3`)**")
            mat = conc["gate_3way"]["matrix"]
            mat_df = pd.DataFrame(
                [
                    {"Human \\ LLM": "1 (YES)", "LLM: 1 (YES)": mat["1"]["1"], "LLM: 2 (NO)": mat["1"]["2"], "LLM: 3 (Uncertain)": mat["1"]["3"]},
                    {"Human \\ LLM": "2 (NO)", "LLM: 1 (YES)": mat["2"]["1"], "LLM: 2 (NO)": mat["2"]["2"], "LLM: 3 (Uncertain)": mat["2"]["3"]},
                    {"Human \\ LLM": "3 (Uncertain)", "LLM: 1 (YES)": mat["3"]["1"], "LLM: 2 (NO)": mat["3"]["2"], "LLM: 3 (Uncertain)": mat["3"]["3"]},
                ]
            )
            st.dataframe(mat_df, use_container_width=True, hide_index=True)

            by_conf = conc["gate_resolved"].get("by_clinician_confidence") or {}
            if by_conf:
                st.markdown("**Stratified by Clinician Confidence (`conf1`)**")
                conf_df = pd.DataFrame(
                    [
                        {
                            "Stratum": k,
                            "N": v["total_evaluated"],
                            "Agreement (%)": v["overall_agreement_pct"],
                            "Kappa": v["cohens_kappa"],
                        }
                        for k, v in by_conf.items()
                    ]
                )
                st.dataframe(conf_df, use_container_width=True, hide_index=True)

        with c_right:
            st.markdown("**Quantitative & Timing Accuracy**")
            nt = conc["numeric_and_timing_metrics"]
            nt_df = pd.DataFrame(
                [
                    {"Metric": "sep6_v3 (Highest Lactate)", "Paired N": nt["sep6_v3"]["paired_n"], "Summary": f"MAE = {nt['sep6_v3']['mae']} mmol/L | Within 0.2 = {nt['sep6_v3']['within_0_2_mmol_pct']}%"},
                    {"Metric": "sep7_v3 (Daily SOFA)", "Paired N": nt["sep7_v3"]["paired_n"], "Summary": f"Day-0 MAE = {nt['sep7_v3']['day0_sofa_mae']} | Peak MAE = {nt['sep7_v3']['peak_sofa_mae']} | Within 1 pt = {nt['sep7_v3']['within_1_point_pct']}%"},
                    {"Metric": "sep_day_v3 (SOFA Rise Day)", "Paired N": nt["sep_day_v3"]["paired_n"], "Summary": f"Exact = {nt['sep_day_v3']['exact_match_pct']}% | MAE = {nt['sep_day_v3']['mae_days']} d"},
                    {"Metric": "sep_days & sep_dt_v3 (Onset)", "Paired N": nt["sep_days"]["paired_n"], "Summary": f"Exact day = {nt['sep_days']['exact_day_match_pct']}% | Combined MAE = {nt['sep_days']['combined_onset_mae_hours']} h"},
                ]
            )
            st.dataframe(nt_df, use_container_width=True, hide_index=True)

            jacc = conc["item_metrics"].get("sep3_v3_positive", {}).get("sep3_organism_jaccard_mean")
            if jacc is not None:
                st.caption(f"Mean positive culture organism token Jaccard (`sep3_v3`): **{jacc}**")

        st.markdown("**Per-Item REDCap Agreement (Shared Open Sepsis Branches)**")
        item_df = pd.DataFrame(
            [
                {
                    "Field": field,
                    "Paired N": m["total_evaluated"],
                    "Agreement (%)": m["overall_agreement_pct"],
                    "Kappa": m["cohens_kappa"],
                    "Sensitivity (%)": m["sensitivity"],
                    "Specificity (%)": m["specificity"],
                    "LLM Missing": m["llm_missing_count"],
                }
                for field, m in conc["item_metrics"].items()
            ]
        )
        st.dataframe(item_df, use_container_width=True, hide_index=True)

        dl_c1, dl_c2 = st.columns(2)
        dl_c1.download_button(
            "📥 Download Concordance Report (.md)",
            data=result["concordance_md"],
            file_name="concordance_report.md",
            mime="text/markdown",
            use_container_width=True,
        )
        disc_csv = _rows_to_csv_bytes(conc["discrepancies"]) or "institution_v3,record_id,vid,field,human_value,llm_value,clinician_conf1,discrepancy_category,llm_evidence_quote,llm_source_ref\n"
        dl_c2.download_button(
            "📥 Download Discrepancies CSV",
            data=disc_csv,
            file_name="discrepancies.csv",
            mime="text/csv",
            use_container_width=True,
        )

    findings = result.get("audit_findings") or []
    with st.expander(f"🔍 REDCap Prompt Dictionary Auditor ({len(findings)} sample findings)", expanded=False):
        if findings:
            st.dataframe(pd.DataFrame(findings), use_container_width=True, hide_index=True)
        else:
            st.success("No prompt/dictionary schema mismatches detected.")
