import pandas as pd
import datetime
from typing import List, Dict, Any

# OMOP Standard Concept ID Mappings (SNOMED / OMOP CDM v5.4)
CONCEPT_MAP = {
    "Sepsis / Septic Shock": 132797, # SNOMED 91302008 Sepsis
    "Acute Ischemic Stroke": 443454, # SNOMED 422504002 Cerebral infarction
    "Acute Respiratory Distress Syndrome (ARDS)": 4195694, # SNOMED 67782005 ARDS
    "Acute Kidney Injury (AKI)": 197320, # SNOMED 14669001 AKI
}

STATUS_VALUE_MAP = {
    "CONFIRMED_POSITIVE": 4181412, # OMOP Concept: Present
    "SUSPECTED_PROBABLE": 4181413, # OMOP Concept: Suspected
    "CONFIRMED_NEGATIVE": 4188540, # OMOP Concept: Absent
    "INDETERMINATE_INSUFFICIENT_DATA": 45877994 # OMOP Concept: Indeterminate
}

def export_to_omop_observation(adjudications: List[Dict[str, Any]], target_condition: str) -> pd.DataFrame:
    """
    Transforms Rose Gold adjudications into standard OMOP CDM OBSERVATION table records.
    Allows partner sites to seamlessly ingest labels back into their native OMOP database.
    """
    obs_rows = []
    base_obs_id = 9000001
    now_dt = datetime.datetime.now(datetime.timezone.utc)
    concept_id = CONCEPT_MAP.get(target_condition, 132797)

    for i, adj in enumerate(adjudications):
        status = adj.get("phenotype_status", "CONFIRMED_NEGATIVE")
        value_concept = STATUS_VALUE_MAP.get(status, 4188540)
        
        obs_rows.append({
            "observation_id": base_obs_id + i,
            "person_id": adj.get("person_id", 0),
            "observation_concept_id": concept_id,
            "observation_date": now_dt.strftime("%Y-%m-%d"),
            "observation_datetime": now_dt.isoformat(),
            "observation_type_concept_id": 32817, # OMOP Concept: NLP / Algorithm Derived
            "value_as_number": adj.get("confidence_score", 0.0),
            "value_as_string": status,
            "value_as_concept_id": value_concept,
            "qualifier_concept_id": 0,
            "unit_concept_id": 0,
            "provider_id": None,
            "visit_occurrence_id": adj.get("visit_occurrence_id", 0),
            "visit_detail_id": None,
            "observation_source_value": f"Rose Gold LLM: {target_condition}",
            "observation_source_concept_id": 0,
            "unit_source_value": "confidence_probability",
            "qualifier_source_value": adj.get("clinical_rationale", "")[:250]
        })

    return pd.DataFrame(obs_rows)
