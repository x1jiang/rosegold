from pydantic import BaseModel, Field, field_validator, ValidationInfo
from typing import Any, List, Optional, Literal
import re


class ClinicalEvidence(BaseModel):
    note_id: Optional[int] = Field(None, description="The OMOP note_id where evidence was found")
    note_date: Optional[str] = Field(None, description="Date of the note containing evidence")
    evidence_quote: str = Field(
        ...,
        description="Exact verbatim textual excerpt from the clinical note. Must not be paraphrased."
    )
    interpretation: str = Field(
        default="Supporting clinical documentation from chart review.",
        description="Brief clinical significance of this excerpt to the adjudication criteria"
    )

    @field_validator("note_id", mode="before")
    @classmethod
    def coerce_note_id(cls, v: Any) -> Optional[int]:
        if v is None or isinstance(v, bool):
            return None
        if isinstance(v, int):
            return v
        if isinstance(v, float):
            return int(v) if v == v else None
        if isinstance(v, str):
            m = re.search(r"\d+", v)
            return int(m.group(0)) if m else None
        return None

    @field_validator("evidence_quote", mode="before")
    @classmethod
    def validate_quote_length(cls, v: Any) -> str:
        s = str(v or "").strip()
        if len(s) < 5:
            raise ValueError("Evidence quote is too brief. Provide a meaningful clinical excerpt.")
        return s


_STATUS_ALIASES = {
    "CONFIRMED_POSITIVE": "CONFIRMED_POSITIVE",
    "POSITIVE": "CONFIRMED_POSITIVE",
    "TRUE": "CONFIRMED_POSITIVE",
    "PRESENT": "CONFIRMED_POSITIVE",
    "DEFINITE": "CONFIRMED_POSITIVE",
    "SUSPECTED_PROBABLE": "SUSPECTED_PROBABLE",
    "SUSPECTED": "SUSPECTED_PROBABLE",
    "PROBABLE": "SUSPECTED_PROBABLE",
    "POSSIBLE": "SUSPECTED_PROBABLE",
    "LIKELY": "SUSPECTED_PROBABLE",
    "CONFIRMED_NEGATIVE": "CONFIRMED_NEGATIVE",
    "NEGATIVE": "CONFIRMED_NEGATIVE",
    "FALSE": "CONFIRMED_NEGATIVE",
    "ABSENT": "CONFIRMED_NEGATIVE",
    "RULED_OUT": "CONFIRMED_NEGATIVE",
    "INDETERMINATE_INSUFFICIENT_DATA": "INDETERMINATE_INSUFFICIENT_DATA",
    "INDETERMINATE": "INDETERMINATE_INSUFFICIENT_DATA",
    "INSUFFICIENT_DATA": "INDETERMINATE_INSUFFICIENT_DATA",
    "UNKNOWN": "INDETERMINATE_INSUFFICIENT_DATA",
}


class RoseGoldAdjudication(BaseModel):
    """
    Structured clinical adjudication with a concise, evidence-linked decision summary.
    """
    clinical_rationale: str = Field(
        ...,
        description="Concise decision summary linking supplied findings to the classification and noting material uncertainty. Do not include private reasoning."
    )
    primary_criteria_met: List[str] = Field(
        default_factory=list,
        description="List of specific clinical consensus criteria satisfied by the chart documentation"
    )
    key_evidence: List[ClinicalEvidence] = Field(
        default_factory=list,
        description="Verbatim supporting text quotes and note dates from the patient's record"
    )
    phenotype_status: Literal[
        "CONFIRMED_POSITIVE",
        "SUSPECTED_PROBABLE",
        "CONFIRMED_NEGATIVE",
        "INDETERMINATE_INSUFFICIENT_DATA"
    ] = Field(
        ...,
        description="Final adjudicated clinical phenotype status"
    )
    condition_present: bool = Field(
        default=False,
        description="True if CONFIRMED_POSITIVE or SUSPECTED_PROBABLE; False otherwise"
    )
    confidence_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Self-reported model confidence between 0.0 and 1.0, not an empirically calibrated probability"
    )
    visit_occurrence_id: Optional[int] = Field(None, description="OMOP visit_occurrence_id")
    person_id: Optional[int] = Field(None, description="OMOP person_id")
    adjudication_timestamp: Optional[str] = Field(None, description="ISO timestamp")
    inference_backend: Optional[str] = Field(
        None,
        description="Actual inference path: mantle, bedrock, vllm, vertex, databricks, openai, hf_cpu, or keyword_rules",
    )

    @field_validator("phenotype_status", mode="before")
    @classmethod
    def normalize_phenotype_status(cls, v: Any) -> str:
        if not v:
            return "INDETERMINATE_INSUFFICIENT_DATA"
        cleaned = re.sub(r"[\s\-/]+", "_", str(v).strip().upper())
        return _STATUS_ALIASES.get(cleaned, str(v).strip().upper())

    @field_validator("confidence_score", mode="before")
    @classmethod
    def normalize_confidence_score(cls, v: Any) -> float:
        if v is None:
            return 0.5
        if isinstance(v, str):
            v = v.strip().rstrip("%")
        try:
            val = float(v)
        except (ValueError, TypeError):
            return 0.5
        if val > 1.0 and val <= 100.0:
            val = val / 100.0
        return max(0.0, min(1.0, val))

    @field_validator("key_evidence", mode="before")
    @classmethod
    def filter_valid_evidence(cls, v: Any) -> List[Any]:
        if not isinstance(v, list):
            return []
        cleaned = []
        for item in v:
            if isinstance(item, ClinicalEvidence):
                cleaned.append(item)
            elif isinstance(item, dict):
                quote = str(item.get("evidence_quote") or item.get("quote") or "").strip()
                if len(quote) >= 5:
                    item_copy = dict(item)
                    item_copy["evidence_quote"] = quote
                    if not item_copy.get("interpretation"):
                        item_copy["interpretation"] = "Supporting clinical excerpt from encounter notes."
                    cleaned.append(item_copy)
            elif isinstance(item, str) and len(item.strip()) >= 5:
                cleaned.append({
                    "evidence_quote": item.strip(),
                    "interpretation": "Supporting clinical excerpt from encounter notes.",
                })
        return cleaned

    @field_validator("condition_present")
    @classmethod
    def sync_condition_present(cls, v: bool, info: ValidationInfo) -> bool:
        status = info.data.get("phenotype_status")
        if status in ["CONFIRMED_POSITIVE", "SUSPECTED_PROBABLE"] and not v:
            return True
        if status in ["CONFIRMED_NEGATIVE", "INDETERMINATE_INSUFFICIENT_DATA"] and v:
            return False
        return v

