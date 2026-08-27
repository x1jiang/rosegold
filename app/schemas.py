from pydantic import BaseModel, Field, field_validator, ValidationInfo
from typing import List, Optional, Literal

class ClinicalEvidence(BaseModel):
    note_id: Optional[int] = Field(None, description="The OMOP note_id where evidence was found")
    note_date: Optional[str] = Field(None, description="Date of the note containing evidence")
    evidence_quote: str = Field(
        ..., 
        description="Exact verbatim textual excerpt from the clinical note. Must not be paraphrased."
    )
    interpretation: str = Field(
        ..., 
        description="Brief clinical significance of this excerpt to the adjudication criteria"
    )

    @field_validator('evidence_quote')
    @classmethod
    def validate_quote_length(cls, v: str) -> str:
        if len(v.strip()) < 5:
            raise ValueError("Evidence quote is too brief. Provide a meaningful clinical excerpt.")
        return v.strip()

class RoseGoldAdjudication(BaseModel):
    """
    Structured Clinical Adjudication Schema with Instructor.
    Note: clinical_rationale is placed first to enforce Chain-of-Thought reasoning before classification.
    """
    clinical_rationale: str = Field(
        ..., 
        description="Step-by-step clinical chain-of-thought reasoning assessing infection, organ dysfunction, or pathology."
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
        ..., 
        description="True if CONFIRMED_POSITIVE or SUSPECTED_PROBABLE; False otherwise"
    )
    confidence_score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Calibrated confidence score between 0.0 and 1.0"
    )
    visit_occurrence_id: Optional[int] = Field(None, description="OMOP visit_occurrence_id")
    person_id: Optional[int] = Field(None, description="OMOP person_id")
    adjudication_timestamp: Optional[str] = Field(None, description="ISO timestamp")
    inference_backend: Optional[str] = Field(
        None,
        description="Actual inference path: vllm, vertex, hf_cpu, or keyword_rules",
    )

    @field_validator('condition_present')
    @classmethod
    def sync_condition_present(cls, v: bool, info: ValidationInfo) -> bool:
        status = info.data.get('phenotype_status')
        if status in ["CONFIRMED_POSITIVE", "SUSPECTED_PROBABLE"] and not v:
            return True
        if status in ["CONFIRMED_NEGATIVE", "INDETERMINATE_INSUFFICIENT_DATA"] and v:
            return False
        return v
