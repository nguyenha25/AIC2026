from typing import List, Dict, Optional, Literal, Union
from pydantic import BaseModel, Field

# ---------------------------------------------------------
# Sub-models cho TRAKE
# ---------------------------------------------------------
class TrakeEvent(BaseModel):
    event_id: str
    text: str
    relation: str

class TrakeOutputConfig(BaseModel):
    one_frame_per_event: bool = True
    strict_order: bool = True

# ---------------------------------------------------------
# Schema Q&A
# ---------------------------------------------------------
class QueryPlanQA(BaseModel):
    schema_version: Literal["1.1"] = "1.1"
    query_id: str
    task: Literal["qa"]
    query_text: str
    language: str = "vi"
    intent: str
    answer_type: str
    entities: List[str] = Field(default_factory=list)
    attributes: List[str] = Field(default_factory=list)
    actions: List[str] = Field(default_factory=list)
    temporal_relation: str = "none"
    preferred_modalities: Dict[str, float] = Field(
        default_factory=lambda: {"clip_l": 0.0, "ocr": 0.0, "asr": 0.0, "caption": 0.0}
    )
    queries: Dict[str, List[str]] = Field(
        default_factory=lambda: {"clip_l": [], "ocr": [], "asr": []}
    )
    uncertainty: float = 0.0
    status: Literal["ok", "fallback", "error"] = "ok"
    error: Optional[str] = None

# ---------------------------------------------------------
# Schema TRAKE
# ---------------------------------------------------------
class QueryPlanTRAKE(BaseModel):
    schema_version: Literal["1.1"] = "1.1"
    query_id: str
    task: Literal["trake"]
    language: str = "vi"
    events: List[TrakeEvent]
    output: TrakeOutputConfig = Field(default_factory=TrakeOutputConfig)
    status: Literal["ok", "fallback", "error"] = "ok"
    error: Optional[str] = None

# ---------------------------------------------------------
# Union Schema chung
# ---------------------------------------------------------
QueryPlan = Union[QueryPlanQA, QueryPlanTRAKE]