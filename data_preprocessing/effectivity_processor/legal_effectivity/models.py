
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

@dataclass
class EffectivityConfig:
    output_base_dir: Path = Path("./data/preprocessed/effectivity")
    prefer_final_provisions: bool = True
    infer_repeal_date_from_document_effective_date: bool = True
    min_confidence: float = 0.35

@dataclass
class EffectivityEvent:
    event_id: str
    event_type: str
    source_document_id: Optional[str]
    source_document_number: Optional[str]
    source_unit_id: Optional[str]
    source_path_text: Optional[str]
    target_scope: str
    target_selector_raw: Optional[str]
    target_document_number: Optional[str]
    target_unit_selector: Optional[Dict[str, Any]]
    date: Optional[str]
    date_role: Optional[str]
    raw_text: str
    status: str
    resolver: str
    confidence: float
    date_inference: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "source_document_id": self.source_document_id,
            "source_document_number": self.source_document_number,
            "source_unit_id": self.source_unit_id,
            "source_path_text": self.source_path_text,
            "target_scope": self.target_scope,
            "target_selector_raw": self.target_selector_raw,
            "target_document_number": self.target_document_number,
            "target_unit_selector": self.target_unit_selector,
            "date": self.date,
            "date_role": self.date_role,
            "date_inference": self.date_inference,
            "raw_text": self.raw_text,
            "status": self.status,
            "resolver": self.resolver,
            "confidence": self.confidence,
            "notes": self.notes,
        }
