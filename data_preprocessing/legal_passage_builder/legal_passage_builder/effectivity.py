
from pathlib import Path
from .utils import canonical_key, none_if_null, read_csv_dicts, read_jsonl, unit_selector_key

class EffectivityStore:
    def __init__(self, effectivity_root):
        self.root = Path(effectivity_root) if effectivity_root else None
        self.document_dates_by_id = {}
        self.document_dates_by_number = {}
        self.unit_overrides = {}
        if self.root:
            self._load_index()
            self._load_unit_overrides()

    def _load_index(self):
        for row in read_csv_dicts(self.root / "effectivity_index.csv"):
            doc_id = row.get("document_id") or ""
            doc_no = row.get("document_number") or ""
            effective_from = none_if_null(row.get("effective_from") or row.get("date"))
            effective_to = none_if_null(row.get("effective_to"))
            dates = {
                "effective_from": effective_from,
                "ceased_from": none_if_null(row.get("ceased_from")) or effective_to,
                "effective_to": effective_to,
                "effectivity_status": row.get("status") or row.get("effectivity_status") or self._default_status(effective_from, effective_to),
                "raw": row,
            }
            if doc_id:
                self.document_dates_by_id[doc_id] = dates
            if doc_no:
                self.document_dates_by_number[doc_no] = dates

    @staticmethod
    def _default_status(effective_from, effective_to):
        if effective_to:
            return "ceased"
        if effective_from:
            return "active"
        return "unknown"

    def _load_unit_overrides(self):
        for row in read_csv_dicts(self.root / "effectivity_unit_overrides.csv"):
            doc_id = row.get("document_id") or ""
            key = (
                doc_id,
                canonical_key(row.get("target_article") or ""),
                canonical_key(row.get("target_clause") or ""),
                canonical_key(row.get("target_point") or ""),
            )
            self.unit_overrides[key] = {
                "effective_from": none_if_null(row.get("effective_from")),
                "ceased_from": none_if_null(row.get("ceased_from") or row.get("effective_to")),
                "effective_to": none_if_null(row.get("effective_to")),
                "effectivity_status": row.get("status") or row.get("effectivity_status") or "unit_override",
                "raw": row,
            }

    def load_package_events(self, package_id):
        if not self.root:
            return []
        p = self.root / package_id / "effectivity_events.jsonl"
        return list(read_jsonl(p)) if p.exists() else []

    def dates_for_unit(self, unit):
        doc_id = unit.get("document_id") or ""
        doc_no = unit.get("document_number") or ""
        base = (self.document_dates_by_id.get(doc_id) or self.document_dates_by_number.get(doc_no) or {
            "effective_from": None,
            "ceased_from": None,
            "effective_to": None,
            "effectivity_status": "unknown",
            "raw": None,
        }).copy()
        key = unit_selector_key(unit)
        for k in [key, (key[0], key[1], key[2], ""), (key[0], key[1], "", "")]:
            if k in self.unit_overrides:
                ov = self.unit_overrides[k]
                base.update({kk: vv for kk, vv in ov.items() if kk != "raw" or vv})
                base["unit_effectivity_override"] = ov.get("raw")
                break
        return base
