from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from .common import (
    ensure_dir,
    normalize_surface,
    read_csv,
    read_jsonl,
    stable_id,
    write_csv,
    write_json,
    write_jsonl,
)


REVIEWED_ACCEPT_VALUES = {"accept", "accepted", "keep"}
AUTO_ACCEPT_VALUES = {"accept_candidate", "auto_accept"}


def build_gazetteer_v2(
    base_gazetteer_root: str | Path,
    reviewed_mined_csv: str | Path,
    output_dir: str | Path,
    min_score: float = 0.0,
    accept_auto_candidates: bool = False,
) -> Dict[str, Any]:
    output_dir = ensure_dir(output_dir)

    aliases = []
    base_alias_path = Path(base_gazetteer_root) / "aliases.jsonl"
    if base_alias_path.exists():
        aliases.extend(list(read_jsonl(base_alias_path)))

    added = []
    accept_values = set(REVIEWED_ACCEPT_VALUES)
    if accept_auto_candidates:
        accept_values.update(AUTO_ACCEPT_VALUES)

    for r in read_csv(reviewed_mined_csv):
        status = (r.get("status") or "").strip().lower()
        if status not in accept_values:
            continue
        try:
            score = float(r.get("score") or 0)
        except Exception:
            score = 0.0
        if score < min_score:
            continue

        surface = normalize_surface(r.get("surface") or "")
        label = (r.get("label") or "").strip()
        canonical = normalize_surface(r.get("canonical") or surface)
        if not surface or not label:
            continue

        item = {
            "entity_id": stable_id(label, canonical, prefix="ent"),
            "surface": surface,
            "canonical": canonical,
            "label": label,
            "count": int(float(r.get("count") or 1)),
            "match_mode": "keep",
            "graph_weight": 1.0,
            "source": "xner_mined_reviewed",
            "score": score,
        }
        added.append(item)
        aliases.append(item)

    # Dedupe by label+surface; prefer base or higher score.
    dedup = {}
    for a in aliases:
        key = (a.get("label"), normalize_surface(a.get("surface") or ""))
        if not key[0] or not key[1]:
            continue
        old = dedup.get(key)
        if old is None:
            dedup[key] = a
            continue
        old_score = float(old.get("score") or 0)
        new_score = float(a.get("score") or 0)
        if old.get("source") == "xner_mined_reviewed" and new_score > old_score:
            dedup[key] = a

    aliases = sorted(dedup.values(), key=lambda x: (x.get("label") or "", x.get("canonical") or "", x.get("surface") or ""))
    write_jsonl(output_dir / "aliases.jsonl", aliases)

    nodes = {}
    by_label = defaultdict(list)
    for a in aliases:
        key = (a.get("label"), a.get("canonical"))
        if key not in nodes:
            nodes[key] = {
                "entity_id": a.get("entity_id") or stable_id(a.get("label") or "", a.get("canonical") or "", prefix="ent"),
                "canonical": a.get("canonical"),
                "label": a.get("label"),
                "aliases": set(),
                "count": 0,
                "is_generic_hub": a.get("match_mode") == "downweight",
                "min_graph_weight": float(a.get("graph_weight", 1.0)),
            }
        nodes[key]["aliases"].add(a.get("surface"))
        nodes[key]["count"] += int(a.get("count") or 0)
        nodes[key]["min_graph_weight"] = min(nodes[key]["min_graph_weight"], float(a.get("graph_weight", 1.0)))
        if a.get("match_mode") == "downweight":
            nodes[key]["is_generic_hub"] = True
        by_label[a.get("label")].append(a.get("surface"))

    canonical_rows = []
    for n in nodes.values():
        canonical_rows.append({
            **n,
            "aliases": sorted(n["aliases"], key=lambda x: (-len(x or ""), x or "")),
        })
    canonical_rows.sort(key=lambda x: (x.get("label") or "", x.get("canonical") or ""))
    write_jsonl(output_dir / "canonical_entities.jsonl", canonical_rows)

    for label, terms in by_label.items():
        terms = sorted(set([t for t in terms if t]), key=lambda x: (-len(x), x))
        with open(output_dir / f"{label.lower()}.txt", "w", encoding="utf-8") as f:
            for t in terms:
                f.write(t + "\n")

    csv_rows = [{
        "entity_id": a.get("entity_id"),
        "surface": a.get("surface"),
        "canonical": a.get("canonical"),
        "label": a.get("label"),
        "count": a.get("count"),
        "match_mode": a.get("match_mode"),
        "graph_weight": a.get("graph_weight"),
        "source": a.get("source", "base"),
        "score": a.get("score"),
    } for a in aliases]
    write_csv(output_dir / "gazetteer_terms.csv", csv_rows, [
        "entity_id", "surface", "canonical", "label", "count", "match_mode", "graph_weight", "source", "score"
    ])

    summary = {
        "base_gazetteer_root": str(base_gazetteer_root),
        "reviewed_mined_csv": str(reviewed_mined_csv),
        "base_alias_count": len(list(read_jsonl(base_alias_path))) if base_alias_path.exists() else 0,
        "added_alias_count": len(added),
        "accept_auto_candidates": accept_auto_candidates,
        "output_alias_count": len(aliases),
        "canonical_entity_count": len(canonical_rows),
        "output_dir": str(output_dir),
    }
    write_json(output_dir / "gazetteer_summary.json", summary)
    return summary
