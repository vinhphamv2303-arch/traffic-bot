from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from .common import (
    LABELS,
    ensure_dir,
    normalize_surface,
    read_csv,
    read_jsonl,
    stable_id,
    write_json,
    write_jsonl,
)


MANUAL_SEEDS = [
    # BEHAVIOR
    ("không đội mũ bảo hiểm", "BEHAVIOR", "manual_core"),
    ("không cài quai đúng quy cách", "BEHAVIOR", "manual_core"),
    ("không chấp hành hiệu lệnh đèn tín hiệu giao thông", "BEHAVIOR", "manual_core"),
    ("vượt đèn đỏ", "BEHAVIOR", "manual_alias"),
    ("đi ngược chiều", "BEHAVIOR", "manual_core"),
    ("chở quá số người quy định", "BEHAVIOR", "manual_core"),
    ("chạy quá tốc độ quy định", "BEHAVIOR", "manual_core"),
    ("sử dụng rượu bia khi điều khiển xe", "BEHAVIOR", "manual_core"),
    ("không có giấy phép lái xe", "CONDITION", "manual_core"),

    # VEHICLE
    ("xe mô tô", "VEHICLE", "manual_core"),
    ("xe gắn máy", "VEHICLE", "manual_core"),
    ("xe ô tô", "VEHICLE", "manual_core"),
    ("xe máy chuyên dùng", "VEHICLE", "manual_core"),
    ("xe đạp máy", "VEHICLE", "manual_core"),

    # DOCUMENT
    ("giấy phép lái xe", "DOCUMENT", "manual_core"),
    ("giấy đăng ký xe", "DOCUMENT", "manual_core"),
    ("chứng nhận kiểm định", "DOCUMENT", "manual_core"),
    ("giấy chứng nhận đăng ký xe", "DOCUMENT", "manual_core"),

    # INFRASTRUCTURE
    ("đèn tín hiệu giao thông", "INFRASTRUCTURE", "manual_core"),
    ("biển báo hiệu đường bộ", "INFRASTRUCTURE", "manual_core"),
    ("đường cao tốc", "INFRASTRUCTURE", "manual_core"),
    ("làn đường", "INFRASTRUCTURE", "manual_core"),

    # VEHICLE_CONDITION_OR_EQUIPMENT
    ("không có gương chiếu hậu", "VEHICLE_CONDITION_OR_EQUIPMENT", "manual_core"),
    ("thay đổi kết cấu xe", "VEHICLE_CONDITION_OR_EQUIPMENT", "manual_core"),
    ("biển số bị che lấp", "VEHICLE_CONDITION_OR_EQUIPMENT", "manual_core"),
    ("thiết bị giám sát hành trình", "VEHICLE_CONDITION_OR_EQUIPMENT", "manual_core"),

    # ACTOR
    ("người điều khiển xe mô tô", "ACTOR", "manual_core"),
    ("người điều khiển xe ô tô", "ACTOR", "manual_core"),
    ("chủ xe", "ACTOR", "manual_core"),
    ("cảnh sát giao thông", "ACTOR", "manual_core"),
]


def load_gazetteer_seeds(gazetteer_root: str | Path, max_per_label: int = 30) -> List[Dict[str, Any]]:
    p = Path(gazetteer_root) / "aliases.jsonl"
    if not p.exists():
        return []

    by_label = defaultdict(list)
    for a in read_jsonl(p):
        if a.get("match_mode") == "reject":
            continue
        label = a.get("label")
        surface = normalize_surface(a.get("surface") or "")
        if not surface or label not in LABELS:
            continue
        # Prefer keep over downweight, longer terms over short hubs.
        priority = 2 if a.get("match_mode") == "keep" else 1
        by_label[label].append((priority, len(surface), surface, a))

    seeds = []
    for label, items in by_label.items():
        items = sorted(items, key=lambda x: (-x[0], -x[1], x[2]))[:max_per_label]
        for _, _, surface, a in items:
            seeds.append({
                "seed_id": stable_id(label, surface, prefix="seed"),
                "surface": surface,
                "label": label,
                "canonical": a.get("canonical") or surface,
                "source": "gazetteer_pruned",
                "entity_id": a.get("entity_id"),
                "match_mode": a.get("match_mode", "keep"),
            })
    return seeds


def load_manual_seed_file(path: str | Path | None) -> List[Dict[str, Any]]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []

    seeds = []
    if p.suffix.lower() == ".jsonl":
        for r in read_jsonl(p):
            label = r.get("label")
            surface = normalize_surface(r.get("surface") or r.get("text") or "")
            if surface and label in LABELS:
                seeds.append({
                    "seed_id": stable_id(label, surface, prefix="seed"),
                    "surface": surface,
                    "label": label,
                    "canonical": normalize_surface(r.get("canonical") or surface),
                    "source": r.get("source") or "manual_file",
                })
        return seeds

    for r in read_csv(p):
        label = r.get("label")
        surface = normalize_surface(r.get("surface") or r.get("text") or "")
        if surface and label in LABELS:
            seeds.append({
                "seed_id": stable_id(label, surface, prefix="seed"),
                "surface": surface,
                "label": label,
                "canonical": normalize_surface(r.get("canonical") or surface),
                "source": r.get("source") or "manual_file",
            })
    return seeds


def build_seeds(
    gazetteer_root: str | Path,
    output_dir: str | Path,
    manual_seed_file: str | Path | None = None,
    max_gazetteer_seeds_per_label: int = 30,
) -> Dict[str, Any]:
    output_dir = ensure_dir(output_dir)

    seeds = []
    for surface, label, source in MANUAL_SEEDS:
        seeds.append({
            "seed_id": stable_id(label, surface, prefix="seed"),
            "surface": surface,
            "label": label,
            "canonical": surface,
            "source": source,
        })

    seeds.extend(load_gazetteer_seeds(gazetteer_root, max_per_label=max_gazetteer_seeds_per_label))
    seeds.extend(load_manual_seed_file(manual_seed_file))

    # Dedupe.
    dedup = {}
    for s in seeds:
        key = (s["label"], normalize_surface(s["surface"]))
        if key not in dedup or s.get("source") == "manual_core":
            dedup[key] = s
    seeds = sorted(dedup.values(), key=lambda x: (x["label"], x["surface"]))

    by_label = defaultdict(int)
    for s in seeds:
        by_label[s["label"]] += 1

    out_path = output_dir / "seeds.jsonl"
    write_jsonl(out_path, seeds)
    summary = {
        "seed_count": len(seeds),
        "by_label": dict(sorted(by_label.items())),
        "output": str(out_path),
    }
    write_json(output_dir / "seed_summary.json", summary)
    return summary
