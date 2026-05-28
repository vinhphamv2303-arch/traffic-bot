
from collections import defaultdict
from pathlib import Path
from .common import ensure_dir, read_jsonl, write_jsonl, write_json, write_csv, normalize_surface, token_count, label_to_filename

def load_match_blocklist(gazetteer_root):
    path = gazetteer_root / "match_blocklist.txt"
    if not path.exists():
        return set()
    terms = set()
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            term = normalize_surface(line.split("#", 1)[0].lstrip("\ufeff"))
            if term:
                terms.add(term)
    return terms

# Các cụm này không hẳn sai, nhưng quá rộng để làm anchor mạnh trong graph.
DEFAULT_DOWNWEIGHT = {
    "sản xuất",
    "vận chuyển",
    "bảo quản",
    "mua bán",
    "mua bán trái phép",
    "tài liệu",
    "giấy phép",
    "chứng chỉ",
    "đường bộ",
    "động cơ",
    "nhãn hiệu",
    "bánh xe",
    "cán bộ",
    "học viên",
}

# Các cụm này thường quá nhiễu, không nên dùng làm entity link.
DEFAULT_REJECT = {
    "phương tiện",
    "xe",
    "đường",
    "thiết bị",
    "hệ thống",
    "giấy tờ",
    "hồ sơ",
    "quy định",
    "đối tượng",
    "cá nhân",
    "tổ chức",
}

# Một số cụm ngắn nhưng domain-specific, nên giữ.
ALWAYS_KEEP = {
    "xe mô tô",
    "xe gắn máy",
    "xe ô tô",
    "xe cơ giới",
    "chủ xe",
    "chủ phương tiện",
    "người lái xe",
    "đường cao tốc",
    "sân sát hạch",
    "sân tập lái",
    "phù hiệu",
    "hộ chiếu",
    "căn cước",
    "căn cước công dân",
    "số khung",
    "số máy",
    "màu sơn",
}

def classify_alias(item, downweight_set=None, reject_set=None, blocklist_set=None, min_tokens_keep=2):
    surface = normalize_surface(item.get("surface") or "")
    canonical = normalize_surface(item.get("canonical") or surface)
    label = item.get("label") or ""
    downweight_set = downweight_set or DEFAULT_DOWNWEIGHT
    reject_set = reject_set or DEFAULT_REJECT
    blocklist_set = blocklist_set or set()

    if surface in ALWAYS_KEEP or canonical in ALWAYS_KEEP:
        return "keep", 1.0, "always_keep"

    if surface in blocklist_set or canonical in blocklist_set:
        return "reject", 0.0, "match_blocklist"

    if surface in reject_set or canonical in reject_set:
        return "reject", 0.0, "manual_reject_generic"

    if surface in downweight_set or canonical in downweight_set:
        return "downweight", 0.25, "manual_downweight_generic_hub"

    # Short one-token anchors are usually too broad unless whitelisted.
    if token_count(surface) <= 1:
        return "downweight", 0.25, "one_token_surface"

    # Generic single-word labels in BEHAVIOR/DOCUMENT/INFRASTRUCTURE are risky.
    if token_count(surface) <= min_tokens_keep and label in {"BEHAVIOR", "DOCUMENT", "INFRASTRUCTURE", "VEHICLE_CONDITION_OR_EQUIPMENT"}:
        # keep if it has a very specific legal/domain compound marker
        safe_markers = ["lái xe", "đăng ký", "kiểm định", "cao tốc", "thu phí", "giám sát", "hành trình", "tín hiệu"]
        if not any(m in surface for m in safe_markers):
            return "downweight", 0.5, "short_surface_sensitive_label"

    return "keep", 1.0, "specific_enough"

def prune_gazetteer(gazetteer_root, output_dir, drop_reject=True):
    gazetteer_root = Path(gazetteer_root)
    output_dir = ensure_dir(output_dir)

    aliases = list(read_jsonl(gazetteer_root / "aliases.jsonl"))
    pruned_aliases = []
    rejected = []
    downweighted = []
    kept = []
    blocklist_set = load_match_blocklist(gazetteer_root)

    for item in aliases:
        mode, weight, reason = classify_alias(item, blocklist_set=blocklist_set)
        out = dict(item)
        out["match_mode"] = mode
        out["graph_weight"] = weight
        out["prune_reason"] = reason

        if mode == "reject":
            rejected.append(out)
            if not drop_reject:
                pruned_aliases.append(out)
        else:
            pruned_aliases.append(out)
            if mode == "downweight":
                downweighted.append(out)
            else:
                kept.append(out)

    # longest first for matcher
    pruned_aliases.sort(key=lambda x: (-len(x.get("surface") or ""), x.get("surface") or ""))
    write_jsonl(output_dir / "aliases.jsonl", pruned_aliases)
    write_jsonl(output_dir / "rejected_aliases.jsonl", rejected)
    write_jsonl(output_dir / "generic_hubs.jsonl", downweighted)

    # Rebuild canonical entities from kept + downweighted
    nodes = {}
    by_label = defaultdict(list)
    for a in pruned_aliases:
        label = a.get("label")
        canonical = a.get("canonical")
        key = (label, canonical)
        if key not in nodes:
            nodes[key] = {
                "entity_id": a.get("entity_id"),
                "canonical": canonical,
                "label": label,
                "aliases": set(),
                "count": 0,
                "min_graph_weight": 1.0,
                "is_generic_hub": False,
            }
        nodes[key]["aliases"].add(a.get("surface"))
        nodes[key]["count"] += int(a.get("count") or 0)
        nodes[key]["min_graph_weight"] = min(nodes[key]["min_graph_weight"], float(a.get("graph_weight", 1.0)))
        if a.get("match_mode") == "downweight":
            nodes[key]["is_generic_hub"] = True
        by_label[label].append(a.get("surface"))

    canonical_rows = []
    for n in nodes.values():
        canonical_rows.append({
            **n,
            "aliases": sorted(n["aliases"], key=lambda x: (-len(x or ""), x or "")),
        })
    canonical_rows.sort(key=lambda x: (x.get("label") or "", x.get("canonical") or ""))
    write_jsonl(output_dir / "canonical_entities.jsonl", canonical_rows)

    # Write label txt files for existing matcher
    for label, terms in by_label.items():
        terms = sorted(set([t for t in terms if t]), key=lambda x: (-len(x), x))
        with open(output_dir / label_to_filename(label), "w", encoding="utf-8") as f:
            for t in terms:
                f.write(t + "\n")

    csv_rows = []
    for a in pruned_aliases:
        csv_rows.append({
            "entity_id": a.get("entity_id"),
            "surface": a.get("surface"),
            "canonical": a.get("canonical"),
            "label": a.get("label"),
            "count": a.get("count"),
            "match_mode": a.get("match_mode"),
            "graph_weight": a.get("graph_weight"),
            "prune_reason": a.get("prune_reason"),
        })
    write_csv(output_dir / "gazetteer_terms_pruned.csv", csv_rows, [
        "entity_id", "surface", "canonical", "label", "count", "match_mode", "graph_weight", "prune_reason"
    ])

    summary = {
        "input_gazetteer_root": str(gazetteer_root),
        "input_alias_count": len(aliases),
        "output_alias_count": len(pruned_aliases),
        "kept_alias_count": len(kept),
        "downweighted_alias_count": len(downweighted),
        "rejected_alias_count": len(rejected),
        "canonical_entity_count": len(canonical_rows),
        "drop_reject": drop_reject,
        "blocklist_count": len(blocklist_set),
        "blocklist_terms": sorted(blocklist_set),
        "outputs": {
            "aliases_jsonl": str(output_dir / "aliases.jsonl"),
            "generic_hubs_jsonl": str(output_dir / "generic_hubs.jsonl"),
            "rejected_aliases_jsonl": str(output_dir / "rejected_aliases.jsonl"),
            "canonical_entities_jsonl": str(output_dir / "canonical_entities.jsonl"),
        }
    }
    write_json(output_dir / "gazetteer_summary.json", summary)
    return summary
