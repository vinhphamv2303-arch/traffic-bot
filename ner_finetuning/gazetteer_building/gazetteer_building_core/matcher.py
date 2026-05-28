
from collections import defaultdict
from pathlib import Path
from .common import (
    ensure_dir, read_jsonl, write_jsonl, write_json, stable_id,
    boundary_ok, find_sentence_package_dirs, collapse_ws
)

def load_blocklist(gazetteer_root):
    path = Path(gazetteer_root) / "match_blocklist.txt"
    if not path.exists():
        return set()
    terms = set()
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            term = collapse_ws(line.split("#", 1)[0]).lstrip("\ufeff").lower()
            if term:
                terms.add(term)
    return terms

def load_aliases(gazetteer_root):
    aliases = list(read_jsonl(Path(gazetteer_root) / "aliases.jsonl"))
    blocklist = load_blocklist(gazetteer_root)
    if blocklist:
        aliases = [
            a for a in aliases
            if (collapse_ws(a.get("surface") or "").lower() not in blocklist)
            and (collapse_ws(a.get("canonical") or "").lower() not in blocklist)
        ]
    aliases.sort(key=lambda x: (-len(x.get("surface") or ""), x.get("surface") or ""))
    return aliases

def find_matches(text, aliases):
    if not text:
        return []
    low = text.lower()
    occupied = [False] * len(text)
    matches = []

    for a in aliases:
        needle = (a.get("surface") or "").lower()
        if not needle:
            continue
        start = 0
        while True:
            idx = low.find(needle, start)
            if idx < 0:
                break
            end = idx + len(needle)
            if boundary_ok(text, idx, end) and not any(occupied[idx:end]):
                for i in range(idx, end):
                    occupied[i] = True
                matches.append({
                    "surface": text[idx:end],
                    "canonical": a.get("canonical"),
                    "label": a.get("label"),
                    "entity_id": a.get("entity_id"),
                    "start": idx,
                    "end": end,
                    "source": "gazetteer",
                    "confidence": 1.0,
                })
            start = idx + 1

    matches.sort(key=lambda x: (x["start"], x["end"]))
    return matches

def links_for_sentence(sentence, aliases):
    text = sentence.get("text") or ""
    links = []
    for m in find_matches(text, aliases):
        links.append({
            "link_id": stable_id(sentence.get("sentence_id") or "", m.get("label") or "", m.get("canonical") or "", str(m["start"]), str(m["end"]), prefix="link"),
            "sentence_id": sentence.get("sentence_id"),
            "passage_id": sentence.get("passage_id"),
            "source_unit_id": sentence.get("source_unit_id"),
            "package_id": sentence.get("package_id"),
            "document_id": sentence.get("document_id"),
            "document_number": sentence.get("document_number"),
            "document_title": sentence.get("document_title"),
            "source_type": sentence.get("source_type"),
            "attachment_id": sentence.get("attachment_id"),
            "unit_type": sentence.get("unit_type"),
            "path_text": sentence.get("path_text"),
            **m,
        })
    return links

def summarize(package_id, sentence_count, links):
    by_label = defaultdict(int)
    by_entity = defaultdict(int)
    sent_ids = set()
    for l in links:
        by_label[l.get("label") or "UNKNOWN"] += 1
        by_entity[(l.get("label") or "", l.get("canonical") or "")] += 1
        if l.get("sentence_id"):
            sent_ids.add(l["sentence_id"])
    top = [
        {"label": k[0], "canonical": k[1], "count": v}
        for k, v in sorted(by_entity.items(), key=lambda x: -x[1])[:50]
    ]
    return {
        "package_id": package_id,
        "sentence_count": sentence_count,
        "sentence_with_entity_count": len(sent_ids),
        "entity_count": len(links),
        "by_label": dict(sorted(by_label.items())),
        "top_entities": top,
    }

def match_all_sentence_packages(sentences_root, gazetteer_root, output_root):
    out_root = ensure_dir(output_root)
    package_dirs = find_sentence_package_dirs(sentences_root)
    aliases = load_aliases(gazetteer_root)

    all_links = []
    global_summary = {
        "package_count": len(package_dirs),
        "sentence_count": 0,
        "sentence_with_entity_count": 0,
        "entity_count": 0,
        "by_label": {},
        "packages": {},
        "gazetteer_root": str(gazetteer_root),
        "alias_count": len(aliases),
    }

    for pkg_dir in package_dirs:
        pkg = pkg_dir.name
        out_dir = ensure_dir(out_root / pkg)
        sentences = list(read_jsonl(pkg_dir / "sentences.jsonl"))
        links = []
        sentence_rows = []

        for s in sentences:
            s_links = links_for_sentence(s, aliases)
            sentence_rows.append({
                **s,
                "entities": s_links,
                "entity_count": len(s_links),
            })
            links.extend(s_links)

        write_jsonl(out_dir / "entity_mentions.jsonl", links)
        write_jsonl(out_dir / "sentences_with_entities.jsonl", sentence_rows)
        pkg_summary = summarize(pkg, len(sentences), links)
        write_json(out_dir / "entity_summary.json", pkg_summary)

        global_summary["packages"][pkg] = pkg_summary
        global_summary["sentence_count"] += pkg_summary["sentence_count"]
        global_summary["sentence_with_entity_count"] += pkg_summary["sentence_with_entity_count"]
        global_summary["entity_count"] += pkg_summary["entity_count"]
        for label, count in pkg_summary["by_label"].items():
            global_summary["by_label"][label] = global_summary["by_label"].get(label, 0) + count
        all_links.extend(links)

    global_summary["by_label"] = dict(sorted(global_summary["by_label"].items()))
    write_jsonl(out_root / "all_entity_mentions.jsonl", all_links)
    write_json(out_root / "entity_summary.json", global_summary)
    return global_summary
