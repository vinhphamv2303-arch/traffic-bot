import argparse
from legal_graph_builder import build_legal_graph

def main():
    ap = argparse.ArgumentParser(description="Build baseline passage-entity legal graph.")
    ap.add_argument("--passages-root", required=True)
    ap.add_argument("--entity-links-root", required=True)
    ap.add_argument("--gazetteer-root", required=True)
    ap.add_argument("--output", "-o", default="./data/preprocessed/legal_graph_v1")
    ap.add_argument("--no-reference-edges", action="store_true")
    ap.add_argument("--strong-only", action="store_true", help="Use only keep/strong entity links, drop downweight links.")
    args = ap.parse_args()

    summary = build_legal_graph(
        passages_root=args.passages_root,
        entity_links_root=args.entity_links_root,
        gazetteer_root=args.gazetteer_root,
        output_dir=args.output,
        include_reference_edges=not args.no_reference_edges,
        strong_only=args.strong_only,
    )

    print("Legal graph build completed")
    print(f"Nodes: {summary['graph']['node_count']}")
    print(f"Edges: {summary['graph']['edge_count']}")
    print(f"By node type: {summary['graph']['by_node_type']}")
    print(f"By edge type: {summary['graph']['by_edge_type']}")
    if summary.get("references"):
        refs = summary["references"]
        print(
            "References: "
            f"{refs['reference_edge_count']} edges, "
            f"{refs['skipped_reference_count']} skipped"
        )
    print(f"Output: {args.output}")

if __name__ == "__main__":
    main()
