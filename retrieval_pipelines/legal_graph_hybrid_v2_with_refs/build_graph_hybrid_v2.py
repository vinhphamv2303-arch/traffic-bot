
import argparse

try:
    from .legal_graph_hybrid_v2.graph_builder import build_graph
except ImportError:
    from legal_graph_hybrid_v2.graph_builder import build_graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity-links-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--references-file", default=None)
    parser.add_argument("--min-cooccur-weight", type=float, default=0.05)
    parser.add_argument("--max-entity-degree-for-cooccur", type=int, default=2500)
    parser.add_argument("--no-create-missing-reference-targets", action="store_true")
    args = parser.parse_args()

    print(build_graph(
        entity_links_dir=args.entity_links_dir,
        output_dir=args.output,
        references_file=args.references_file,
        min_cooccur_weight=args.min_cooccur_weight,
        max_entity_degree_for_cooccur=args.max_entity_degree_for_cooccur,
        create_missing_reference_targets=not args.no_create_missing_reference_targets,
    ))


if __name__ == "__main__":
    main()
