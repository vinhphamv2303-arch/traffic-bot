
import argparse

try:
    from .legal_graph_hybrid_v2.entity_links import build_hybrid_entity_links
except ImportError:
    from legal_graph_hybrid_v2.entity_links import build_hybrid_entity_links


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gazetteer-entities-root", required=True)
    parser.add_argument("--gliner-entities-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--no-inherited", action="store_true")
    args = parser.parse_args()

    summary = build_hybrid_entity_links(
        args.gazetteer_entities_root,
        args.gliner_entities_root,
        args.output,
        include_inherited=not args.no_inherited,
    )
    print(summary)


if __name__ == "__main__":
    main()
