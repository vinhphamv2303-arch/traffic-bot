import argparse
from pathlib import Path

from .config import ResolverConfig
from .resolver import ReferenceResolver

def main():
    ap = argparse.ArgumentParser(description="Resolve legal reference mentions using rule-based inventory matching.")
    ap.add_argument("--parsed-root", "-i", required=True)
    ap.add_argument("--output", "-o", default="./data/preprocessed/resolved_references")
    ap.add_argument("--resolved-threshold", type=float, default=0.90)
    ap.add_argument("--ambiguous-threshold", type=float, default=0.65)
    ap.add_argument("--max-candidates", type=int, default=8)
    args = ap.parse_args()

    cfg = ResolverConfig(
        parsed_root=Path(args.parsed_root),
        output_root=Path(args.output),
        resolved_threshold=args.resolved_threshold,
        ambiguous_threshold=args.ambiguous_threshold,
        max_candidates=args.max_candidates,
    )
    summary = ReferenceResolver(cfg).resolve_all()
    print("Reference resolution completed")
    print(f"Packages: {summary['package_count']}")
    print(f"Total mentions: {summary['total_mentions']}")
    print(f"Resolved: {summary['total_resolved']}")
    print(f"Ambiguous: {summary['total_ambiguous']}")
    print(f"Unresolved: {summary['total_unresolved']}")
    print(f"Output: {cfg.output_root}")

if __name__ == "__main__":
    main()
