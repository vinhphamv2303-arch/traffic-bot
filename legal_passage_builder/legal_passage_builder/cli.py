import argparse
from pathlib import Path
from .builder import PassageBuilder
from .config import PassageBuilderConfig

def main():
    ap = argparse.ArgumentParser(description="Build LinearRAG-style legal passages from parsed units.")
    ap.add_argument("--parsed-root", "-i", required=True)
    ap.add_argument("--output", "-o", default="./data/preprocessed/passages")
    ap.add_argument("--effectivity-root", default=None)
    ap.add_argument("--resolved-refs-root", default=None)
    ap.add_argument("--no-container-passages", action="store_true")
    args = ap.parse_args()

    cfg = PassageBuilderConfig(
        parsed_root=Path(args.parsed_root),
        output_root=Path(args.output),
        effectivity_root=Path(args.effectivity_root) if args.effectivity_root else None,
        resolved_refs_root=Path(args.resolved_refs_root) if args.resolved_refs_root else None,
        include_container_passages=not args.no_container_passages,
    )
    summary = PassageBuilder(cfg).build_all()
    print("🎉 Passage building completed")
    print(f"Packages: {summary['package_count']}")
    print(f"Total passages: {summary['total_passages']}")
    print(f"Atomic passages: {summary['total_atomic']}")
    print(f"Container passages: {summary['total_container']}")
    print(f"Output: {cfg.output_root}")

if __name__ == "__main__":
    main()
