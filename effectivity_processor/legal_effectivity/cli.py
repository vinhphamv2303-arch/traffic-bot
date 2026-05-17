
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Tuple
from .extractor import EffectivityExtractor
from .models import EffectivityConfig
from .utils import find_units_files, units_file_output_name

def main() -> None:
    ap = argparse.ArgumentParser(description="Extract legal effectivity/repeal events from parsed units.jsonl.")
    ap.add_argument("--input", "-i", required=True, help="Path to parsed folder or one units.jsonl")
    ap.add_argument("--output", "-o", default="./data/preprocessed/effectivity", help="Output base folder")
    ap.add_argument("--scan-all-units", action="store_true", help="Scan all units, not only likely final provisions")
    ap.add_argument("--min-confidence", type=float, default=0.35)
    args = ap.parse_args()

    config = EffectivityConfig(
        output_base_dir=Path(args.output),
        prefer_final_provisions=not args.scan_all_units,
        min_confidence=args.min_confidence,
    )
    extractor = EffectivityExtractor(config)
    units_files = find_units_files(args.input)
    if not units_files:
        print(f"⚠️ Không tìm thấy units.jsonl trong: {args.input}")
        return

    print(f"🔍 Tìm thấy {len(units_files)} file units.jsonl")
    print("-" * 60)
    success, total_events = 0, 0
    all_events = []
    successful_units_files = []
    errors: List[Tuple[str, str]] = []
    for units_path in units_files:
        try:
            events = extractor.extract_from_units_file(units_path)
            success += 1
            total_events += len(events)
            all_events.extend(events)
            successful_units_files.append(units_path)
            print(f"✅ {units_file_output_name(units_path)}: {len(events)} events")
        except Exception as e:
            errors.append((str(units_path), str(e)))
            print(f"❌ {units_path}: {e}")

    if successful_units_files:
        index_path = extractor.write_effectivity_index_csv(
            successful_units_files,
            all_events,
            Path(args.output) / "effectivity_index.csv",
        )
        unit_overrides_path = extractor.write_unit_overrides_csv(
            all_events,
            Path(args.output) / "effectivity_unit_overrides.csv",
        )
        unresolved_path = extractor.write_unresolved_effectivity_csv(
            all_events,
            Path(args.output) / "effectivity_unresolved.csv",
        )
        print(f"📄 Index CSV: {index_path}")
        print(f"📄 Unit overrides CSV: {unit_overrides_path}")
        print(f"📄 Unresolved CSV: {unresolved_path}")

    print("\n" + "=" * 60)
    print(f"🎉 Thành công: {success}/{len(units_files)} | Tổng events: {total_events}")
    if errors:
        print(f"❌ Thất bại: {len(errors)}")
        for path, msg in errors:
            print(f"   - {path}: {msg}")

if __name__ == "__main__":
    main()
