#!/usr/bin/env python3
"""Convert a local MIMIC-III-Ext-Notes v1.0.0 extract into OMOP NOTE + VISIT tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.mimic_ext_notes import load_source_bundle, write_omop_bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare credentialed MIMIC-III-Ext-Notes files for Rose Gold."
    )
    parser.add_argument("--source", required=True, help="Directory containing notes.csv and optional labels.csv")
    parser.add_argument("--output", required=True, help="Directory for omop_notes.csv, omop_visits.csv, and manifest.json")
    args = parser.parse_args(argv)

    notes, labels = load_source_bundle(args.source)
    paths = write_omop_bundle(notes, args.output, labels)
    print(f"Wrote OMOP notes: {paths['notes']}")
    print(f"Wrote OMOP visits: {paths['visits']}")
    if "labels" in paths:
        print(f"Wrote validated labels: {paths['labels']}")
    print(f"Wrote manifest: {paths['manifest']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
