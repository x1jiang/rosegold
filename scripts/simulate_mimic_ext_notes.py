#!/usr/bin/env python3
"""Simulate MIMIC-III-Ext-Notes tables from the synthetic OMOP cohort."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.mimic_ext_simulate import simulate_from_omop


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Simulate MIMIC-III-Ext-Notes CSVs from synthetic OMOP notes.")
    parser.add_argument("--notes_path", default="data/synthetic_notes.csv")
    parser.add_argument("--output", default="data/synthetic_mimic_ext_notes")
    args = parser.parse_args(argv)
    paths = simulate_from_omop(args.notes_path, args.output)
    for key, value in paths.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
