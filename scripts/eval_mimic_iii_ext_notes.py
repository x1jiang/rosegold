#!/usr/bin/env python3
"""Score concept predictions or emit MIMIC-III-Ext-Notes evaluation prompts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.mimic_ext_notes import (
    build_concept_eval_prompt,
    join_labels,
    load_source_bundle,
    phenotype_gold,
    score_label_predictions,
    validate_labels,
    validate_notes,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate or prompt MIMIC-III-Ext-Notes labels.")
    parser.add_argument("--source", help="Directory containing notes.csv and labels.csv")
    parser.add_argument("--notes", help="Path to notes.csv")
    parser.add_argument("--labels", help="Path to labels.csv")
    parser.add_argument("--predictions", help="CSV/JSONL of row_id,concept,detection,encounter,negation")
    parser.add_argument("--write-prompts", help="Write concept-eval prompt JSONL to this path")
    parser.add_argument("--target_condition", default="Sepsis / Septic Shock")
    parser.add_argument("--phenotype-gold", help="Write visit-level phenotype gold CSV")
    args = parser.parse_args(argv)

    if args.source:
        notes, labels = load_source_bundle(args.source)
        if labels is None:
            raise SystemExit("labels.csv is required for evaluation")
    else:
        if not args.notes or not args.labels:
            raise SystemExit("provide --source or both --notes and --labels")
        notes = validate_notes(pd.read_csv(args.notes))
        labels = validate_labels(pd.read_csv(args.labels), notes)

    if args.write_prompts:
        merged = join_labels(notes, labels)
        with open(args.write_prompts, "w", encoding="utf-8") as handle:
            for row in merged.itertuples(index=False):
                handle.write(
                    json.dumps(
                        {
                            "row_id": int(row.row_id),
                            "hadm_id": int(row.hadm_id),
                            "concept": row.concept,
                            "trigger_word": row.trigger_word,
                            "prompt": build_concept_eval_prompt(row.text, row.concept, row.trigger_word),
                        }
                    )
                    + "\n"
                )
        print(f"Wrote {len(merged)} concept prompts to {args.write_prompts}")

    if args.phenotype_gold:
        gold = phenotype_gold(notes, labels, args.target_condition)
        gold.to_csv(args.phenotype_gold, index=False)
        print(f"Wrote phenotype gold for {args.target_condition} to {args.phenotype_gold}")

    if args.predictions:
        if args.predictions.endswith(".jsonl"):
            preds = pd.read_json(args.predictions, lines=True)
        else:
            preds = pd.read_csv(args.predictions)
        metrics = score_label_predictions(labels, preds)
        print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
