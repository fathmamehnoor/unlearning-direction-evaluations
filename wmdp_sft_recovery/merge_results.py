#!/usr/bin/env python
"""Merge recovery-result CSV rows into a canonical CSV, deduped by checkpoint.

Dedup key is (arm, unlearning_method, checkpoint_step) -- one row per model per
checkpoint. When the same key appears more than once (e.g. the Modal Volume's
running CSV overlaps rows already in your local file), the row with the latest
timestamp_utc wins. Idempotent: re-running, or merging an overlapping source,
never creates duplicates. Missing target or sources are treated as empty.

Usage:
    python merge_results.py --target results/.../recovery_results.csv \
                            --source /tmp/rows_from_modal.csv [more.csv ...]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

KEY = ("arm", "unlearning_method", "checkpoint_step")


def load(path: str) -> Tuple[List[dict], List[str]]:
    p = Path(path)
    if not p.exists():
        return [], []
    with p.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def row_key(row: dict) -> tuple:
    return tuple(str(row.get(k, "")) for k in KEY)


def _step(row: dict) -> float:
    try:
        return float(row.get("checkpoint_step") or 0)
    except (TypeError, ValueError):
        return 0.0


def merge(target: str, sources: List[str]):
    rows_by_key: dict = {}
    fieldnames: List[str] = []

    def ingest(rows: List[dict], fns: List[str]) -> None:
        for fn in fns:
            if fn not in fieldnames:
                fieldnames.append(fn)
        for row in rows:
            key = row_key(row)
            prev = rows_by_key.get(key)
            # Latest timestamp wins; ties keep the incoming (later-listed) row.
            if prev is None or str(row.get("timestamp_utc", "")) >= str(prev.get("timestamp_utc", "")):
                rows_by_key[key] = row

    # Target first, then sources (sources are newer, so they win ties).
    t_rows, t_fns = load(target)
    ingest(t_rows, t_fns)
    for src in sources:
        s_rows, s_fns = load(src)
        ingest(s_rows, s_fns)

    merged = sorted(
        rows_by_key.values(),
        key=lambda r: (r.get("arm", ""), r.get("unlearning_method", ""), _step(r)),
    )
    return merged, fieldnames


def write(path: str, rows: List[dict], fieldnames: List[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True, help="Canonical CSV to update in place.")
    ap.add_argument("--source", nargs="+", required=True, help="One or more CSVs to merge in.")
    args = ap.parse_args()

    before, _ = load(args.target)
    merged, fieldnames = merge(args.target, args.source)
    write(args.target, merged, fieldnames)
    print(f"merged -> {args.target}: {len(before)} rows before, {len(merged)} after dedup "
          f"(key = arm+method+checkpoint_step, latest timestamp wins)")


if __name__ == "__main__":
    main()
