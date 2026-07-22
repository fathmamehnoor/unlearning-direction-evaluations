#!/usr/bin/env python
"""Assemble the one unambiguous recovery table from recovery_results.csv.

Reads the appended per-checkpoint rows and prints (and writes) a table where
every recovery number sits next to its utility number, plus deltas against each
model's own no-SFT baseline. The reading rule the table is built to support:

  * WMDP-bio up + MMLU flat            -> real recovery (suppression, not removal)
  * WMDP-bio and MMLU up together      -> SFT churn, not hidden knowledge
  * WMDP-bio prob rising ahead of acc  -> knowledge surfacing internally first
  * unlearned model moves differently  -> the change is about the unlearning,
    from the full-knowledge control        not a generic SFT side effect

No third-party deps; pure stdlib so it runs anywhere the CSV lands.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Optional


def to_float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value, pct: bool = False) -> str:
    f = to_float(value)
    if f is None:
        return "--"
    return f"{f * 100:.1f}%" if pct else f"{f:.3f}"


def signed(value: Optional[float], pct: bool = False, digits: int = 1) -> str:
    if value is None:
        return "--"
    scaled = value * 100 if pct else value
    return f"{scaled:+.{digits}f}" + ("pp" if pct else "")


def load_rows(csv_path: Path) -> List[dict]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def baseline_for(rows: List[dict], model_key: str) -> Optional[dict]:
    for row in rows:
        if row.get("_model_key") == model_key and to_float(row.get("checkpoint_step")) == 0:
            return row
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-csv", type=Path, default=Path("results/wmdp_sft_recovery/recovery_results.csv"))
    parser.add_argument("--out-md", type=Path, default=Path("results/wmdp_sft_recovery/recovery_table.md"))
    args = parser.parse_args()

    rows = load_rows(args.results_csv)
    # Group by (arm, unlearning_method) -- the model identity independent of the
    # per-checkpoint label suffix.
    for row in rows:
        row["_model_key"] = f"{row.get('arm','')}::{row.get('unlearning_method','')}"

    keys: List[str] = []
    for row in rows:
        if row["_model_key"] not in keys:
            keys.append(row["_model_key"])

    header = (
        "| model (arm) | step | WMDP-bio acc | WMDP-bio prob | MMLU | GSM8K | "
        "dWMDP acc | dWMDP prob | dMMLU |"
    )
    sep = "|" + "---|" * 9
    lines = [
        "# WMDP unrelated-SFT recovery",
        "",
        "Recovery (WMDP-bio) read next to utility (MMLU) and fine-tune-took (GSM8K).",
        "`d*` columns are deltas vs each model's own step-0 (no-SFT) baseline.",
        "",
        "GSM8K confirms *uptake only* -- read the rise from baseline, not the absolute",
        "level (chat template + few-shot makes the level odd). If bio stays flat, a "
        "risen GSM8K rules out \"nothing happened\".",
        "",
        header,
        sep,
    ]
    warnings: List[str] = []

    for key in keys:
        model_rows = [r for r in rows if r["_model_key"] == key]
        model_rows.sort(key=lambda r: to_float(r.get("checkpoint_step")) or 0)
        base = baseline_for(rows, key)
        b_wmdp = to_float(base.get("wmdp_bio_acc")) if base else None
        b_prob = to_float(base.get("wmdp_bio_correct_option_prob")) if base else None
        b_mmlu = to_float(base.get("mmlu_acc")) if base else None
        arm = model_rows[0].get("arm", "")
        method = model_rows[0].get("unlearning_method", "")
        name = f"{method} ({arm})"

        # Integrity guards: the recovery delta is baseline-minus-SFT, so all rows
        # of a model MUST share one precision, and any row whose probability
        # self-check failed cannot be trusted. Surface both loudly.
        if base is None:
            warnings.append(
                f"{name}: NO step-0 (no-SFT) baseline row -- every delta is "
                f"undefined (dashes) until you run the baseline eval for this model."
            )
        precisions = {r.get("eval_precision", "") for r in model_rows if r.get("eval_precision", "")}
        if len(precisions) > 1:
            warnings.append(
                f"{name}: MIXED eval precision across rows {sorted(precisions)} -- "
                f"the recovery delta is invalid; re-run every row at one precision."
            )
        bad_prob = [r.get("checkpoint_step", "?") for r in model_rows if r.get("prob_check", "ok") == "MISMATCH"]
        if bad_prob:
            warnings.append(
                f"{name}: probability self-check FAILED at step(s) {bad_prob} -- "
                f"the WMDP-bio prob column is untrustworthy for those rows."
            )

        for row in model_rows:
            wmdp = to_float(row.get("wmdp_bio_acc"))
            prob = to_float(row.get("wmdp_bio_correct_option_prob"))
            mmlu = to_float(row.get("mmlu_acc"))
            d_wmdp = (wmdp - b_wmdp) if (wmdp is not None and b_wmdp is not None) else None
            d_prob = (prob - b_prob) if (prob is not None and b_prob is not None) else None
            d_mmlu = (mmlu - b_mmlu) if (mmlu is not None and b_mmlu is not None) else None
            lines.append(
                f"| {name} | {row.get('checkpoint_step','')} | "
                f"{fmt(wmdp, pct=True)} | {fmt(prob)} | {fmt(mmlu, pct=True)} | "
                f"{fmt(row.get('gsm8k_acc'), pct=True)} | "
                f"{signed(d_wmdp, pct=True)} | {signed(d_prob, digits=3)} | {signed(d_mmlu, pct=True)} |"
            )

    if warnings:
        lines.append("")
        lines.append("## Integrity warnings")
        lines.append("")
        for w in warnings:
            lines.append(f"- {w}")

    table = "\n".join(lines) + "\n"
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(table, encoding="utf-8")
    print(table)
    if warnings:
        print("\n".join("WARNING: " + w for w in warnings))
    print(f"wrote {args.out_md}")


if __name__ == "__main__":
    main()
