"""Paired significance testing for two conditions scored on the SAME
question set -- e.g. a model's no-SFT baseline vs an SFT-recovered checkpoint
(wmdp_sft_recovery/eval_recovery_lm_eval.py), or a direction-ablation
condition vs baseline/another condition (unlearning_direction_evaluation's
wmdp_bio_lm_eval_ablation.py). Both scripts persist per-doc correctness to
`per_doc_correctness/<task>.json` (doc_id -> bool) precisely so this script
can consume them.

Why this exists instead of just comparing two aggregate accuracies: every
condition here is scored on the exact same ~1273 wmdp_bio questions, so the
two accuracy numbers are NOT independent samples -- treating them as such
(e.g. two separate Wilson/normal confidence intervals, or a two-sample
z-test) either overstates or understates significance depending on how
correlated the two conditions' errors are. The correct tools for paired
binary outcomes are:

1. McNemar's test -- operates only on the DISCORDANT pairs (docs where the
   two conditions disagree). With a few thousand docs and a small accuracy
   delta, discordant-pair counts are often small, where the classic
   chi-square McNemar (with continuity correction) is a poor approximation;
   this script reports the EXACT binomial McNemar test as primary (via
   scipy.stats.binomtest) and the chi-square version alongside it for
   cross-reference.
2. A paired bootstrap CI on the accuracy delta -- resampling doc_ids (not
   independently resampling each condition) preserves the pairing in every
   resample, giving a CI on "B minus A" that correctly accounts for their
   correlation instead of naively combining two marginal CIs.

Usage (comparing the selected direction's ablation against the matched
control, on the same wmdp_bio_lm_eval_ablation.py run's persisted per-doc
correctness):

    python analyze_paired_recovery.py \\
      --correctness-a outputs/ilu-rmu/wmdp_bio_lm_eval/generic_mean_diff/per_doc_correctness/wmdp_bio/selected_direction_ablation.json \\
      --correctness-b outputs/ilu-rmu/wmdp_bio_lm_eval/generic_mean_diff/per_doc_correctness/wmdp_bio/matched_control_ablation.json \\
      --label-a selected_direction --label-b matched_control \\
      --output-json outputs/ilu-rmu/wmdp_bio_lm_eval/generic_mean_diff/mcnemar_selected_vs_matched_control.json

Or against baseline, or against any individual random_direction_ablation_i.json
in the same per_doc_correctness/wmdp_bio/ directory -- this script only cares
that both files are doc_id -> bool maps over the same question set; it has no
opinion about which pipeline produced them (the identical script also lives
in wmdp_sft_recovery/ for the SFT-recovery arm's baseline-vs-SFT-checkpoint
comparisons).

    python analyze_paired_recovery.py \\
      --correctness-a results/wmdp_sft_recovery/eval/idk_ap_wmdp_llama3_8b_step0/per_doc_correctness/wmdp_bio.json \\
      --correctness-b results/wmdp_sft_recovery/eval/idk_ap_wmdp_llama3_8b_gsm8k_6000_step6000/per_doc_correctness/wmdp_bio.json \\
      --label-a idk_ap_baseline --label-b idk_ap_sft_6000 \\
      --output-json results/wmdp_sft_recovery/idk_ap_paired_analysis.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from scipy.stats import binomtest, chi2
except ModuleNotFoundError:
    binomtest = None
    chi2 = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--correctness-a", required=True, help="Path to condition A's per_doc_correctness/<task>.json.")
    parser.add_argument("--correctness-b", required=True, help="Path to condition B's per_doc_correctness/<task>.json.")
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--num-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def require_scipy() -> None:
    if binomtest is None or chi2 is None:
        raise ModuleNotFoundError("scipy is required (pip install scipy) -- it ships as a transitive "
                                   "dependency of lm_eval/transformers in this environment already, so "
                                   "this should only fire if you're running this script somewhere lm_eval isn't installed.")


def load_correctness(path: str) -> Dict[str, bool]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def align(a: Dict[str, bool], b: Dict[str, bool]) -> Tuple[List[str], List[bool], List[bool]]:
    keys_a, keys_b = set(a), set(b)
    shared = sorted(keys_a & keys_b, key=lambda k: (len(k), k))
    only_a = keys_a - keys_b
    only_b = keys_b - keys_a
    if only_a or only_b:
        print(f"WARNING: {len(only_a)} doc_ids only in A, {len(only_b)} only in B -- these are dropped from "
              "the paired analysis. This should be 0 for two runs of the same task/limit; if it isn't, "
              "double check both runs used the same --limit and the same lm_eval task version.")
    if not shared:
        raise ValueError("No shared doc_ids between the two correctness files -- can't pair anything.")
    return shared, [a[k] for k in shared], [b[k] for k in shared]


def mcnemar_table(vals_a: List[bool], vals_b: List[bool]) -> Dict[str, int]:
    n11 = sum(1 for x, y in zip(vals_a, vals_b) if x and y)
    n10 = sum(1 for x, y in zip(vals_a, vals_b) if x and not y)
    n01 = sum(1 for x, y in zip(vals_a, vals_b) if not x and y)
    n00 = sum(1 for x, y in zip(vals_a, vals_b) if not x and not y)
    return {"both_correct": n11, "a_only_correct": n10, "b_only_correct": n01, "both_wrong": n00}


def mcnemar_exact(n10: int, n01: int) -> float:
    """Exact two-sided McNemar test: Binomial(n10+n01, 0.5), testing whether
    the discordant pairs split evenly between the two directions. Preferred
    over the chi-square approximation whenever n10+n01 is small (a common
    case here, since discordant pairs are rare when the accuracy delta is a
    couple percentage points on ~1273 docs)."""
    require_scipy()
    n = n10 + n01
    if n == 0:
        return 1.0
    k = min(n10, n01)
    return float(binomtest(k, n, p=0.5, alternative="two-sided").pvalue)


def mcnemar_chi_square(n10: int, n01: int) -> Tuple[float, float]:
    """Classic continuity-corrected McNemar chi-square, for cross-reference
    against the exact test above -- returns (statistic, p_value). Only a
    reasonable approximation when n10+n01 is not small (rule of thumb: >=25)."""
    require_scipy()
    n = n10 + n01
    if n == 0:
        return 0.0, 1.0
    stat = (abs(n10 - n01) - 1) ** 2 / n
    p = float(chi2.sf(stat, df=1))
    return float(stat), p


def paired_bootstrap_delta_ci(
    vals_a: List[bool], vals_b: List[bool], num_bootstrap: int, seed: int,
) -> Dict[str, float]:
    """Bootstrap CI on delta = acc(B) - acc(A), resampling doc INDICES (not
    each condition independently) so every resample preserves the pairing --
    this is what makes it a valid CI on the paired delta rather than a naive
    combination of two marginal CIs that ignores their correlation."""
    n = len(vals_a)
    rng = random.Random(seed)
    deltas = []
    for _ in range(num_bootstrap):
        idx = [rng.randrange(n) for _ in range(n)]
        acc_a = sum(vals_a[i] for i in idx) / n
        acc_b = sum(vals_b[i] for i in idx) / n
        deltas.append(acc_b - acc_a)
    deltas.sort()
    point_delta = sum(vals_b) / n - sum(vals_a) / n
    lo = deltas[int(0.025 * num_bootstrap)]
    hi = deltas[int(0.975 * num_bootstrap) - 1]
    return {
        "point_delta": point_delta,
        "ci95_lo": lo,
        "ci95_hi": hi,
        "num_bootstrap": num_bootstrap,
        "n_docs": n,
    }


def main() -> None:
    args = parse_args()
    require_scipy()
    correctness_a = load_correctness(args.correctness_a)
    correctness_b = load_correctness(args.correctness_b)
    doc_ids, vals_a, vals_b = align(correctness_a, correctness_b)

    table = mcnemar_table(vals_a, vals_b)
    n10, n01 = table["a_only_correct"], table["b_only_correct"]
    exact_p = mcnemar_exact(n10, n01)
    chi_stat, chi_p = mcnemar_chi_square(n10, n01)
    ci = paired_bootstrap_delta_ci(vals_a, vals_b, args.num_bootstrap, args.seed)

    acc_a = sum(vals_a) / len(vals_a)
    acc_b = sum(vals_b) / len(vals_b)

    summary = {
        "label_a": args.label_a,
        "label_b": args.label_b,
        "n_docs_paired": len(doc_ids),
        "acc_a": acc_a,
        "acc_b": acc_b,
        "mcnemar_table": table,
        "mcnemar_exact_p_value": exact_p,
        "mcnemar_chi_square_stat": chi_stat,
        "mcnemar_chi_square_p_value": chi_p,
        "chi_square_reliable": (n10 + n01) >= 25,
        "paired_bootstrap_delta_ci": ci,
    }

    print(json.dumps(summary, indent=2))
    print()
    print(f"{args.label_a}: acc={acc_a:.4f}   {args.label_b}: acc={acc_b:.4f}   delta={ci['point_delta']:+.4f}")
    print(f"McNemar table: both_correct={table['both_correct']} "
          f"{args.label_a}_only={table['a_only_correct']} {args.label_b}_only={table['b_only_correct']} "
          f"both_wrong={table['both_wrong']}")
    print(f"McNemar exact p-value: {exact_p:.4g}  (n10+n01={n10 + n01})")
    print(f"McNemar chi-square: stat={chi_stat:.3f} p={chi_p:.4g} "
          f"({'reliable' if summary['chi_square_reliable'] else 'UNRELIABLE -- n10+n01 < 25, trust the exact test'})")
    print(f"Paired bootstrap 95% CI on delta ({args.label_b} - {args.label_a}): "
          f"[{ci['ci95_lo']:+.4f}, {ci['ci95_hi']:+.4f}]  (n_bootstrap={ci['num_bootstrap']})")

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")
        print(f"\nWrote {args.output_json}")


if __name__ == "__main__":
    main()
