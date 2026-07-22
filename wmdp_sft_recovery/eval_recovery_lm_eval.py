#!/usr/bin/env python
"""Evaluate WMDP-bio recovery, MMLU utility, and GSM8K uptake with lm_eval.

This is the *measurement half* of the unrelated-SFT recovery experiment. It
runs one model (a base/unlearned checkpoint, optionally with a GSM8K LoRA
adapter on top) through lm-evaluation-harness and appends a single row holding
all three signals so the reading is unambiguous from the first row:

  wmdp_bio   -- the RECOVERY signal. Argmax accuracy AND, separately, the mean
                probability mass on the correct option. Probability rising ahead
                of accuracy is the cleanest signature of knowledge surfacing
                internally before it flips the answer -- suppression, not
                removal. That is the sharpest thing this arm can show and it
                costs nothing extra, so it is always recorded.
  mmlu       -- the UTILITY GATE. Bio up with MMLU flat is real recovery. Bio
                and MMLU moving together is SFT churn, not hidden knowledge.
  gsm8k      -- the FINE-TUNE-TOOK check. If bio stays flat, a risen GSM8K
                score rules out "nothing happened" as the reason.

Every recovery number must be read next to the utility number, always. That is
why all three are emitted on one CSV row per checkpoint.

Usage (no adapter -- the no-SFT baseline for a model):
    python eval_recovery_lm_eval.py \
      --model-name OPTML-Group/ILU-RMU-WMDP-llama3-8b-instruct \
      --model-label ilu_rmu --unlearning-method ILU-RMU --arm unlearned \
      --checkpoint-step 0 --notes unlearned_no_sft_baseline

Usage (with a GSM8K LoRA checkpoint on top of the same base):
    python eval_recovery_lm_eval.py \
      --model-name OPTML-Group/ILU-RMU-WMDP-llama3-8b-instruct \
      --adapter-path results/.../checkpoint-examples-1000 \
      --model-label ilu_rmu --unlearning-method ILU-RMU --arm unlearned \
      --checkpoint-step 1000 --notes unrelated_gsm8k_sft
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

try:
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
except ModuleNotFoundError:  # deferred so --help works without the GPU env
    simple_evaluate = None
    HFLM = None


RESULT_COLUMNS = [
    "timestamp_utc",
    "arm",
    "model_name",
    "model_label",
    "unlearning_method",
    "adapter_path",
    "num_sft_examples",
    "checkpoint_step",
    "eval_precision",
    "wmdp_bio_acc",
    "wmdp_bio_acc_stderr",
    "wmdp_bio_correct_option_prob",
    "wmdp_bio_n",
    "prob_check",
    "mmlu_acc",
    "mmlu_acc_stderr",
    "gsm8k_acc",
    "gsm8k_acc_stderr",
    "notes",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_slug(value: str) -> str:
    import re

    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return slug.strip("_") or "model"


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
        f.write("\n")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-name", required=True, help="Base/unlearned model repo/path (the adapter's base).")
    parser.add_argument("--adapter-path", default=None, help="Optional GSM8K LoRA checkpoint to load on top.")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--model-label", default=None)
    parser.add_argument("--unlearning-method", default="none")
    parser.add_argument("--arm", default="unlearned", choices=["unlearned", "full_knowledge_control"])

    parser.add_argument("--tasks", default="wmdp_bio,mmlu,gsm8k", help="Comma-separated lm_eval tasks.")
    parser.add_argument(
        "--prob-tasks",
        default="wmdp_bio",
        help="Comma-separated multiple-choice tasks to compute correct-option probability for.",
    )
    parser.add_argument("--num-fewshot", type=int, default=None, help="Override lm_eval's per-task default shots.")
    parser.add_argument("--limit", type=int, default=None, help="Global per-task doc limit (smoke tests only).")
    parser.add_argument(
        "--task-limits",
        default=None,
        help="Per-task doc limits as 'task:N' comma-separated, e.g. 'mmlu:20'. Overrides "
        "--limit for those tasks; others stay at --limit (or full). For a group like mmlu, "
        "N is applied PER subject. Never limit wmdp_bio -- it is the recovery signal.",
    )

    parser.add_argument("--checkpoint-step", type=int, default=0, help="Examples-seen for this checkpoint; 0 = no-SFT baseline.")
    parser.add_argument("--num-sft-examples", type=int, default=None, help="Defaults to --checkpoint-step.")
    parser.add_argument(
        "--strict-prob-check",
        action="store_true",
        help="Exit non-zero if the recomputed argmax accuracy disagrees with lm_eval's own "
        "reported acc (i.e. the correct-option-probability parsing is wrong). Default: warn loudly.",
    )

    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--load-in-4bit", action="store_true", help="Evaluate on a 4-bit base (matches the QLoRA base).")
    parser.add_argument("--batch-size", default="auto", help="lm_eval batch size, e.g. 'auto', 'auto:4', or an int.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--apply-chat-template", dest="apply_chat_template", action="store_true", default=True)
    parser.add_argument("--no-chat-template", dest="apply_chat_template", action="store_false")

    parser.add_argument("--results-csv", type=Path, default=Path("results/wmdp_sft_recovery/recovery_results.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/wmdp_sft_recovery/eval"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--notes", default="")
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        print("Ignoring unknown launcher arguments:", unknown)
    return args


def require_lm_eval() -> None:
    if simple_evaluate is None or HFLM is None:
        raise ModuleNotFoundError(
            "lm_eval is not installed in this environment. Install lm-evaluation-harness "
            "(pip install lm_eval) on the GPU box before running this script."
        )


def build_model(args: argparse.Namespace):
    kwargs = dict(
        pretrained=args.model_name,
        tokenizer=args.tokenizer or args.model_name,
        dtype=args.dtype,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        batch_size=args.batch_size,
    )
    if args.adapter_path:
        kwargs["peft"] = args.adapter_path
    if args.load_in_4bit:
        kwargs["load_in_4bit"] = True
    return HFLM(**kwargs)


def eval_precision_tag(args: argparse.Namespace) -> str:
    """A stamp recording the exact precision this row was evaluated at.

    Precision MUST be identical across a model's baseline row and its SFT rows,
    because the reported recovery is baseline-minus-SFT and any precision
    difference between those two endpoints lands straight in that delta. This
    tag is written to every CSV row so drift is auditable: all rows of one model
    must share it. aggregate_table.py flags a model whose rows disagree."""
    return args.dtype + ("+4bit" if args.load_in_4bit else "")


def parse_task_limits(spec: Optional[str]) -> Dict[str, Optional[int]]:
    """Parse 'mmlu:20,foo:50' into {'mmlu': 20, 'foo': 50}."""
    out: Dict[str, Optional[int]] = {}
    if not spec:
        return out
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"--task-limits entry '{part}' must be 'task:N'.")
        task, value = part.rsplit(":", 1)
        out[task.strip()] = int(value)
    return out


def get_metric(results: dict, task: str, prefixes: Sequence[str]):
    """Fetch the first available metric for `task` matching any of `prefixes`.

    lm_eval keys look like 'acc,none' or 'exact_match,strict-match'. Groups like
    mmlu report their aggregate under the group name. Returns (value, key)."""
    task_metrics = results.get(task)
    if not task_metrics:
        return None, None
    for prefix in prefixes:
        for key, value in task_metrics.items():
            if key == prefix or key.startswith(prefix + ","):
                if isinstance(value, (int, float)):
                    return float(value), key
    return None, None


def softmax(logls: List[float]) -> List[float]:
    m = max(logls)
    exps = [math.exp(x - m) for x in logls]
    total = sum(exps)
    return [e / total for e in exps]


def gold_index(sample: dict) -> Optional[int]:
    """Best-effort gold-option index for a multiple-choice logged sample."""
    target = sample.get("target")
    if isinstance(target, bool):
        target = int(target)
    if isinstance(target, int):
        return target
    if isinstance(target, str):
        stripped = target.strip()
        if stripped.isdigit():
            return int(stripped)
        if len(stripped) == 1 and stripped.upper() in "ABCDEFGH":
            return ord(stripped.upper()) - ord("A")
    doc = sample.get("doc") or {}
    for key in ("answer", "label", "gold"):
        val = doc.get(key)
        if isinstance(val, int):
            return val
        if isinstance(val, str) and len(val) == 1 and val.upper() in "ABCDEFGH":
            return ord(val.upper()) - ord("A")
    return None


def choice_loglikelihoods(sample: dict) -> Optional[List[float]]:
    """Pull the per-choice loglikelihoods out of a logged MC sample."""
    resps = sample.get("filtered_resps")
    if resps is None:
        resps = sample.get("resps")
    if not isinstance(resps, list) or not resps:
        return None
    lls: List[float] = []
    for entry in resps:
        # entry is typically (loglikelihood, is_greedy) or a nested [[ll, greedy]]
        cur = entry
        while isinstance(cur, (list, tuple)) and cur and isinstance(cur[0], (list, tuple)):
            cur = cur[0]
        if isinstance(cur, (list, tuple)):
            cur = cur[0]
        try:
            lls.append(float(cur))
        except (TypeError, ValueError):
            return None
    return lls


def per_doc_correctness(samples: List[dict]) -> Dict[str, bool]:
    """Maps a stable doc_id (string) -> whether the argmax choice matches gold.

    Persisted per run so that a baseline row and its SFT-checkpoint rows (same
    task, same eval set) can be paired up by doc_id afterwards -- for a paired
    McNemar test or a paired bootstrap CI on the recovery delta, neither of
    which an aggregate accuracy + stderr alone can give you. Keyed by doc_id
    rather than position so pairing is correct even if list order ever
    diverges between two runs of the same task."""
    out: Dict[str, bool] = {}
    for i, sample in enumerate(samples):
        doc_id = sample.get("doc_id")
        key = str(doc_id) if doc_id is not None else str(i)
        if doc_id is None:
            print(f"WARNING: sample {i} has no doc_id; pairing this doc by position, which is only "
                  "correct if every run's docs are iterated in the exact same order.")
        lls = choice_loglikelihoods(sample)
        gold = gold_index(sample)
        if lls is None or gold is None or gold >= len(lls):
            continue
        out[key] = (max(range(len(lls)), key=lambda idx: lls[idx]) == gold)
    return out


def correct_option_probability(samples: List[dict]) -> Optional[dict]:
    """Mean softmax probability mass placed on the gold option across docs."""
    probs, argmax_hits, n = [], 0, 0
    for sample in samples:
        lls = choice_loglikelihoods(sample)
        gold = gold_index(sample)
        if lls is None or gold is None or gold >= len(lls):
            continue
        p = softmax(lls)
        probs.append(p[gold])
        if max(range(len(lls)), key=lambda i: lls[i]) == gold:
            argmax_hits += 1
        n += 1
    if n == 0:
        return None
    return {
        "mean_correct_option_prob": sum(probs) / n,
        "argmax_accuracy_recomputed": argmax_hits / n,
        "n_scored": n,
    }


def validate_prob_against_acc(prob_info: dict, metrics: dict, strict: bool) -> dict:
    """Cross-check the correct-option-probability parsing against lm_eval itself.

    The probability is derived by hand-unwrapping each MC sample's per-choice
    loglikelihoods -- a shape that is version-dependent in lm_eval and produces
    plausible-looking numbers even when parsed wrong. But argmax over the SAME
    loglikelihoods must reproduce lm_eval's own reported `acc` for that task. If
    the recomputed argmax matches, we are reading the right loglikelihoods and
    the probability is trustworthy; if it does not, filtered_resps parsing is
    wrong (or the sample format changed) and the probability column is garbage.

    This is a free assertion -- argmax accuracy is already computed inside
    correct_option_probability -- and it is the difference between a metric you
    can put weight on and one a reviewer can dismiss. Warns loudly on mismatch;
    exits non-zero when strict."""
    status = "ok"
    checks = {}
    for task, info in prob_info.items():
        recomputed = info.get("argmax_accuracy_recomputed")
        reported, _ = get_metric(metrics, task, ["acc"])
        n = max(int(info.get("n_scored", 0)), 1)
        # Allow at most ~one doc's worth of tie-break divergence (argmax on
        # exactly-equal loglikelihoods can break ties differently), plus a hair
        # for float noise. Anything larger means a real parsing mismatch.
        tol = max(1.5 / n, 5e-3)
        if reported is None or recomputed is None:
            checks[task] = {"reported_acc": reported, "recomputed_acc": recomputed, "status": "unverifiable"}
            status = status if status != "MISMATCH" else status
            continue
        diff = abs(recomputed - reported)
        ok = diff <= tol
        checks[task] = {
            "reported_acc": reported,
            "recomputed_acc": recomputed,
            "abs_diff": diff,
            "tolerance": tol,
            "status": "ok" if ok else "MISMATCH",
        }
        if not ok:
            status = "MISMATCH"
            banner = "=" * 72
            print(
                f"\n{banner}\n"
                f"PROBABILITY SELF-CHECK FAILED for task '{task}'.\n"
                f"  recomputed argmax acc = {recomputed:.4f}\n"
                f"  lm_eval reported acc  = {reported:.4f}\n"
                f"  |diff| = {diff:.4f} > tol = {tol:.4f}\n"
                f"The per-choice loglikelihood parsing (choice_loglikelihoods/gold_index)\n"
                f"is out of sync with lm_eval's sample format -- the '{task}' correct-option\n"
                f"probability column is NOT trustworthy for this run.\n{banner}\n",
                file=sys.stderr,
            )
    if status == "MISMATCH" and strict:
        raise SystemExit("Aborting: probability self-check failed (--strict-prob-check).")
    return {"status": status, "per_task": checks}


def append_row(csv_path: Path, row: dict, columns: Sequence[str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    fieldnames = list(columns)
    for key in row:
        if key not in fieldnames:
            fieldnames.append(key)
    # Preserve a stable header if the file already exists.
    if file_exists:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            existing = next(csv.reader(f), None)
        if existing:
            fieldnames = existing + [k for k in fieldnames if k not in existing]
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def main() -> None:
    args = parse_args()
    require_lm_eval()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    model_label = args.model_label or safe_slug(args.adapter_path or args.model_name)
    num_sft = args.num_sft_examples if args.num_sft_examples is not None else args.checkpoint_step
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    prob_tasks = {t.strip() for t in args.prob_tasks.split(",") if t.strip()}

    run_dir = args.output_dir / safe_slug(f"{model_label}_step{args.checkpoint_step}")
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "eval_config.json", vars(args))

    # Bucket tasks by their effective doc limit (per-task overrides win over the
    # global --limit) and run each bucket, reusing the one loaded model. This is
    # how mmlu can be capped (e.g. 20/subject) while wmdp_bio and gsm8k stay
    # full -- wmdp_bio is the recovery signal and must never be subsampled.
    task_limits = parse_task_limits(args.task_limits)
    buckets: Dict[Optional[int], List[str]] = {}
    for task in tasks:
        eff = task_limits.get(task, args.limit)
        buckets.setdefault(eff, []).append(task)

    lm = build_model(args)
    metrics: dict = {}
    samples: dict = {}
    run_configs: dict = {}
    for lim, bucket_tasks in buckets.items():
        res = simple_evaluate(
            model=lm,
            tasks=bucket_tasks,
            num_fewshot=args.num_fewshot,
            limit=lim,
            log_samples=True,
            apply_chat_template=args.apply_chat_template,
            random_seed=args.seed,
            numpy_random_seed=args.seed,
            torch_random_seed=args.seed,
            fewshot_random_seed=args.seed,
        )
        metrics.update(res["results"])
        samples.update(res.get("samples", {}))
        run_configs.update(res.get("configs", {}))

    # Recovery signal + utility gate + fine-tune-took check.
    wmdp_acc, _ = get_metric(metrics, "wmdp_bio", ["acc"])
    wmdp_stderr, _ = get_metric(metrics, "wmdp_bio", ["acc_stderr"])
    mmlu_acc, _ = get_metric(metrics, "mmlu", ["acc"])
    mmlu_stderr, _ = get_metric(metrics, "mmlu", ["acc_stderr"])
    # gsm8k default reports exact_match under strict-match / flexible-extract.
    gsm8k_acc, gsm8k_key = get_metric(metrics, "gsm8k", ["exact_match", "acc"])
    gsm8k_stderr, _ = get_metric(metrics, "gsm8k", ["exact_match_stderr", "acc_stderr"])

    prob_info = {}
    for task in prob_tasks:
        task_samples = samples.get(task)
        if task_samples:
            info = correct_option_probability(task_samples)
            if info:
                prob_info[task] = info
    wmdp_prob = prob_info.get("wmdp_bio", {}).get("mean_correct_option_prob")
    wmdp_n = prob_info.get("wmdp_bio", {}).get("n_scored")

    # Free self-check: recomputed argmax must reproduce lm_eval's own acc, else
    # the probability parsing is wrong and the prob column is untrustworthy.
    prob_check = validate_prob_against_acc(prob_info, metrics, args.strict_prob_check)

    # Per-doc correctness, for paired McNemar / paired CI analysis across this
    # model's baseline vs SFT-checkpoint rows later (see analyze_paired_recovery.py).
    for task_name, task_samples in samples.items():
        write_json(run_dir / "per_doc_correctness" / f"{task_name}.json", per_doc_correctness(task_samples))

    row = {
        "timestamp_utc": utc_now(),
        "arm": args.arm,
        "model_name": args.model_name,
        "model_label": model_label,
        "unlearning_method": args.unlearning_method,
        "adapter_path": args.adapter_path or "",
        "num_sft_examples": num_sft,
        "checkpoint_step": args.checkpoint_step,
        "eval_precision": eval_precision_tag(args),
        "wmdp_bio_acc": wmdp_acc,
        "wmdp_bio_acc_stderr": wmdp_stderr,
        "wmdp_bio_correct_option_prob": wmdp_prob,
        "wmdp_bio_n": wmdp_n,
        "prob_check": prob_check["status"],
        "mmlu_acc": mmlu_acc,
        "mmlu_acc_stderr": mmlu_stderr,
        "gsm8k_acc": gsm8k_acc,
        "gsm8k_acc_stderr": gsm8k_stderr,
        "notes": args.notes,
    }
    append_row(args.results_csv, row, RESULT_COLUMNS)

    # Full artifacts for the record (samples are large; keep them per-run).
    write_json(run_dir / "lm_eval_results.json", {"results": metrics, "configs": run_configs})
    write_json(run_dir / "correct_option_probability.json", {"prob_info": prob_info, "prob_check": prob_check})
    write_json(run_dir / "result_row.json", row)
    print(json.dumps(
        {"row": row, "gsm8k_metric_key": gsm8k_key, "prob_info": prob_info, "prob_check": prob_check},
        indent=2, default=str,
    ))


if __name__ == "__main__":
    main()
