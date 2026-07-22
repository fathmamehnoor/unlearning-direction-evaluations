"""Evaluate WMDP-Bio accuracy under directional ablation using lm-evaluation-harness.

This script replaces the custom forced-choice scoring in
`wmdp_bio_refusal_direction_eval.py` (which under-measured accuracy due to a
prompt-formatting bug) with `lm_eval`'s own wmdp_bio task, which is already
validated against known-good numbers (see `updated tests/wmdp-bio-eval/*.json`).

It takes an already-selected refusal direction (a `.pt` artifact, e.g. from
`extract_refusal_direction.py`'s `direction_mean_diff.pt` / `direction_cosmic.pt`,
or from the official COSMIC pipeline's `direction.pt`) and runs `lm_eval` three
ways on the same loaded model:

  1. baseline                    -- no hook
  2. selected_direction_ablation -- directional ablation with the supplied direction
  3. random_direction_ablation_i -- directional ablation with N random unit
                                     directions, as a control

Pass `--skip-baseline` and/or `--num-random-controls 0` when calling this
script multiple times for the same model (e.g. once per prompt-source x
selection-method direction) to avoid recomputing the baseline / random
controls redundantly -- run it once in full, then with `--skip-baseline
--num-random-controls 0` for the remaining directions on that model.

Directional ablation is applied via forward-pre-hooks on every decoder layer
(all token positions), matching Arditi et al. (2024) Eq. 4 / Appendix E:
projecting a direction out at every layer's input is mathematically
equivalent to projecting it out after every attention/MLP contribution too.

Usage:

    python scripts/wmdp_bio_lm_eval_ablation.py \
      --model meta-llama/Meta-Llama-3-8B-Instruct \
      --direction-path outputs/wmdp_bio_cosmic_eval/wmdp_selected_controller.pt \
      --output-root outputs/wmdp_bio_lm_eval_ablation \
      --apply-chat-template

Or with a direction produced by the official COSMIC repo
(wang-research-lab/COSMIC, `pipeline/runs/{model_alias}/direction.pt`):

    python scripts/wmdp_bio_lm_eval_ablation.py \
      --model meta-llama/Meta-Llama-3-8B-Instruct \
      --direction-path /path/to/COSMIC/pipeline/runs/{model_alias}/direction.pt \
      --direction-key direction \
      --output-root outputs/wmdp_bio_lm_eval_ablation_cosmic_official
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ModuleNotFoundError:
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None

try:
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
except ModuleNotFoundError:
    simple_evaluate = None
    HFLM = None


DEFAULT_MODEL = ""
DEFAULT_TASKS = "wmdp_bio"
DEFAULT_OUTPUT_ROOT = "outputs/wmdp_bio_lm_eval_ablation"

HARDWARE_PROFILES = {
    "a100": {"batch_size": "auto", "dtype": "bfloat16"},
    "t4x2": {"batch_size": "auto:4", "dtype": "float16"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL, required=not bool(DEFAULT_MODEL), help="Hugging Face model id or local checkpoint path.")
    parser.add_argument("--tokenizer", default="", help="Defaults to --model.")
    parser.add_argument("--model-label", default="", help="Optional label for output rows. Defaults to the model path basename.")
    parser.add_argument("--direction-path", required=True, help="Path to a .pt artifact containing the selected refusal direction.")
    parser.add_argument("--direction-key", default="", help="Explicit dict key to read the direction from, if auto-detection is ambiguous.")
    parser.add_argument("--matched-control-direction-path", default="",
                         help="Optional path to a matched-construction control direction .pt artifact "
                              "(e.g. extract_refusal_direction.py's direction_{method}_matched_control.pt: "
                              "a split-half difference-in-means null at the same position/layer as the "
                              "selected direction, carrying no genuine refusal signal). If given, adds a "
                              "matched_control_ablation condition alongside the isotropic random-unit "
                              "controls -- a stronger null than random vectors, since it's built by the "
                              "same construction mechanism as the real direction.")
    parser.add_argument("--tasks", default=DEFAULT_TASKS, help="Comma-separated lm_eval task names.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--hardware-profile", choices=["a100", "t4x2", "manual"], default="a100")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default=None)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--batch-size", default=None, help="lm_eval batch_size, e.g. 'auto', 'auto:4', or an int.")
    parser.add_argument("--apply-chat-template", dest="apply_chat_template", action="store_true", default=True)
    parser.add_argument("--no-chat-template", dest="apply_chat_template", action="store_false")
    parser.add_argument("--num-random-controls", type=int, default=8,
                         help="Random unit-direction ablation controls. The 'selected direction beats "
                              "random directions by more than noise' claim needs enough of these to "
                              "characterize the control distribution's mean/std, not just a point or two -- "
                              "3 is too few; 5-10 is the practical range.")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip the no-hook baseline condition (e.g. if already run for this model).")
    parser.add_argument("--limit", type=int, default=None, help="Optional lm_eval sample limit, useful for smoke tests.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true")
    args, _ = parser.parse_known_args()
    if not args.model:
        parser.error("--model is required")
    if not args.tokenizer:
        args.tokenizer = args.model
    apply_hardware_profile(args)
    return args


def apply_hardware_profile(args: argparse.Namespace) -> None:
    if args.hardware_profile == "manual":
        if args.batch_size is None:
            args.batch_size = "auto:2"
        if args.dtype is None:
            args.dtype = "float16"
        return
    profile = HARDWARE_PROFILES[args.hardware_profile]
    if args.batch_size is None:
        args.batch_size = profile["batch_size"]
    if args.dtype is None:
        args.dtype = profile["dtype"]


def require_dependencies() -> None:
    missing = []
    if torch is None:
        missing.append("torch")
    if AutoModelForCausalLM is None or AutoTokenizer is None:
        missing.append("transformers")
    if simple_evaluate is None or HFLM is None:
        missing.append("lm_eval")
    if missing:
        raise ModuleNotFoundError(f"Missing required dependencies: {', '.join(sorted(set(missing)))}")


def set_seed(seed: int) -> None:
    require_dependencies()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dtype_from_name(name: str):
    require_dependencies()
    if name == "auto":
        return torch.float16 if torch.cuda.is_available() else torch.float32
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def release_memory() -> None:
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def slugify(value: str) -> str:
    import re
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return slug.strip("_") or "model"


def model_label(args: argparse.Namespace) -> str:
    return args.model_label or slugify(Path(str(args.model).rstrip("/")).name)


def load_model_and_tokenizer(args: argparse.Namespace):
    require_dependencies()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=args.trust_remote_code, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype_from_name(args.dtype),
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
    )
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return model, tokenizer


def get_decoder_layers(model):
    base = getattr(model, "model", model)
    if hasattr(base, "layers"):
        return base.layers
    if hasattr(base, "decoder") and hasattr(base.decoder, "layers"):
        return base.decoder.layers
    if hasattr(base, "transformer") and hasattr(base.transformer, "h"):
        return base.transformer.h
    raise AttributeError("Could not locate decoder layers on this causal LM.")


def attention_module(layer):
    for name in ("self_attn", "attention", "attn"):
        if hasattr(layer, name):
            return getattr(layer, name)
    return None


def mlp_module(layer):
    for name in ("mlp", "feed_forward", "ffn"):
        if hasattr(layer, name):
            return getattr(layer, name)
    return None


def unit_vector(vector: "torch.Tensor") -> "torch.Tensor":
    vector = vector.detach().float().cpu().flatten()
    norm = vector.norm()
    if norm <= 1e-8:
        raise ValueError("Direction has near-zero norm.")
    return (vector / norm).contiguous()


# Common key names used across this repo's own controller artifacts
# (wmdp_bio_refusal_direction_eval.py, cosmic_refusal_controller.py) and
# plausible names for the official COSMIC repo's direction.pt. If none of
# these match, pass --direction-key explicitly.
CANDIDATE_DIRECTION_KEYS = ["direction", "basis", "r", "vector", "refusal_direction", "steering_vector"]


def direction_from_payload(payload, hidden_size: int, explicit_key: str) -> "torch.Tensor":
    if torch.is_tensor(payload):
        direction = payload
    elif isinstance(payload, dict):
        keys_to_try = [explicit_key] if explicit_key else CANDIDATE_DIRECTION_KEYS
        direction = None
        for key in keys_to_try:
            if key and key in payload and torch.is_tensor(payload[key]):
                direction = payload[key]
                break
        if direction is None:
            raise ValueError(
                f"Could not find a tensor direction in artifact dict. Available keys: "
                f"{sorted(payload.keys())}. Pass --direction-key to pick one explicitly."
            )
    else:
        raise ValueError(f"Unsupported direction artifact type: {type(payload)}")

    direction = direction.float().cpu()
    if direction.ndim > 1:
        # e.g. a (1, hidden_size) or (hidden_size, 1) basis matrix from a rank-1 controller
        if direction.numel() == hidden_size:
            direction = direction.flatten()
        elif direction.shape[0] == hidden_size:
            direction = direction[:, 0]
        elif direction.shape[-1] == hidden_size:
            direction = direction[0]
        else:
            raise ValueError(f"Direction shape {tuple(direction.shape)} cannot be interpreted for hidden size {hidden_size}.")
    if direction.numel() != hidden_size:
        raise ValueError(f"Direction size {direction.numel()} does not match model hidden size {hidden_size}.")
    return unit_vector(direction)


DIRECTION_METADATA_KEYS = ["condition", "selection_method", "prompt_source", "position", "layer",
                           "control_type", "matched_to_condition"]


def load_direction(path: Path, hidden_size: int, explicit_key: str) -> "Tuple[torch.Tensor, dict]":
    if not path.exists():
        raise FileNotFoundError(f"Direction artifact not found: {path}")
    payload = torch.load(path, map_location="cpu")
    direction = direction_from_payload(payload, hidden_size, explicit_key)
    metadata = {}
    if isinstance(payload, dict):
        metadata = {k: payload[k] for k in DIRECTION_METADATA_KEYS if k in payload}
    return direction, metadata


def random_unit_direction(hidden_size: int, seed: int) -> "torch.Tensor":
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return unit_vector(torch.randn(hidden_size, generator=generator))


@contextmanager
def all_layer_all_token_ablation(model, direction: Optional["torch.Tensor"]):
    """Directional ablation at every point that writes to the residual stream --
    block input, self-attention output, and MLP output -- for every decoder
    layer, all token positions (Arditi et al. 2024 Eq. 4 / Appendix E; COSMIC
    S3.1). A pre-hook on the block input alone is NOT sufficient: within a
    layer, attention can write a component along `direction` into the
    residual stream that reaches this layer's own MLP un-ablated, only being
    removed at the *next* layer's pre-hook -- so the MLP's contribution is
    still computed under a nonzero projection onto `direction`. Ablating the
    attention and MLP submodule outputs directly closes that gap. No-op
    context if direction is None.
    """
    if direction is None:
        yield
        return
    handles = []

    def project_away(hidden: "torch.Tensor") -> "torch.Tensor":
        local_direction = direction.to(device=hidden.device, dtype=hidden.dtype)
        projection = hidden @ local_direction
        return hidden - projection.unsqueeze(-1) * local_direction

    def pre_hook(_module, inputs):
        return (project_away(inputs[0]),) + tuple(inputs[1:])

    def output_hook(_module, _inputs, output):
        if torch.is_tensor(output):
            return project_away(output)
        if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
            return (project_away(output[0]),) + tuple(output[1:])
        return output

    try:
        for layer_idx, layer in enumerate(get_decoder_layers(model)):
            handles.append(layer.register_forward_pre_hook(pre_hook))
            attn = attention_module(layer)
            if attn is None:
                raise AttributeError(f"Layer {layer_idx} has no recognizable attention module for full ablation coverage.")
            handles.append(attn.register_forward_hook(output_hook))
            mlp = mlp_module(layer)
            if mlp is None:
                raise AttributeError(f"Layer {layer_idx} has no recognizable MLP module for full ablation coverage.")
            handles.append(mlp.register_forward_hook(output_hook))
        yield
    finally:
        for handle in handles:
            handle.remove()


def run_lm_eval(model, tokenizer, args: argparse.Namespace) -> Tuple[dict, dict]:
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size)
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    results = simple_evaluate(
        model=lm,
        tasks=tasks,
        apply_chat_template=args.apply_chat_template,
        limit=args.limit,
        log_samples=True,
        random_seed=args.seed,
        numpy_random_seed=args.seed,
        torch_random_seed=args.seed,
    )
    return results["results"], results.get("samples", {})


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


# ## Per-doc correctness, for paired comparisons across conditions (McNemar's
# test / paired CIs on the accuracy delta) that a bare aggregate accuracy +
# random-control z-score can't give you -- those treat each condition's
# accuracy as an independent sample, when in fact every condition here is
# scored on the exact same 1273 wmdp_bio questions. Logic mirrors
# wmdp_sft_recovery/eval_recovery_lm_eval.py's gold_index/choice_loglikelihoods
# (duplicated rather than imported -- these are separate top-level pipelines).

def gold_index(sample: dict) -> Optional[int]:
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
    resps = sample.get("filtered_resps")
    if resps is None:
        resps = sample.get("resps")
    if not isinstance(resps, list) or not resps:
        return None
    lls: List[float] = []
    for entry in resps:
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
    Keyed by doc_id (not position) so two conditions' correctness arrays can
    be paired up correctly even if either run's sample list order ever
    diverges -- falls back to positional index with a loud warning if a
    sample has no doc_id."""
    out: Dict[str, bool] = {}
    for i, sample in enumerate(samples):
        doc_id = sample.get("doc_id")
        key = str(doc_id) if doc_id is not None else str(i)
        if doc_id is None:
            print(f"WARNING: sample {i} has no doc_id; pairing this doc by position, which is only "
                  "correct if every condition's docs are iterated in the exact same order.")
        lls = choice_loglikelihoods(sample)
        gold = gold_index(sample)
        if lls is None or gold is None or gold >= len(lls):
            continue
        out[key] = (max(range(len(lls)), key=lambda idx: lls[idx]) == gold)
    return out


ACC_KEY = "acc,none"


def summarize_random_controls(summary_rows: List[dict]) -> List[dict]:
    """Aggregate random_direction_ablation_* rows per task into a mean/std
    control distribution, and compare selected_direction_ablation's accuracy
    against it. This is what makes "selected direction beats random controls
    by more than noise" checkable directly from the output file, instead of
    requiring the reader to eyeball N separate unaggregated rows. Produces
    nothing for a task if this run computed zero random controls (e.g. the
    --skip-baseline --num-random-controls 0 runs that reuse an earlier run's
    controls) -- in that case, compare against the aggregate row from the run
    that did compute them.
    """
    by_task: Dict[str, dict] = {}
    for row in summary_rows:
        bucket = by_task.setdefault(row["task"], {"random": [], "selected": None})
        if row["condition"].startswith("random_direction_ablation_"):
            bucket["random"].append(row)
        elif row["condition"] == "selected_direction_ablation":
            bucket["selected"] = row

    aggregate_rows = []
    for task, bucket in by_task.items():
        accs = [float(r[ACC_KEY]) for r in bucket["random"] if ACC_KEY in r]
        if not accs:
            continue
        accs_arr = np.array(accs, dtype=float)
        random_mean = float(accs_arr.mean())
        random_std = float(accs_arr.std(ddof=1)) if accs_arr.size > 1 else 0.0
        agg_row = {
            "condition": "random_direction_ablation_summary",
            "task": task,
            "n_random_controls": int(accs_arr.size),
            "random_acc_mean": random_mean,
            "random_acc_std": random_std,
            "random_acc_values": accs,
        }
        selected = bucket["selected"]
        if selected is not None and ACC_KEY in selected:
            selected_acc = float(selected[ACC_KEY])
            agg_row["selected_acc"] = selected_acc
            agg_row["selected_minus_random_mean"] = selected_acc - random_mean
            agg_row["selected_control_z"] = (
                (selected_acc - random_mean) / random_std if random_std > 0 else None
            )
        aggregate_rows.append(agg_row)
    return aggregate_rows


def main() -> None:
    args = parse_args()
    require_dependencies()
    set_seed(args.seed)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "run_config.json", vars(args))

    label = model_label(args)
    print(f"[{time.strftime('%H:%M:%S')}] loading model: {args.model}")
    model, tokenizer = load_model_and_tokenizer(args)
    hidden_size = int(model.config.hidden_size)

    selected_direction, direction_metadata = load_direction(Path(args.direction_path), hidden_size, args.direction_key)
    if direction_metadata:
        print(f"[{time.strftime('%H:%M:%S')}] direction metadata: {direction_metadata}")
    random_directions = [
        random_unit_direction(hidden_size, args.seed + 1000 + idx)
        for idx in range(args.num_random_controls)
    ]

    matched_control_direction = None
    matched_control_metadata: dict = {}
    if args.matched_control_direction_path:
        matched_control_direction, matched_control_metadata = load_direction(
            Path(args.matched_control_direction_path), hidden_size, ""
        )
        print(f"[{time.strftime('%H:%M:%S')}] matched-control direction metadata: {matched_control_metadata}")

    conditions: List[Tuple[str, Optional["torch.Tensor"]]] = []
    if not args.skip_baseline:
        conditions.append(("baseline", None))
    conditions.append(("selected_direction_ablation", selected_direction))
    if matched_control_direction is not None:
        conditions.append(("matched_control_ablation", matched_control_direction))
    for idx, direction in enumerate(random_directions):
        conditions.append((f"random_direction_ablation_{idx}", direction))

    all_results = {}
    summary_rows = []
    for condition, direction in conditions:
        print(f"[{time.strftime('%H:%M:%S')}] running lm_eval for condition={condition}")
        with all_layer_all_token_ablation(model, direction):
            task_results, task_samples = run_lm_eval(model, tokenizer, args)
        all_results[condition] = task_results
        for task_name, samples in task_samples.items():
            correctness = per_doc_correctness(samples)
            write_json(output_root / "per_doc_correctness" / task_name / f"{condition}.json", correctness)
        for task, metrics in task_results.items():
            row = {"model": args.model, "model_label": label, "condition": condition, "task": task}
            if condition == "selected_direction_ablation":
                row.update({f"direction_{k}": v for k, v in direction_metadata.items()})
            if condition == "matched_control_ablation":
                row.update({f"direction_{k}": v for k, v in matched_control_metadata.items()})
            row.update({k: v for k, v in metrics.items() if not isinstance(v, (dict, list))})
            if not any("stderr" in k for k in metrics):
                print(f"WARNING: lm_eval returned no stderr metric for task={task} condition={condition}; "
                      f"metric keys were: {sorted(metrics.keys())}. The mean/std across random controls "
                      "(computed below) is the fallback measure of noise for this task.")
            summary_rows.append(row)
        write_json(output_root / "wmdp_bio_lm_eval_results.json", all_results)
        write_json(output_root / "wmdp_bio_lm_eval_summary.json", summary_rows)
        release_memory()

    control_summary_rows = summarize_random_controls(summary_rows)
    if control_summary_rows:
        summary_rows = summary_rows + control_summary_rows
        write_json(output_root / "wmdp_bio_lm_eval_summary.json", summary_rows)

    print(f"[{time.strftime('%H:%M:%S')}] summary:")
    for row in summary_rows:
        if row["condition"] == "random_direction_ablation_summary":
            continue
        acc = row.get(ACC_KEY)
        stderr = next((v for k, v in row.items() if "stderr" in k), None)
        print(f"  {row['condition']:<32} {row['task']:<16} acc={acc} stderr={stderr}")
    for row in control_summary_rows:
        z = row.get("selected_control_z")
        z_str = f"{z:.2f}" if isinstance(z, (int, float)) else "n/a (no selected_direction_ablation row or zero-std controls in this run)"
        print(f"  [{row['task']}] random controls (n={row['n_random_controls']}): "
              f"acc mean={row['random_acc_mean']:.4f} std={row['random_acc_std']:.4f} | "
              f"selected acc={row.get('selected_acc')} | selected_control_z={z_str}")
    print(f"[{time.strftime('%H:%M:%S')}] wrote outputs to {output_root}")

    del model
    del tokenizer
    release_memory()


if __name__ == "__main__":
    main()
