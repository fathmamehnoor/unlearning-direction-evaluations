"""Extract and select a refusal direction, correctly, using a generic
harmful/harmless prompt source (Arditi et al. 2024's own AdvBench-vs-Alpaca
setup).

Both selection algorithms are computed from the same candidate sweep and can
be produced in a single run:

* ``mean_diff``: Arditi et al. (2024) Appendix C.1's actual causal selection
  -- for each candidate direction, ablate it on held-out harmful prompts
  (bypass_score), add it on held-out harmless prompts (induce_score), and
  measure the KL divergence it introduces on harmless prompts (kl_score).
  Select the direction that minimizes bypass_score subject to
  induce_score > 0, kl_score < 0.1, and layer < 0.8L.

* ``cosmic``: Siu et al. (2025) COSMIC's concept-inversion cosine-similarity
  selection -- restricted to the low-similarity layers (bottom 10% by
  harmful/harmless cosine similarity), score each candidate by how much
  ablating it on harmful prompts makes their activations resemble natural
  harmless activations (S_comply), and how much adding it on harmless
  prompts makes their activations resemble natural harmful activations
  (S_refuse). Select argmax(S_refuse + S_comply) subject to the same KL and
  layer-fraction filters, plus S_refuse > 0.

Fixes vs. the earlier ``wmdp_bio_refusal_direction_eval.py``:

1. No more double-BOS/special-token corruption. Chat-templated prompts are
   tokenized once via ``apply_chat_template(tokenize=True)`` and padded with
   ``tokenizer.pad`` -- never re-tokenized with a plain ``tokenizer(text)``
   call, which is what silently duplicated the BOS token before.
2. Candidates are drawn from the full grid of the last 5 post-instruction
   token positions (Arditi Table 5 / 6, COSMIC Section 2.3) x every layer,
   not just the very last token. For Llama-3 8B specifically, Arditi's own
   appendix found position i*=-5 (not -1) to be optimal, which the old
   single-position search could never find.
3. ``mean_diff`` reimplements Arditi's actual causal algorithm (bypass /
   induce / KL scores) instead of a raw validation projection-gap proxy.
4. ``cosmic`` scoring restricts activations to the post-instruction
   positions (not averaged across the whole prompt), and applies the same
   KL-divergence / late-layer / induce-threshold filters used in the
   official COSMIC repo (wang-research-lab/COSMIC).

Left-padding is used throughout so that position index -1 is always the true
final token of every row regardless of prompt length -- this sidesteps the
need for any "find the last non-pad token" bookkeeping.

Usage:

    python scripts/extract_refusal_direction.py \\
      --model OPTML-Group/IDK-AP-WMDP-llama3-8b-instruct \\
      --model-label idk-ap \\
      --selection-method both \\
      --output-root outputs/direction_extraction/idk-ap_generic
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import math
import random
import re
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset

try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ModuleNotFoundError:
    torch = None
    F = None
    AutoModelForCausalLM = None
    AutoTokenizer = None


# ## Constants

ALPACA_DATASET = "tatsu-lab/alpaca"
ALPACA_SPLIT = "train"
ADVBENCH_URL = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"

PROMPT_COLUMN_CANDIDATES = ["goal", "Goal", "prompt", "Prompt", "behavior", "Behavior"]

HARDWARE_PROFILES = {
    "a100": {"batch_size": 8, "dtype": "bfloat16", "gpu_memory": "72GiB", "cpu_memory": "64GiB"},
    "t4x2": {"batch_size": 2, "dtype": "float16", "gpu_memory": "14GiB", "cpu_memory": "30GiB"},
}


# ## Args

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Hugging Face model id or local checkpoint path.")
    parser.add_argument("--model-label", default="", help="Optional label for output rows. Defaults to the model path basename.")
    parser.add_argument("--tokenizer", default="", help="Defaults to --model.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--selection-method", choices=["mean_diff", "cosmic", "both"], default="both")
    parser.add_argument("--hardware-profile", choices=["a100", "t4x2", "manual"], default="a100")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default=None)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--gpu-memory", default=None)
    parser.add_argument("--cpu-memory", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-input-tokens", type=int, default=512)
    parser.add_argument("--train-size-per-class", type=int, default=128, help="Arditi et al. default is 128.")
    parser.add_argument("--val-size-per-class", type=int, default=32, help="Arditi et al. default is 32.")
    parser.add_argument("--num-candidate-positions", type=int, default=5,
                         help="Number of trailing post-instruction token positions to search (Arditi default 5).")
    parser.add_argument("--candidate-layer-stride", type=int, default=1,
                         help="Only build candidates every Nth layer, to cut compute. 1 = every layer.")
    parser.add_argument("--cosmic-low-layer-frac", type=float, default=0.10)
    parser.add_argument("--cosmic-min-eval-layers", type=int, default=1)
    parser.add_argument("--cosmic-eval-position", type=int, default=-1,
                         help="Single post-instruction position COSMIC's concept-inversion score is computed at "
                              "(the 'first output token', COSMIC S3.3.2). Must be one of the --num-candidate-positions "
                              "trailing positions searched for candidates.")
    parser.add_argument("--late-layer-exclusion-frac", type=float, default=0.8,
                         help="Exclude candidates from layers >= this fraction of total depth (Arditi/COSMIC: 0.8).")
    parser.add_argument("--kl-threshold", type=float, default=0.1, help="Max allowed KL divergence on harmless val prompts.")
    parser.add_argument("--refusal-strings", default="I", help="Comma-separated strings whose first token marks refusal (Arditi Table 4: Llama-3 Instruct -> 'I').")
    parser.add_argument("--prompt-format", choices=["chat_template", "raw"], default="chat_template")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--skip-matched-control", action="store_true",
                         help="Skip computing the matched-construction control direction (split-half "
                              "difference-in-means over the harmless training pool, at the same "
                              "position/layer as the selected direction). This is a stronger null than an "
                              "isotropic random unit vector: same construction mechanism, same activation "
                              "statistics, zero genuine refusal signal. Computed by default (2 extra "
                              "forward passes over 64 prompts per selected method).")
    args, _ = parser.parse_known_args()
    if not args.tokenizer:
        args.tokenizer = args.model
    apply_hardware_profile(args)
    return args


def apply_hardware_profile(args: argparse.Namespace) -> None:
    if args.hardware_profile == "manual":
        if args.batch_size is None:
            args.batch_size = 2
        if args.dtype is None:
            args.dtype = "float16"
        if args.gpu_memory is None:
            args.gpu_memory = "14GiB"
        if args.cpu_memory is None:
            args.cpu_memory = "30GiB"
        return
    profile = HARDWARE_PROFILES[args.hardware_profile]
    for key, value in profile.items():
        if getattr(args, key) is None:
            setattr(args, key, value)


# ## Runtime helpers

def require_dependencies() -> None:
    missing = []
    if torch is None:
        missing.append("torch")
    if AutoModelForCausalLM is None or AutoTokenizer is None:
        missing.append("transformers")
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


def max_memory(args: argparse.Namespace) -> Optional[Dict[object, str]]:
    require_dependencies()
    if not args.device_map or not torch.cuda.is_available():
        return None
    memory = {idx: args.gpu_memory for idx in range(torch.cuda.device_count())}
    memory["cpu"] = args.cpu_memory
    return memory


def release_memory() -> None:
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return slug.strip("_") or "model"


def model_label(args: argparse.Namespace) -> str:
    return args.model_label or slugify(Path(str(args.model).rstrip("/")).name)


def normalize_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(obj), f, indent=2, ensure_ascii=False)
        f.write("\n")


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows)).to_csv(path, index=False)


def json_safe(value):
    if torch is not None and torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


# ## Prompt datasets

def infer_column(columns, candidates, label):
    lower_map = {str(col).lower(): str(col) for col in columns}
    for candidate in candidates:
        if candidate in columns:
            return candidate
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    raise ValueError(f"Could not infer {label} column from columns: {list(columns)}")


def load_advbench_prompts(n_prompts: int, seed: int) -> List[str]:
    try:
        with urllib.request.urlopen(ADVBENCH_URL, timeout=30) as response:
            raw = response.read().decode("utf-8")
        df = pd.read_csv(io.StringIO(raw))
        prompt_col = infer_column(df.columns, PROMPT_COLUMN_CANDIDATES, "AdvBench prompt")
        prompts = [normalize_text(v) for v in df[prompt_col].tolist()]
        prompts = [p for p in prompts if p]
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load AdvBench harmful prompts from {ADVBENCH_URL}: {exc}. No fallback prompt "
            "list is configured -- fix network access / the dataset source before retrying, rather than "
            "silently continuing on a different (and much smaller, less validated) prompt set."
        ) from exc
    rng = random.Random(seed)
    rng.shuffle(prompts)
    if len(prompts) < n_prompts:
        raise ValueError(f"Need {n_prompts} AdvBench prompts, only found {len(prompts)}.")
    return prompts[:n_prompts]


def load_alpaca_harmless_prompts(n_prompts: int, seed: int, exclude: Optional[Set[str]] = None) -> List[str]:
    try:
        dataset = load_dataset(ALPACA_DATASET, split=ALPACA_SPLIT)
        prompts = []
        for item in dataset:
            instruction = normalize_text(item.get("instruction", ""))
            input_text = normalize_text(item.get("input", ""))
            if instruction and not input_text:
                prompts.append(instruction)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load Alpaca harmless prompts ({ALPACA_DATASET}, split={ALPACA_SPLIT}): {exc}. No "
            "fallback prompt list is configured -- fix network access / the dataset source before "
            "retrying, rather than silently continuing on a different (and much smaller, less validated) "
            "prompt set."
        ) from exc
    if exclude:
        prompts = [p for p in prompts if p not in exclude]
    rng = random.Random(seed)
    rng.shuffle(prompts)
    if len(prompts) < n_prompts:
        raise ValueError(f"Need {n_prompts} Alpaca prompts (after excluding {len(exclude) if exclude else 0} "
                          f"already-used prompts), only found {len(prompts)}.")
    return prompts[:n_prompts]


def load_direction_prompts(args: argparse.Namespace, n_total: int) -> Tuple[List[str], List[str]]:
    return (
        load_advbench_prompts(n_total, args.seed),
        load_alpaca_harmless_prompts(n_total, args.seed),
    )


# ## Model / tokenizer

def load_tokenizer(args: argparse.Namespace):
    require_dependencies()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=args.trust_remote_code, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_model(args: argparse.Namespace):
    require_dependencies()
    kwargs = {
        "torch_dtype": dtype_from_name(args.dtype),
        "trust_remote_code": args.trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if args.device_map:
        kwargs["device_map"] = args.device_map
        memory = max_memory(args)
        if memory:
            kwargs["max_memory"] = memory
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return model


def get_input_device(model):
    return model.get_input_embeddings().weight.device


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


# ## Fixed tokenization: never re-tokenize an already chat-templated string.

def flatten_token_ids(result) -> List[int]:
    """Normalize the return value of apply_chat_template(tokenize=True, ...) /
    tokenizer(...) to a flat List[int] of token ids. The shape returned for a
    single conversation has not been stable across transformers/tokenizers
    versions: a flat List[int] (most versions), a BatchEncoding/dict with an
    "input_ids" key (sometimes nested one level as a batch-of-one), a tensor,
    or -- observed on a newer "tokenizers backend" transformers release -- a
    length-1 list wrapping a single tokenizers.Encoding object, whose `.ids`
    attribute holds the actual per-token ids. Silently assuming the old shape
    previously surfaced as a crash trying to tokenizer.decode() an Encoding
    object as if it were a single token id; unwrap defensively instead.
    """
    seen_types = []
    for _ in range(5):  # a handful of unwrap steps covers every shape seen so far
        seen_types.append(type(result).__name__)
        if isinstance(result, list) and result and all(isinstance(t, int) for t in result):
            return result
        if hasattr(result, "input_ids"):
            result = result.input_ids
            continue
        if isinstance(result, dict) and "input_ids" in result:
            result = result["input_ids"]
            continue
        if torch is not None and torch.is_tensor(result):
            result = result.tolist()
            continue
        if hasattr(result, "ids"):
            result = list(result.ids)
            continue
        if isinstance(result, (list, tuple)) and len(result) == 1:
            result = result[0]
            continue
        break
    raise TypeError(
        f"Could not normalize a tokenizer output to a flat List[int] of token ids after "
        f"unwrapping through types {seen_types}. The installed transformers/tokenizers "
        "version likely returns a shape not handled here -- inspect the raw object "
        "(print(type(result)), dir(result)) and extend flatten_token_ids()."
    )


def render_chat_ids(tokenizer, prompt: str, prompt_format: str) -> List[int]:
    if prompt_format == "raw" or not getattr(tokenizer, "chat_template", None):
        return flatten_token_ids(tokenizer(prompt, add_special_tokens=True))
    result = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
    )
    return flatten_token_ids(result)


def encode_batch(tokenizer, prompts: Sequence[str], args: argparse.Namespace, device):
    id_lists = []
    for prompt in prompts:
        ids = render_chat_ids(tokenizer, prompt, args.prompt_format)
        if len(ids) > args.max_input_tokens:
            ids = ids[-args.max_input_tokens:]
        if len(ids) < args.num_candidate_positions:
            raise ValueError(
                f"Rendered prompt has only {len(ids)} tokens, fewer than "
                f"--num-candidate-positions={args.num_candidate_positions}."
            )
        id_lists.append(ids)
    encoded = tokenizer.pad({"input_ids": id_lists}, return_tensors="pt", padding=True)
    return encoded.to(device)


def resolve_refusal_token_ids(tokenizer, refusal_strings: str) -> List[int]:
    ids = set()
    for s in refusal_strings.split(","):
        s = s.strip()
        if not s:
            continue
        encoded = tokenizer.encode(s, add_special_tokens=False)
        if encoded:
            ids.add(encoded[-1])
    if not ids:
        raise ValueError(f"Could not resolve any refusal token ids from: {refusal_strings!r}")
    return sorted(ids)


# ## Unified scored forward pass: intervention + activation collection + refusal metric + KL

def refusal_logit_metric_batch(logprobs: "torch.Tensor", refusal_mask: "torch.Tensor") -> "torch.Tensor":
    # logprobs: (batch, vocab); refusal_mask: (vocab,) bool
    log_p_refusal = torch.logsumexp(logprobs[:, refusal_mask], dim=-1)
    log_p_comply = torch.logsumexp(logprobs[:, ~refusal_mask], dim=-1)
    return log_p_refusal - log_p_comply


def kl_div_batch(baseline_logprobs: "torch.Tensor", current_logprobs: "torch.Tensor") -> "torch.Tensor":
    # KL(baseline || current), per row. (batch, vocab) -> (batch,)
    return (baseline_logprobs.exp() * (baseline_logprobs - current_logprobs)).sum(dim=-1)


def run_scored_pass(
    model,
    tokenizer,
    prompts: Sequence[str],
    args: argparse.Namespace,
    positions: Sequence[int],
    refusal_mask: "torch.Tensor",
    n_layers: int,
    direction: Optional["torch.Tensor"] = None,
    mode: str = "none",
    add_layer_idx: Optional[int] = None,
    scale: float = 1.0,
    baseline_logprobs: Optional[Sequence["torch.Tensor"]] = None,
) -> dict:
    """Runs `prompts` through `model` with an optional intervention.

    mode: "none" | "ablate" (every residual-stream write, all layers, all positions --
          block input + attention output + MLP output, per Arditi et al. 2024 Eq. 4 /
          Appendix E and COSMIC S3.1) | "add" (add_layer_idx block-input only, all
          positions, per Arditi et al. 2024 Eq. 3)

    Returns a dict with:
      position_layer_sum: (len(positions), n_layers, hidden) float64 running sum
      position_layer_count: int (number of examples summed)
      refusal_metric_sum: float
      kl_sum: float or None
      n_examples: int
      final_logprobs: list of per-example (vocab,) cpu tensors, in prompt order
    """
    require_dependencies()
    input_device = get_input_device(model)
    layers = get_decoder_layers(model)
    hidden_size = int(model.config.hidden_size)
    local_direction = direction.detach().float().cpu() if direction is not None else None

    position_layer_sum = torch.zeros(len(positions), n_layers, hidden_size, dtype=torch.float64)
    position_layer_count = 0
    refusal_metric_sum = 0.0
    kl_sum = 0.0 if baseline_logprobs is not None else None
    n_examples = 0
    final_logprobs_out: List["torch.Tensor"] = []

    def project_away(hidden: "torch.Tensor", dir_local: "torch.Tensor") -> "torch.Tensor":
        projection = hidden @ dir_local
        return hidden - projection.unsqueeze(-1) * dir_local

    def make_pre_hook(layer_idx: int):
        def hook(_module, inputs):
            nonlocal position_layer_sum, position_layer_count
            hidden = inputs[0]
            pos_vecs = hidden[:, positions, :].detach().float().cpu()
            position_layer_sum[:, layer_idx, :] += pos_vecs.sum(dim=0).double()
            if local_direction is None or mode == "none":
                return None
            dir_local = local_direction.to(device=hidden.device, dtype=hidden.dtype)
            if mode == "ablate":
                return (project_away(hidden, dir_local),) + inputs[1:]
            if mode == "add" and layer_idx == add_layer_idx:
                return (hidden + scale * dir_local,) + inputs[1:]
            return None
        return hook

    def make_output_hook():
        # Ablation must also zero the direction where the attention and MLP
        # submodules write their own contribution to the residual stream --
        # otherwise a component along `dir_local` written by this layer's
        # attention leaks, un-ablated, into this layer's own MLP before the
        # *next* layer's pre-hook removes it. Only relevant to "ablate" mode;
        # "add" is a single-point injection (Arditi et al. 2024 Eq. 3).
        def hook(_module, _inputs, output):
            if local_direction is None or mode != "ablate":
                return None
            if torch.is_tensor(output):
                dir_local = local_direction.to(device=output.device, dtype=output.dtype)
                return project_away(output, dir_local)
            if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
                dir_local = local_direction.to(device=output[0].device, dtype=output[0].dtype)
                return (project_away(output[0], dir_local),) + tuple(output[1:])
            return None
        return hook

    handles = []
    try:
        for idx, layer in enumerate(layers):
            handles.append(layer.register_forward_pre_hook(make_pre_hook(idx)))
            if mode == "ablate" and local_direction is not None:
                attn = attention_module(layer)
                if attn is None:
                    raise AttributeError(f"Layer {idx} has no recognizable attention module for full ablation coverage.")
                handles.append(attn.register_forward_hook(make_output_hook()))
                mlp = mlp_module(layer)
                if mlp is None:
                    raise AttributeError(f"Layer {idx} has no recognizable MLP module for full ablation coverage.")
                handles.append(mlp.register_forward_hook(make_output_hook()))
        with torch.inference_mode():
            for start in range(0, len(prompts), args.batch_size):
                batch_prompts = prompts[start:start + args.batch_size]
                encoded = encode_batch(tokenizer, batch_prompts, args, input_device)
                outputs = model(**encoded, use_cache=False)
                final_logits = outputs.logits[:, -1, :].float()
                final_logprobs = F.log_softmax(final_logits, dim=-1)
                metric = refusal_logit_metric_batch(final_logprobs, refusal_mask.to(final_logprobs.device))
                refusal_metric_sum += float(metric.sum().item())
                if baseline_logprobs is not None:
                    batch_baseline = torch.stack(baseline_logprobs[start:start + len(batch_prompts)]).to(final_logprobs.device)
                    kl = kl_div_batch(batch_baseline, final_logprobs)
                    kl_sum += float(kl.sum().item())
                final_logprobs_out.extend(list(final_logprobs.detach().cpu()))
                position_layer_count += len(batch_prompts)
                n_examples += len(batch_prompts)
                del encoded, outputs, final_logits, final_logprobs
                release_memory()
    finally:
        for handle in handles:
            handle.remove()

    return {
        "position_layer_mean": (position_layer_sum / max(1, position_layer_count)).float(),
        "refusal_metric_mean": refusal_metric_sum / max(1, n_examples),
        "kl_mean": (kl_sum / max(1, n_examples)) if kl_sum is not None else None,
        "n_examples": n_examples,
        "final_logprobs": final_logprobs_out,
    }


# ## Candidate construction and layer selection

def unit_vector(vector: "torch.Tensor") -> "torch.Tensor":
    vector = vector.detach().float().cpu().flatten()
    norm = vector.norm()
    return vector / norm.clamp_min(1e-12)


def split_prompts_in_half(prompts: Sequence[str], seed: int) -> Tuple[List[str], List[str]]:
    shuffled = list(prompts)
    random.Random(seed).shuffle(shuffled)
    midpoint = len(shuffled) // 2
    return shuffled[:midpoint], shuffled[midpoint:]


def compute_matched_control_direction(
    model,
    tokenizer,
    args: argparse.Namespace,
    already_used_harmless: Set[str],
    position: int,
    layer_idx: int,
    positions: Sequence[int],
    refusal_mask: "torch.Tensor",
    n_layers: int,
) -> Tuple["torch.Tensor", float, dict]:
    """A same-construction, content-free control direction: draw a fresh
    harmless prompt pool disjoint from every harmless prompt already used in
    real extraction (train + val -- not just the train split), split it into
    two random halves, and take their difference-in-means at the SAME
    (position, layer) the real direction was selected from. Both halves are
    drawn from the identical harmless distribution and share no prompts with
    the real direction's own harmless reference set, so this direction
    carries no genuine refusal/knowledge-suppression signal and no
    shared-sample idiosyncrasy -- only difference-in-means sampling noise --
    while still being a real, activation-consistent, data-driven direction
    (unlike an isotropic random Gaussian vector, which is nearly orthogonal
    to everything in a high-dimensional residual stream by concentration of
    measure). Ablating it tests: does merely ablating *any* difference-in-
    means direction built this way, at this same spot, move WMDP-Bio
    accuracy, even with no real semantic contrast behind it.
    """
    control_load_seed = args.seed + 777
    control_split_seed = args.seed + 778
    control_pool = load_alpaca_harmless_prompts(args.train_size_per_class, control_load_seed, exclude=already_used_harmless)
    half_a, half_b = split_prompts_in_half(control_pool, control_split_seed)
    mean_a = run_scored_pass(model, tokenizer, half_a, args, positions, refusal_mask, n_layers)["position_layer_mean"]
    mean_b = run_scored_pass(model, tokenizer, half_b, args, positions, refusal_mask, n_layers)["position_layer_mean"]
    pos_idx = positions.index(position)
    raw = mean_a[pos_idx, layer_idx] - mean_b[pos_idx, layer_idx]
    metadata = {
        "control_pool_size": len(control_pool),
        "control_load_seed": control_load_seed,
        "control_split_seed": control_split_seed,
        "control_half_a_size": len(half_a),
        "control_half_b_size": len(half_b),
        "control_disjoint_from_n_prompts": len(already_used_harmless),
    }
    return unit_vector(raw), float(raw.norm().item()), metadata


def build_candidates(
    harmful_train_means: "torch.Tensor",
    harmless_train_means: "torch.Tensor",
    positions: Sequence[int],
    args: argparse.Namespace,
) -> List[dict]:
    # harmful/harmless_train_means: (n_positions, n_layers, hidden)
    n_layers = harmful_train_means.shape[1]
    candidates = []
    for pos_idx, position in enumerate(positions):
        for layer_idx in range(0, n_layers, args.candidate_layer_stride):
            raw = harmful_train_means[pos_idx, layer_idx] - harmless_train_means[pos_idx, layer_idx]
            candidates.append({
                "condition": f"mean_diff_pos{position}_layer{layer_idx}",
                "position": position,
                "layer": layer_idx,
                "direction": unit_vector(raw),
                "raw_norm": float(raw.norm().item()),
            })
    return candidates


def layer_cosine_similarity(harmful_means: "torch.Tensor", harmless_means: "torch.Tensor", eval_pos_idx: int) -> List[dict]:
    # means: (n_positions, n_layers, hidden). COSMIC S3.4 scores the "first output
    # token" position only (the last post-instruction position), not an average
    # across the whole position grid used for candidate construction.
    n_layers = harmful_means.shape[1]
    rows = []
    for layer_idx in range(n_layers):
        h = harmful_means[eval_pos_idx, layer_idx, :]
        n = harmless_means[eval_pos_idx, layer_idx, :]
        sim = float(F.cosine_similarity(h, n, dim=0).item())
        rows.append({"layer": layer_idx, "class_cosine_similarity": sim})
    return rows


def select_low_similarity_layers(rows: Sequence[dict], args: argparse.Namespace) -> List[int]:
    sorted_rows = sorted(rows, key=lambda row: row["class_cosine_similarity"])
    n_keep = max(args.cosmic_min_eval_layers, int(math.ceil(len(sorted_rows) * args.cosmic_low_layer_frac)))
    return [int(row["layer"]) for row in sorted_rows[:n_keep]]


def cosine_of_layer_means(a: "torch.Tensor", b: "torch.Tensor", layers: Sequence[int], eval_pos_idx: int) -> float:
    # a, b: (n_positions, n_layers, hidden). COSMIC S3.3.2 computes activations at the
    # single "first output token" position (i=-1), concatenates the per-layer mean
    # vectors over L_low into one vector per condition, and takes a SINGLE cosine
    # similarity on that concatenation -- not a mean of per-layer cosines. The two are
    # different quantities (concatenated-cosine implicitly weights each layer's
    # contribution by that layer's activation norm via the shared denominator; a mean
    # of per-layer cosines weights every layer equally regardless of magnitude), and
    # can rank candidates differently.
    a_concat = torch.cat([a[eval_pos_idx, layer_idx, :] for layer_idx in layers], dim=0)
    b_concat = torch.cat([b[eval_pos_idx, layer_idx, :] for layer_idx in layers], dim=0)
    return float(F.cosine_similarity(a_concat, b_concat, dim=0).item())


# ## Main

def main() -> None:
    args = parse_args()
    require_dependencies()
    set_seed(args.seed)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "run_config.json", vars(args))
    label = model_label(args)

    n_total = args.train_size_per_class + args.val_size_per_class
    harmful_all, harmless_all = load_direction_prompts(args, n_total)
    harmful_train = harmful_all[:args.train_size_per_class]
    harmful_val = harmful_all[args.train_size_per_class:]
    harmless_train = harmless_all[:args.train_size_per_class]
    harmless_val = harmless_all[args.train_size_per_class:]
    write_csv(output_root / "direction_prompts.csv", [
        {"split": "train", "class": "harmful", "prompt": p} for p in harmful_train
    ] + [
        {"split": "val", "class": "harmful", "prompt": p} for p in harmful_val
    ] + [
        {"split": "train", "class": "harmless", "prompt": p} for p in harmless_train
    ] + [
        {"split": "val", "class": "harmless", "prompt": p} for p in harmless_val
    ])

    print(f"[{time.strftime('%H:%M:%S')}] loading tokenizer/model: {args.model}")
    tokenizer = load_tokenizer(args)
    model = load_model(args)
    n_layers = len(get_decoder_layers(model))
    positions = list(range(-args.num_candidate_positions, 0))

    refusal_token_ids = resolve_refusal_token_ids(tokenizer, args.refusal_strings)
    print(f"[{time.strftime('%H:%M:%S')}] refusal token ids: {refusal_token_ids} "
          f"({[tokenizer.decode([t]) for t in refusal_token_ids]!r})")
    vocab_size = int(model.get_output_embeddings().weight.shape[0])
    refusal_mask = torch.zeros(vocab_size, dtype=torch.bool)
    refusal_mask[refusal_token_ids] = True

    sample_ids = render_chat_ids(tokenizer, harmless_train[0], args.prompt_format)
    print(f"[{time.strftime('%H:%M:%S')}] sample rendered tail tokens (positions {positions}): "
          f"{[tokenizer.decode([t]) for t in sample_ids[-args.num_candidate_positions:]]!r}")

    # --- Train pass (no intervention): candidate generation + Low layer selection
    print(f"[{time.strftime('%H:%M:%S')}] collecting train activations (harmful)")
    harmful_train_res = run_scored_pass(model, tokenizer, harmful_train, args, positions, refusal_mask, n_layers)
    print(f"[{time.strftime('%H:%M:%S')}] collecting train activations (harmless)")
    harmless_train_res = run_scored_pass(model, tokenizer, harmless_train, args, positions, refusal_mask, n_layers)
    harmful_train_means = harmful_train_res["position_layer_mean"]
    harmless_train_means = harmless_train_res["position_layer_mean"]

    candidates = build_candidates(harmful_train_means, harmless_train_means, positions, args)
    print(f"[{time.strftime('%H:%M:%S')}] built {len(candidates)} candidate directions "
          f"({len(positions)} positions x {len(range(0, n_layers, args.candidate_layer_stride))} layers)")

    eval_pos_idx = positions.index(args.cosmic_eval_position)
    layer_sim_rows = layer_cosine_similarity(harmful_train_means, harmless_train_means, eval_pos_idx)
    write_csv(output_root / "layer_similarity.csv", layer_sim_rows)
    low_layers = select_low_similarity_layers(layer_sim_rows, args)
    print(f"[{time.strftime('%H:%M:%S')}] COSMIC low-similarity layers (Llow): {low_layers}")

    # --- Baseline val passes (no intervention)
    print(f"[{time.strftime('%H:%M:%S')}] baseline val pass (harmful)")
    baseline_harmful_val = run_scored_pass(model, tokenizer, harmful_val, args, positions, refusal_mask, n_layers)
    print(f"[{time.strftime('%H:%M:%S')}] baseline val pass (harmless)")
    baseline_harmless_val = run_scored_pass(model, tokenizer, harmless_val, args, positions, refusal_mask, n_layers)
    baseline_harmless_logprobs = baseline_harmless_val["final_logprobs"]

    late_layer_cutoff = args.late_layer_exclusion_frac * n_layers

    # --- Score every candidate
    rows = []
    for idx, candidate in enumerate(candidates, start=1):
        t0 = time.time()
        direction = candidate["direction"]
        layer_idx = candidate["layer"]

        ablate_harmful = run_scored_pass(
            model, tokenizer, harmful_val, args, positions, refusal_mask, n_layers,
            direction=direction, mode="ablate",
        )
        add_harmless = run_scored_pass(
            model, tokenizer, harmless_val, args, positions, refusal_mask, n_layers,
            direction=direction, mode="add", add_layer_idx=layer_idx, scale=candidate["raw_norm"],
        )
        ablate_harmless = run_scored_pass(
            model, tokenizer, harmless_val, args, positions, refusal_mask, n_layers,
            direction=direction, mode="ablate", baseline_logprobs=baseline_harmless_logprobs,
        )

        bypass_score = ablate_harmful["refusal_metric_mean"]
        induce_score = add_harmless["refusal_metric_mean"]
        kl_score = ablate_harmless["kl_mean"]

        s_comply = cosine_of_layer_means(baseline_harmless_val["position_layer_mean"], ablate_harmful["position_layer_mean"], low_layers, eval_pos_idx)
        s_refuse = cosine_of_layer_means(add_harmless["position_layer_mean"], baseline_harmful_val["position_layer_mean"], low_layers, eval_pos_idx)
        cosmic_score = s_refuse + s_comply

        passes_late_layer_filter = layer_idx < late_layer_cutoff
        passes_kl_filter = kl_score < args.kl_threshold
        passes_mean_diff_filter = passes_late_layer_filter and passes_kl_filter and induce_score > 0
        passes_cosmic_filter = passes_late_layer_filter and passes_kl_filter and s_refuse > 0

        rows.append({
            "condition": candidate["condition"],
            "position": candidate["position"],
            "layer": layer_idx,
            "raw_norm": candidate["raw_norm"],
            "bypass_score": bypass_score,
            "induce_score": induce_score,
            "kl_score": kl_score,
            "s_refuse": s_refuse,
            "s_comply": s_comply,
            "cosmic_score": cosmic_score,
            "passes_late_layer_filter": passes_late_layer_filter,
            "passes_kl_filter": passes_kl_filter,
            "passes_mean_diff_filter": passes_mean_diff_filter,
            "passes_cosmic_filter": passes_cosmic_filter,
        })
        if idx % 5 == 0 or idx == len(candidates):
            write_csv(output_root / "candidate_diagnostics_partial.csv", rows)
            elapsed = time.time() - t0
            print(f"[{time.strftime('%H:%M:%S')}] scored {idx}/{len(candidates)} candidates "
                  f"({candidate['condition']}, {elapsed:.1f}s/candidate)")

    write_csv(output_root / "candidate_diagnostics.csv", rows)

    def save_selected(condition_row: dict, method: str) -> None:
        by_condition = {c["condition"]: c for c in candidates}
        candidate = by_condition[condition_row["condition"]]
        artifact_path = output_root / f"direction_{method}.pt"
        torch.save({
            "condition": candidate["condition"],
            "direction": candidate["direction"],
            "position": candidate["position"],
            "layer": candidate["layer"],
            "selection_method": method,
            "prompt_source": "generic",
            "selection_row": condition_row,
            "model": args.model,
            "model_label": label,
            "hook_scope": "all_layers_all_tokens:block_input_pre_hook,self_attention_forward_output,mlp_forward_output",
            "cosmic_eval_position": args.cosmic_eval_position,
        }, artifact_path)
        write_json(output_root / f"direction_{method}_summary.json", {
            "selected_condition": candidate["condition"],
            "selection_method": method,
            "prompt_source": "generic",
            "selected_row": condition_row,
            "artifact": str(artifact_path),
        })
        print(f"[{time.strftime('%H:%M:%S')}] selected {candidate['condition']} via {method} -> {artifact_path}")

        if not args.skip_matched_control:
            print(f"[{time.strftime('%H:%M:%S')}] computing matched-construction control for {method} "
                  f"at position={candidate['position']} layer={candidate['layer']}")
            control_direction, control_raw_norm, control_metadata = compute_matched_control_direction(
                model, tokenizer, args, set(harmless_all),
                candidate["position"], candidate["layer"], positions, refusal_mask, n_layers,
            )
            control_artifact_path = output_root / f"direction_{method}_matched_control.pt"
            torch.save({
                "condition": f"matched_control_pos{candidate['position']}_layer{candidate['layer']}",
                "direction": control_direction,
                "position": candidate["position"],
                "layer": candidate["layer"],
                "raw_norm": control_raw_norm,
                "selection_method": method,
                "control_type": "matched_construction_split_half_harmless",
                "prompt_source": "generic",
                "matched_to_condition": candidate["condition"],
                "model": args.model,
                "model_label": label,
                "hook_scope": "all_layers_all_tokens:block_input_pre_hook,self_attention_forward_output,mlp_forward_output",
                **control_metadata,
            }, control_artifact_path)
            print(f"[{time.strftime('%H:%M:%S')}] saved matched-construction control -> {control_artifact_path} "
                  f"(disjoint from {control_metadata['control_disjoint_from_n_prompts']} already-used harmless "
                  f"prompts; load_seed={control_metadata['control_load_seed']} "
                  f"split_seed={control_metadata['control_split_seed']})")

    def filter_failure_report(rows: Sequence[dict], score_key: str) -> str:
        n_fail_layer = sum(1 for r in rows if not r["passes_late_layer_filter"])
        n_fail_kl = sum(1 for r in rows if not r["passes_kl_filter"])
        n_fail_score = sum(1 for r in rows if not (r[score_key] > 0))
        return (
            f"{len(rows)} candidates scored; "
            f"{n_fail_layer} failed the late-layer cutoff (layer >= {late_layer_cutoff:.1f}), "
            f"{n_fail_kl} failed the KL threshold (kl_score >= {args.kl_threshold}), "
            f"{n_fail_score} failed {score_key}>0. "
            "See candidate_diagnostics.csv (already written) for the full per-candidate breakdown."
        )

    saved_any = False

    if args.selection_method in ("mean_diff", "both"):
        mean_diff_candidates = [r for r in rows if r["passes_mean_diff_filter"]]
        if not mean_diff_candidates:
            print(
                "ERROR: no candidate passed the mean_diff filters (Arditi et al. 2024 Appendix C.1: "
                "induce_score>0, kl_score<kl_threshold, layer<late-layer cutoff). Refusing to fall back "
                "to an unfiltered minimum-bypass_score candidate -- that selects for whatever most "
                "suppresses the refusal token with no guarantee of a bounded effect on harmless-prompt "
                "behavior or the outcome being a semantic (rather than late-layer token-shortcut) "
                "direction, which is precisely the failure mode these filters exist to rule out. "
                f"{filter_failure_report(rows, 'induce_score')} "
                "No direction_mean_diff.pt will be saved for this run. Re-run with a looser "
                "--kl-threshold / --late-layer-exclusion-frac if that's the binding constraint, or "
                "investigate why induce_score never exceeds 0 -- e.g. this model may not reliably refuse "
                "using the token(s) in --refusal-strings, which would make this output-token-dependent "
                "metric unreliable here even if a real refusal direction exists. That's exactly the "
                "failure mode COSMIC's activation-based selection (scored independently below, from the "
                "same candidate sweep) is designed to be robust to."
            )
        else:
            selected_row = min(mean_diff_candidates, key=lambda r: r["bypass_score"])
            save_selected(selected_row, "mean_diff")
            saved_any = True

    if args.selection_method in ("cosmic", "both"):
        cosmic_candidates = [r for r in rows if r["passes_cosmic_filter"]]
        if not cosmic_candidates:
            print(
                "ERROR: no candidate passed the cosmic filters (COSMIC S3.3.2: s_refuse>0, "
                "kl_score<kl_threshold, layer<late-layer cutoff). Refusing to fall back to an unfiltered "
                "maximum-cosmic_score candidate for the same reason as the mean_diff path above. "
                f"{filter_failure_report(rows, 's_refuse')} "
                "No direction_cosmic.pt will be saved for this run. Re-run with a looser --kl-threshold / "
                "--late-layer-exclusion-frac if that's the binding constraint, or investigate why "
                "s_refuse never exceeds 0."
            )
        else:
            selected_row = max(cosmic_candidates, key=lambda r: r["cosmic_score"])
            save_selected(selected_row, "cosmic")
            saved_any = True

    if not saved_any:
        raise RuntimeError(
            "Neither requested selection method saved a direction for this run -- see the ERROR "
            "messages above for each method's filter-failure breakdown. candidate_diagnostics.csv has "
            "already been written with every candidate's raw scores for manual inspection."
        )

    del model
    del tokenizer
    release_memory()
    print(f"[{time.strftime('%H:%M:%S')}] wrote outputs to {output_root}")


if __name__ == "__main__":
    main()
