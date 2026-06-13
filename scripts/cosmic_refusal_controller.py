"""Select a COSMIC-style refusal/IDK controller from paired TOFU prompts.

The selector builds candidate directions and subspaces from positive
modified-IDK prompts versus negative plain-QA prompts, scores them with
intervention-based internal activation similarity on low-similarity layers,
and can optionally apply behavior-first validation before saving a selected
``cosmic_selected_controller.pt`` artifact.
"""

import argparse
import math
import gc
import random
import re
import sys
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ModuleNotFoundError:
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None


DEFAULT_MODEL = ""
DEFAULT_PAIRED_CSV = "outputs/paired_idk_plain_prompts.csv"
DEFAULT_OUTPUT_ROOT = "outputs/cosmic_refusal_controller"
COSMIC_EVAL_LAYER_SIMILARITY_CSV = "cosmic_eval_layer_similarity.csv"
COSMIC_DIRECTION_CANDIDATES_CSV = "cosmic_direction_candidates.csv"
COSMIC_SELECTION_METRICS_CSV = "cosmic_selection_metrics.csv"
COSMIC_CONDITION_SPECS_CSV = "cosmic_condition_specs.csv"
COSMIC_BEHAVIOR_SELECTION_METRICS_CSV = "cosmic_behavior_selection_metrics.csv"
COSMIC_BEHAVIOR_SELECTION_GENERATIONS_CSV = "cosmic_behavior_selection_generations_side_by_side.csv"
COSMIC_BEHAVIOR_SELECTION_DECISION_CSV = "cosmic_behavior_selection_decision.csv"
COSMIC_BEHAVIOR_VALIDATION_METRICS_CSV = "cosmic_behavior_validation_metrics.csv"
COSMIC_GENERATIONS_SIDE_BY_SIDE_CSV = "cosmic_generations_side_by_side.csv"
COSMIC_DIAGNOSTIC_DECISION_CSV = "cosmic_diagnostic_decision.csv"
COSMIC_SELECTED_CONTROLLER_PT = "cosmic_selected_controller.pt"
HARDWARE_PROFILES = {
    "a100": {
        "batch_size": 8,
        "dtype": "bfloat16",
        "max_memory_gb_per_gpu": 72.0,
        "cpu_memory_gb": 64.0,
    },
    "t4x2": {
        "batch_size": 4,
        "dtype": "float16",
        "max_memory_gb_per_gpu": 14.0,
        "cpu_memory_gb": 28.0,
    },
}

REQUIRED_PAIRED_COLUMNS = [
    "source_index",
    "sample_id",
    "question",
    "positive_prompt_variant",
    "positive_prompt",
    "positive_generation",
    "positive_refusal_family",
    "negative_prompt",
    "negative_generation",
    "negative_prompt_variant",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, required=not bool(DEFAULT_MODEL), help="Hugging Face model id or local checkpoint path.")
    parser.add_argument("--paired-csv", default=DEFAULT_PAIRED_CSV)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--hardware-profile", default="a100", choices=["a100", "t4x2", "manual"])
    parser.add_argument("--dtype", default=None, choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-memory-gb-per-gpu", type=float, default=None)
    parser.add_argument("--cpu-memory-gb", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-frac", type=float, default=0.60)
    parser.add_argument("--val-frac", type=float, default=0.20)
    parser.add_argument("--post-instruction-positions", default="-5,-4,-3,-2,-1")
    parser.add_argument("--top-k-values", default="3,5,8")
    parser.add_argument("--same-layer-pca-ranks", default="3,5,8")
    parser.add_argument("--cosmic-max-scored-conditions", type=int, default=96)
    parser.add_argument("--cosmic-max-control-conditions-per-type", type=int, default=16)
    parser.add_argument("--cosmic-low-layer-frac", type=float, default=0.10)
    parser.add_argument("--cosmic-min-eval-layers", type=int, default=1)
    parser.add_argument("--cosmic-eval-position", type=int, default=-1)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--allow-truncated-prompts", action="store_true")
    parser.add_argument("--prompt-format", default="chat_template", choices=["chat_template", "raw", "auto"])
    parser.add_argument("--cosmic-selection-csv", default="", help="CSV with cosmic_score rows. Defaults to output_root/cosmic_selection_metrics.csv.")
    parser.add_argument("--selection-mode", default="behavior_first", choices=["cosmic", "behavior_first"])
    parser.add_argument("--behavior-selection-samples", type=int, default=16)
    parser.add_argument("--behavior-selection-max-candidates", type=int, default=10)
    parser.add_argument("--behavior-selection-ranks", default="1,3,5,8")
    parser.add_argument("--behavior-selection-ablation-scales", default="0.5,1.0,1.5,2.0")
    parser.add_argument("--behavior-selection-addition-scales", default="1.0")
    parser.add_argument("--behavior-selection-require-internal-directionality", action="store_true")
    parser.add_argument("--behavior-min-positive-ablation-effect", type=float, default=0.0)
    parser.add_argument("--behavior-min-negative-addition-effect", type=float, default=0.0)
    parser.add_argument("--behavior-max-plain-ablation-refusal-delta", type=float, default=0.0)
    parser.add_argument("--behavior-max-plain-ablation-degenerate-delta", type=float, default=0.0)
    parser.add_argument("--behavior-validation-samples", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--addition-scale", type=float, default=1.0)
    parser.add_argument("--skip-selected-artifact", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    args, _ = parser.parse_known_args()
    if not args.model:
        parser.error("--model is required")
    apply_hardware_profile(args)
    return args


def apply_hardware_profile(args: argparse.Namespace) -> None:
    if args.hardware_profile == "manual":
        if args.batch_size is None:
            args.batch_size = 4
        if args.dtype is None:
            args.dtype = "float16"
        if args.max_memory_gb_per_gpu is None:
            args.max_memory_gb_per_gpu = 14.0
        if args.cpu_memory_gb is None:
            args.cpu_memory_gb = 28.0
        return

    profile = HARDWARE_PROFILES[args.hardware_profile]
    for key, value in profile.items():
        if getattr(args, key) is None:
            setattr(args, key, value)


# ## Runtime Helpers

def require_model_dependencies() -> None:
    missing = []
    if torch is None:
        missing.append("torch")
    if AutoModelForCausalLM is None or AutoTokenizer is None:
        missing.append("transformers")
    if missing:
        names = ", ".join(sorted(set(missing)))
        raise ModuleNotFoundError(f"Missing required model-loading dependencies: {names}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    require_model_dependencies()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dtype_from_name(name: str):
    require_model_dependencies()
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def get_max_memory(args: argparse.Namespace) -> Optional[Dict]:
    require_model_dependencies()
    if args.device_map != "auto" or not torch.cuda.is_available():
        return None
    max_memory = {idx: f"{args.max_memory_gb_per_gpu}GiB" for idx in range(torch.cuda.device_count())}
    max_memory["cpu"] = f"{args.cpu_memory_gb}GiB"
    return max_memory


def normalize_device(device) -> Optional["torch.device"]:
    require_model_dependencies()
    if device in (None, "cpu", "disk"):
        return None
    if isinstance(device, int):
        return torch.device(f"cuda:{device}")
    if isinstance(device, str):
        return torch.device(device)
    if isinstance(device, torch.device) and device.type not in ("cpu", "meta"):
        return device
    return None


def model_input_device(model) -> "torch.device":
    require_model_dependencies()
    if hasattr(model, "hf_device_map") and model.hf_device_map:
        for key in ["model.embed_tokens", "embed_tokens", "transformer.wte"]:
            device = normalize_device(model.hf_device_map.get(key))
            if device is not None:
                return device
        for device in model.hf_device_map.values():
            normalized = normalize_device(device)
            if normalized is not None:
                return normalized
    embeddings = model.get_input_embeddings()
    if embeddings is not None:
        return next(embeddings.parameters()).device
    return next(model.parameters()).device


def release_memory() -> None:
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def load_paired_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Paired CSV not found: {path}")
    df = pd.read_csv(path)
    missing = sorted(set(REQUIRED_PAIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Paired CSV is missing required columns: {missing}")
    if df.empty:
        raise ValueError(f"Paired CSV has no rows: {path}")
    return df


def load_model_and_tokenizer(model_name: str, args: argparse.Namespace):
    require_model_dependencies()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype_from_name(args.dtype),
        device_map=args.device_map,
        max_memory=get_max_memory(args),
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer


def print_paired_csv_summary(df: pd.DataFrame) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] paired rows: {len(df)}")
    print(f"[{time.strftime('%H:%M:%S')}] unique sources: {df['source_index'].nunique()}")
    print(f"[{time.strftime('%H:%M:%S')}] positive prompt variants: {df['positive_prompt_variant'].value_counts().to_dict()}")
    print(f"[{time.strftime('%H:%M:%S')}] negative prompt variants: {df['negative_prompt_variant'].value_counts().to_dict()}")


# ## Data Split

def validate_split_fractions(train_frac: float, val_frac: float) -> None:
    if train_frac <= 0 or val_frac <= 0:
        raise ValueError("--train-frac and --val-frac must both be positive.")
    if train_frac + val_frac >= 1.0:
        raise ValueError("--train-frac + --val-frac must be less than 1.0 so heldout examples remain.")


def split_source_indices(df: pd.DataFrame, args: argparse.Namespace) -> Dict[str, list]:
    validate_split_fractions(args.train_frac, args.val_frac)
    source_indices = sorted(int(source_index) for source_index in df["source_index"].drop_duplicates().tolist())
    rng = random.Random(args.seed)
    rng.shuffle(source_indices)
    n_sources = len(source_indices)
    train_end = int(round(n_sources * args.train_frac))
    val_end = int(round(n_sources * (args.train_frac + args.val_frac)))
    splits = {
        "train": source_indices[:train_end],
        "val": source_indices[train_end:val_end],
        "heldout": source_indices[val_end:],
    }
    if not splits["train"] or not splits["val"] or not splits["heldout"]:
        raise ValueError(
            "Source split produced an empty split. Adjust --train-frac/--val-frac "
            f"or provide more paired sources. Counts: { {name: len(values) for name, values in splits.items()} }"
        )
    return splits


def paired_records_for_sources(df: pd.DataFrame, source_indices, split_name: str) -> list:
    source_set = set(int(source_index) for source_index in source_indices)
    subset = df[df["source_index"].astype(int).isin(source_set)].copy()
    subset = subset.sort_values(["source_index", "positive_prompt_variant"]).reset_index(drop=True)
    records = []
    for idx, row in subset.iterrows():
        records.append({
            "record_id": f"{split_name}_{int(row['source_index'])}_{idx}",
            "split": split_name,
            "source_index": int(row["source_index"]),
            "sample_id": str(row["sample_id"]),
            "positive_prompt_variant": str(row["positive_prompt_variant"]),
            "positive_prompt": str(row["positive_prompt"]),
            "negative_prompt": str(row["negative_prompt"]),
            "question": str(row["question"]),
        })
    return records


def split_summary_rows(df: pd.DataFrame, splits: Dict[str, list]) -> list:
    rows = []
    for split_name, source_indices in splits.items():
        source_set = set(int(source_index) for source_index in source_indices)
        subset = df[df["source_index"].astype(int).isin(source_set)].copy()
        rows.append({
            "split": split_name,
            "unique_sources": len(source_indices),
            "paired_rows": len(subset),
            "positive_prompt_variant_counts": subset["positive_prompt_variant"].value_counts().to_dict(),
            "negative_prompt_variant_counts": subset["negative_prompt_variant"].value_counts().to_dict(),
        })
    return rows


def write_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


# ## Candidate Extraction

def parse_int_list(spec: str) -> List[int]:
    return [int(piece.strip()) for piece in spec.split(",") if piece.strip()]


def parse_float_list(spec: str) -> List[float]:
    return [float(piece.strip()) for piece in spec.split(",") if piece.strip()]


def looks_chat_formatted(prompt: str) -> bool:
    markers = ["<|start_header_id|>", "<|im_start|>", "[INST]", "<s>[INST]", "### Assistant:"]
    return any(marker in str(prompt) for marker in markers)


def maybe_apply_chat_template(tokenizer, prompts: Sequence[str], prompt_format: str) -> List[str]:
    prompts = [str(prompt) for prompt in prompts]
    if prompt_format == "raw":
        return prompts
    if not hasattr(tokenizer, "apply_chat_template") or tokenizer.chat_template is None:
        if prompt_format == "chat_template":
            raise ValueError("Tokenizer has no chat_template; use --prompt-format raw.")
        return prompts
    formatted = []
    for prompt in prompts:
        if prompt_format == "auto" and looks_chat_formatted(prompt):
            formatted.append(prompt)
        else:
            formatted.append(tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            ))
    return formatted


def encode_batch(
    tokenizer,
    prompts: Sequence[str],
    device,
    max_input_tokens: Optional[int],
    allow_truncated_prompts: bool,
    context: str,
    prompt_format: str,
):
    prompts = maybe_apply_chat_template(tokenizer, prompts, prompt_format)
    tokenized = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=max_input_tokens is not None,
        max_length=max_input_tokens,
    )
    if max_input_tokens is not None and not allow_truncated_prompts:
        lengths = [len(ids) for ids in tokenizer(prompts, padding=False, truncation=False)["input_ids"]]
        truncated = [(idx, length) for idx, length in enumerate(lengths) if length > max_input_tokens]
        if truncated:
            examples = ", ".join(f"batch_index={idx}:tokens={length}" for idx, length in truncated[:5])
            raise ValueError(
                f"{context} exceeded --max-input-tokens={max_input_tokens}; truncated examples would be: {examples}. "
                "Increase --max-input-tokens or pass --allow-truncated-prompts."
            )
    return tokenized.to(device)


def get_decoder_layers(model):
    base = getattr(model, "model", model)
    if hasattr(base, "layers"):
        return base.layers
    if hasattr(base, "decoder") and hasattr(base.decoder, "layers"):
        return base.decoder.layers
    if hasattr(base, "transformer") and hasattr(base.transformer, "h"):
        return base.transformer.h
    raise AttributeError("Could not locate decoder layers on this causal LM.")


def unit_vector(vector):
    require_model_dependencies()
    norm = vector.norm()
    if norm <= 1e-8:
        return torch.zeros_like(vector)
    return vector / norm


@torch.inference_mode() if torch is not None else (lambda fn: fn)
def collect_resid_pre_by_layer_position(
    model,
    tokenizer,
    prompts: Sequence[str],
    positions: Sequence[int],
    batch_size: int,
    max_input_tokens: Optional[int],
    allow_truncated_prompts: bool,
    prompt_format: str,
    intervention_factory=None,
):
    require_model_dependencies()
    device = model_input_device(model)
    layers = get_decoder_layers(model)
    chunks = []
    for start in range(0, len(prompts), batch_size):
        prompt_batch = list(prompts[start:start + batch_size])
        encoded = encode_batch(
            tokenizer,
            prompt_batch,
            device,
            max_input_tokens,
            allow_truncated_prompts,
            "activation extraction prompts",
            prompt_format,
        )
        with (intervention_factory() if intervention_factory is not None else nullcontext()):
            attention_mask = encoded["attention_mask"]
            if tokenizer.padding_side == "left":
                last_nonpad = attention_mask.shape[1] - 1 - torch.flip(attention_mask, dims=[1]).long().argmax(dim=1)
            else:
                last_nonpad = attention_mask.long().sum(dim=1) - 1
            batch_layer_states = [None for _ in range(len(layers))]
            handles = []

            def make_hook(layer_idx: int):
                def hook_fn(_module, inputs):
                    hidden = inputs[0]
                    seq_len = hidden.shape[1]
                    resolved_rows = []
                    for position in positions:
                        if position < 0:
                            idx = last_nonpad.to(hidden.device) + position + 1
                        else:
                            idx = torch.full_like(last_nonpad, position).to(hidden.device)
                        idx = idx.clamp(0, seq_len - 1)
                        resolved_rows.append(hidden[torch.arange(hidden.shape[0], device=hidden.device), idx, :])
                    batch_layer_states[layer_idx] = torch.stack(resolved_rows, dim=1).detach().float().cpu()
                    return None

                return hook_fn

            for layer_idx, layer in enumerate(layers):
                handles.append(layer.register_forward_pre_hook(make_hook(layer_idx)))
            try:
                model(**encoded, use_cache=False)
            finally:
                for handle in handles:
                    handle.remove()
        if any(state is None for state in batch_layer_states):
            missing = [idx for idx, state in enumerate(batch_layer_states) if state is None]
            raise RuntimeError(f"Missing resid_pre activations for layers: {missing}")
        chunks.append(torch.stack(batch_layer_states, dim=1))
        del encoded, batch_layer_states
        release_memory()
    return torch.cat(chunks, dim=0)


def extract_direction_candidates(positive_acts, negative_acts, positions: Sequence[int]) -> List[dict]:
    if positive_acts.shape != negative_acts.shape:
        raise ValueError(f"Positive and negative activation shapes differ: {positive_acts.shape} vs {negative_acts.shape}")
    candidates = []
    n_layers = int(positive_acts.shape[1])
    for layer_idx in range(n_layers):
        for pos_idx, source_position in enumerate(positions):
            pos_layer = positive_acts[:, layer_idx, pos_idx, :]
            neg_layer = negative_acts[:, layer_idx, pos_idx, :]
            direction = pos_layer.mean(dim=0) - neg_layer.mean(dim=0)
            unit = unit_vector(direction)
            pos_proj = pos_layer.float() @ unit
            neg_proj = neg_layer.float() @ unit
            pooled = torch.sqrt(0.5 * (pos_proj.var(unbiased=False) + neg_proj.var(unbiased=False))).item()
            cohens_d = float(((pos_proj.mean() - neg_proj.mean()) / max(pooled, 1e-8)).item())
            candidates.append({
                "candidate_key": f"layer{layer_idx}_pos{source_position}",
                "layer_idx": layer_idx,
                "source_position": int(source_position),
                "direction_norm": float(direction.norm().item()),
                "train_projection_gap": float((pos_proj.mean() - neg_proj.mean()).item()),
                "train_cohens_d": cohens_d,
                "direction": direction.cpu(),
                "unit_direction": unit.cpu(),
            })
    return candidates


def candidate_record_for_csv(candidate: dict) -> dict:
    return {key: value for key, value in candidate.items() if key not in {"direction", "unit_direction"}}


def cross_candidate_cosine_rows(candidates: Sequence[dict]) -> List[dict]:
    rows = []
    for left in candidates:
        left_vec = left["unit_direction"]
        for right in candidates:
            right_vec = right["unit_direction"]
            rows.append({
                "left_key": left["candidate_key"],
                "right_key": right["candidate_key"],
                "left_layer": left["layer_idx"],
                "right_layer": right["layer_idx"],
                "left_position": left["source_position"],
                "right_position": right["source_position"],
                "cosine": float(torch.dot(left_vec, right_vec).item()),
            })
    return rows


# ## Candidate Types

def qr_basis(vectors: Sequence) -> "torch.Tensor":
    require_model_dependencies()
    if not vectors:
        raise ValueError("Need at least one vector to build a QR basis.")
    mat = torch.stack([vector.float().cpu() for vector in vectors], dim=1)
    q, _ = torch.linalg.qr(mat, mode="reduced")
    return q.contiguous()


def subspace_basis(positive_acts, negative_acts, rank: int) -> "torch.Tensor":
    require_model_dependencies()
    positive_acts = positive_acts.float()
    negative_acts = negative_acts.float()
    mean_delta = positive_acts.mean(dim=0) - negative_acts.mean(dim=0)
    basis_cols = []
    if mean_delta.norm() > 1e-8:
        mean_unit = mean_delta / mean_delta.norm()
        basis_cols.append(mean_unit)
        positive_resid = positive_acts - (positive_acts @ mean_unit).unsqueeze(-1) * mean_unit.unsqueeze(0)
        negative_resid = negative_acts - (negative_acts @ mean_unit).unsqueeze(-1) * mean_unit.unsqueeze(0)
    else:
        positive_resid = positive_acts
        negative_resid = negative_acts
    if len(basis_cols) < rank:
        residual = torch.cat([
            positive_resid - positive_resid.mean(dim=0, keepdim=True),
            negative_resid - negative_resid.mean(dim=0, keepdim=True),
        ], dim=0)
        _, _, vh = torch.linalg.svd(residual.float(), full_matrices=False)
        for row in vh:
            basis_cols.append(row)
            if len(basis_cols) >= rank:
                break
    basis = torch.stack(basis_cols, dim=1)
    q, _ = torch.linalg.qr(basis, mode="reduced")
    return q[:, :rank].contiguous()


def shuffled_label_subspace_basis(positive_acts, negative_acts, rank: int, seed: int) -> "torch.Tensor":
    require_model_dependencies()
    combined = torch.cat([positive_acts.float().cpu(), negative_acts.float().cpu()], dim=0)
    n_pos = int(positive_acts.shape[0])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    perm = torch.randperm(combined.shape[0], generator=generator)
    shuffled_pos = combined[perm[:n_pos]]
    shuffled_neg = combined[perm[n_pos:]]
    return subspace_basis(shuffled_pos, shuffled_neg, rank)


def random_basis_like(d_model: int, rank: int, seed: int) -> "torch.Tensor":
    require_model_dependencies()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    mat = torch.randn(d_model, rank, generator=generator)
    q, _ = torch.linalg.qr(mat, mode="reduced")
    return q[:, :rank].contiguous()


def ranked_candidates_for_stack(candidates: Sequence[dict], k: int) -> List[dict]:
    ranked = sorted(
        candidates,
        key=lambda item: (float(item["train_cohens_d"]), float(item["train_projection_gap"]), float(item["direction_norm"])),
        reverse=True,
    )
    selected = []
    used_layers = set()
    for candidate in ranked:
        layer_idx = int(candidate["layer_idx"])
        if layer_idx in used_layers:
            continue
        selected.append(candidate)
        used_layers.add(layer_idx)
        if len(selected) >= k:
            break
    if len(selected) < k:
        for candidate in ranked:
            if candidate in selected:
                continue
            selected.append(candidate)
            if len(selected) >= k:
                break
    return selected[:k]


def build_candidate_condition_specs(
    candidates: Sequence[dict],
    positive_acts,
    negative_acts,
    positions: Sequence[int],
    args: argparse.Namespace,
) -> List[dict]:
    if not candidates:
        raise ValueError("Cannot build candidate condition specs without direction candidates.")
    top_k_values = sorted(set(parse_int_list(args.top_k_values)))
    same_layer_pca_ranks = sorted(set(parse_int_list(args.same_layer_pca_ranks)))
    d_model = int(candidates[0]["unit_direction"].shape[0])
    condition_specs = []

    for candidate in candidates:
        condition_specs.append({
            "condition": f"rank1_{candidate['candidate_key']}",
            "basis_type": "rank1_mean_difference",
            "rank": 1,
            "basis": candidate["unit_direction"].unsqueeze(1).contiguous(),
            "source_candidate_keys": candidate["candidate_key"],
            "source_candidates": [candidate],
            "control_type": "extracted",
        })

    for rank in same_layer_pca_ranks:
        if rank <= 1:
            continue
        for candidate in candidates:
            layer_idx = int(candidate["layer_idx"])
            pos_idx = positions.index(int(candidate["source_position"]))
            pos_layer = positive_acts[:, layer_idx, pos_idx, :]
            neg_layer = negative_acts[:, layer_idx, pos_idx, :]

            basis = subspace_basis(pos_layer, neg_layer, rank)
            effective_rank = int(basis.shape[1])
            condition_specs.append({
                "condition": f"same_layer_pca_rank{rank}_{candidate['candidate_key']}",
                "basis_type": "same_layer_mean_delta_plus_residual_pca",
                "rank": effective_rank,
                "requested_rank": rank,
                "basis": basis,
                "source_candidate_keys": candidate["candidate_key"],
                "source_candidates": [candidate],
                "control_type": "extracted",
            })

            shuffled_basis = shuffled_label_subspace_basis(pos_layer, neg_layer, rank, args.seed + 8000 + layer_idx * 100 + rank)
            condition_specs.append({
                "condition": f"same_layer_shuffled_label_rank{rank}_{candidate['candidate_key']}",
                "basis_type": "same_layer_shuffled_label_control",
                "rank": int(shuffled_basis.shape[1]),
                "requested_rank": rank,
                "basis": shuffled_basis,
                "source_candidate_keys": candidate["candidate_key"],
                "source_candidates": [candidate],
                "control_type": "shuffled_label",
            })

            random_same_layer_basis = random_basis_like(d_model, rank, args.seed + 10000 + layer_idx * 100 + rank)
            condition_specs.append({
                "condition": f"same_layer_random_rank{rank}_{candidate['candidate_key']}",
                "basis_type": "same_layer_random_control",
                "rank": rank,
                "requested_rank": rank,
                "basis": random_same_layer_basis,
                "source_candidate_keys": candidate["candidate_key"],
                "source_candidates": [candidate],
                "control_type": "random",
            })

    for k in top_k_values:
        if k <= 1:
            continue
        selected = ranked_candidates_for_stack(candidates, k)
        if len(selected) < k:
            continue
        condition_specs.append({
            "condition": f"top{k}_stack_mean_difference",
            "basis_type": "topk_layer_vector_stack_qr",
            "rank": k,
            "basis": qr_basis([candidate["unit_direction"] for candidate in selected])[:, :k].contiguous(),
            "source_candidate_keys": "|".join(candidate["candidate_key"] for candidate in selected),
            "source_candidates": selected,
            "control_type": "extracted",
        })

    for rank in sorted(set([1, *top_k_values, *same_layer_pca_ranks])):
        if rank <= 0:
            continue
        condition_specs.append({
            "condition": f"global_random_rank{rank}",
            "basis_type": "global_random_control",
            "rank": rank,
            "requested_rank": rank,
            "basis": random_basis_like(d_model, rank, args.seed + 5000 + rank),
            "source_candidate_keys": "",
            "source_candidates": [],
            "control_type": "random",
        })

    return condition_specs


def condition_spec_record_for_csv(spec: dict) -> dict:
    return {
        key: value
        for key, value in spec.items()
        if key not in {"basis", "source_candidates"}
    }


def select_best_by_cosmic_score(cosmic_rows: Sequence[dict]) -> dict:
    if not cosmic_rows:
        raise ValueError("No COSMIC rows available for selection.")
    scored_rows = [row for row in cosmic_rows if pd.notna(row.get("cosmic_score"))]
    if not scored_rows:
        raise ValueError("No finite cosmic_score values available for selection.")
    return max(scored_rows, key=lambda row: float(row["cosmic_score"]))


def source_position_index(candidate: dict, positions: Sequence[int]) -> int:
    return positions.index(int(candidate["source_position"]))


def condition_projection_values(spec: dict, acts, positions: Sequence[int]):
    require_model_dependencies()
    candidates = spec.get("source_candidates", [])
    if spec["basis_type"] == "topk_layer_vector_stack_qr" and candidates:
        projections = []
        for candidate in candidates:
            layer_idx = int(candidate["layer_idx"])
            pos_idx = source_position_index(candidate, positions)
            vector = candidate["unit_direction"].float().cpu()
            projections.append(acts[:, layer_idx, pos_idx, :].float() @ vector)
        return torch.stack(projections, dim=1).mean(dim=1)

    if candidates:
        layer_idx = int(candidates[0]["layer_idx"])
        pos_idx = source_position_index(candidates[0], positions)
    else:
        layer_idx = 0
        pos_idx = 0
    hidden = acts[:, layer_idx, pos_idx, :].float()
    basis = spec["basis"].float().cpu()
    projection = hidden @ basis
    if int(spec["rank"]) == 1 or spec["basis_type"] == "rank1_mean_difference":
        return projection[:, 0]
    return projection.pow(2).sum(dim=1).sqrt()


def activation_separability_condition(spec: dict, positive_acts, negative_acts, positions: Sequence[int]) -> dict:
    pos_values = condition_projection_values(spec, positive_acts, positions)
    neg_values = condition_projection_values(spec, negative_acts, positions)
    pos_mean = float(pos_values.mean().item())
    neg_mean = float(neg_values.mean().item())
    pos_std = float(pos_values.std(unbiased=False).item())
    neg_std = float(neg_values.std(unbiased=False).item())
    pooled = math.sqrt(0.5 * (pos_std ** 2 + neg_std ** 2))
    gap = pos_mean - neg_mean
    cohens_d = gap / max(pooled, 1e-8)
    return {
        "condition": spec["condition"],
        "basis_type": spec["basis_type"],
        "rank": int(spec["rank"]),
        "requested_rank": spec.get("requested_rank", ""),
        "control_type": spec.get("control_type", ""),
        "source_candidate_keys": spec.get("source_candidate_keys", ""),
        "n_val_pairs": int(pos_values.shape[0]),
        "val_positive_projection_mean": pos_mean,
        "val_negative_projection_mean": neg_mean,
        "val_positive_projection_std": pos_std,
        "val_negative_projection_std": neg_std,
        "val_projection_gap": gap,
        "val_cohens_d": cohens_d,
        "activation_separability_score": cohens_d,
        "separability_score_source": "val_projection_positive_minus_negative",
    }


def prefilter_condition_specs_for_cosmic(
    condition_specs: Sequence[dict],
    positive_acts,
    negative_acts,
    positions: Sequence[int],
) -> tuple:
    separability_rows = [
        activation_separability_condition(spec, positive_acts, negative_acts, positions)
        for spec in condition_specs
    ]
    rows_by_condition = {row["condition"]: row for row in separability_rows}
    specs_by_condition = {spec["condition"]: spec for spec in condition_specs}
    return separability_rows, rows_by_condition, specs_by_condition


def cosine_between_mean_vectors(left, right) -> float:
    left_mean = left.float().mean(dim=0)
    right_mean = right.float().mean(dim=0)
    denom = left_mean.norm().clamp_min(1e-8) * right_mean.norm().clamp_min(1e-8)
    return float((torch.dot(left_mean, right_mean) / denom).item())


def select_low_similarity_eval_layers(
    positive_acts,
    negative_acts,
    positions: Sequence[int],
    args: argparse.Namespace,
) -> tuple:
    if args.cosmic_low_layer_frac <= 0:
        raise ValueError("--cosmic-low-layer-frac must be positive.")
    eval_pos_idx = positions.index(args.cosmic_eval_position)
    n_layers = int(positive_acts.shape[1])
    n_eval_layers = max(args.cosmic_min_eval_layers, int(math.ceil(n_layers * args.cosmic_low_layer_frac)))
    n_eval_layers = min(n_layers, n_eval_layers)
    rows = []
    for layer_idx in range(n_layers):
        cosine = cosine_between_mean_vectors(
            positive_acts[:, layer_idx, eval_pos_idx, :],
            negative_acts[:, layer_idx, eval_pos_idx, :],
        )
        rows.append({
            "row_type": "base_eval_layer_selection",
            "layer_idx": layer_idx,
            "eval_position": args.cosmic_eval_position,
            "train_positive_negative_cosine": cosine,
            "selected_eval_layer": False,
            "selection_rule": f"lowest_{args.cosmic_low_layer_frac:.3f}_train_class_cosine",
        })
    selected_layer_indices = [
        row["layer_idx"]
        for row in sorted(rows, key=lambda item: float(item["train_positive_negative_cosine"]))[:n_eval_layers]
    ]
    selected_set = set(selected_layer_indices)
    for row in rows:
        row["selected_eval_layer"] = row["layer_idx"] in selected_set
    return selected_layer_indices, eval_pos_idx, rows


def ranked_specs_by_separability(
    specs: Sequence[dict],
    separability_rows_by_condition: Dict[str, dict],
) -> List[dict]:
    return sorted(
        specs,
        key=lambda spec: float(separability_rows_by_condition[spec["condition"]]["activation_separability_score"]),
        reverse=True,
    )


def select_specs_for_cosmic_scoring(
    condition_specs: Sequence[dict],
    separability_rows_by_condition: Dict[str, dict],
    args: argparse.Namespace,
) -> List[dict]:
    if args.cosmic_max_scored_conditions <= 0:
        return list(condition_specs)
    extracted = [spec for spec in condition_specs if spec.get("control_type") == "extracted"]
    random_controls = [spec for spec in condition_specs if spec.get("control_type") == "random"]
    shuffled_controls = [spec for spec in condition_specs if spec.get("control_type") == "shuffled_label"]
    selected = []
    selected.extend(ranked_specs_by_separability(extracted, separability_rows_by_condition)[:args.cosmic_max_scored_conditions])
    selected.extend(ranked_specs_by_separability(random_controls, separability_rows_by_condition)[:args.cosmic_max_control_conditions_per_type])
    selected.extend(ranked_specs_by_separability(shuffled_controls, separability_rows_by_condition)[:args.cosmic_max_control_conditions_per_type])
    deduped = []
    seen = set()
    for spec in selected:
        if spec["condition"] in seen:
            continue
        deduped.append(spec)
        seen.add(spec["condition"])
    return deduped


def select_eval_activation_slice(acts, eval_layer_indices: Sequence[int], eval_pos_idx: int):
    return acts[:, list(eval_layer_indices), eval_pos_idx, :].float()


def flatten_activation_examples(acts):
    return acts.float().reshape(acts.shape[0], -1)


def mean_cosine_to_reference(acts, reference, eval_layer_indices: Sequence[int], eval_pos_idx: int):
    values = flatten_activation_examples(select_eval_activation_slice(acts, eval_layer_indices, eval_pos_idx))
    ref = reference.float().reshape(1, -1)
    values = values / values.norm(dim=1, keepdim=True).clamp_min(1e-8)
    ref = ref / ref.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return float((values @ ref.T).mean().item())


def intervention_similarity_row(
    spec: dict,
    separability_row: dict,
    positive_acts,
    negative_acts,
    ablated_positive_acts,
    added_negative_acts,
    eval_layer_indices: Sequence[int],
    eval_pos_idx: int,
) -> dict:
    positive_mean = select_eval_activation_slice(positive_acts, eval_layer_indices, eval_pos_idx).mean(dim=0)
    negative_mean = select_eval_activation_slice(negative_acts, eval_layer_indices, eval_pos_idx).mean(dim=0)

    baseline_positive_to_negative = mean_cosine_to_reference(positive_acts, negative_mean, eval_layer_indices, eval_pos_idx)
    ablated_positive_to_negative = mean_cosine_to_reference(ablated_positive_acts, negative_mean, eval_layer_indices, eval_pos_idx)
    baseline_negative_to_positive = mean_cosine_to_reference(negative_acts, positive_mean, eval_layer_indices, eval_pos_idx)
    added_negative_to_positive = mean_cosine_to_reference(added_negative_acts, positive_mean, eval_layer_indices, eval_pos_idx)

    positive_ablation_similarity_gain = ablated_positive_to_negative - baseline_positive_to_negative
    negative_addition_similarity_gain = added_negative_to_positive - baseline_negative_to_positive
    cosmic_score = 0.5 * (positive_ablation_similarity_gain + negative_addition_similarity_gain)

    row = dict(separability_row)
    row.update({
        "baseline_positive_to_negative_similarity": baseline_positive_to_negative,
        "ablated_positive_to_negative_similarity": ablated_positive_to_negative,
        "positive_ablation_similarity_gain": positive_ablation_similarity_gain,
        "baseline_negative_to_positive_similarity": baseline_negative_to_positive,
        "added_negative_to_positive_similarity": added_negative_to_positive,
        "negative_addition_similarity_gain": negative_addition_similarity_gain,
        "cosmic_score": cosmic_score,
        "score_source": "intervention_internal_activation_similarity",
        "eval_layer_indices": "|".join(str(idx) for idx in eval_layer_indices),
        "eval_position": "",
    })
    return row


def build_cosmic_ranked_stack_specs(
    scored_rows: Sequence[dict],
    specs_by_condition: Dict[str, dict],
    args: argparse.Namespace,
) -> List[dict]:
    top_k_values = sorted(set(parse_int_list(args.top_k_values)))
    rank1_rows = [
        row for row in scored_rows
        if row.get("basis_type") == "rank1_mean_difference"
        and row.get("control_type") == "extracted"
        and pd.notna(row.get("cosmic_score"))
        and str(row.get("condition")) in specs_by_condition
    ]
    ranked = sorted(rank1_rows, key=lambda row: float(row["cosmic_score"]), reverse=True)
    stack_specs = []
    for k in top_k_values:
        if k <= 1 or len(ranked) < k:
            continue
        selected_specs = [specs_by_condition[row["condition"]] for row in ranked[:k]]
        selected_candidates = []
        for spec in selected_specs:
            selected_candidates.extend(spec.get("source_candidates", []))
        if len(selected_candidates) < k:
            continue
        stack_specs.append({
            "condition": f"cosmic_top{k}_rank1_stack_mean_difference",
            "basis_type": "cosmic_ranked_topk_layer_vector_stack_qr",
            "rank": k,
            "basis": qr_basis([candidate["unit_direction"] for candidate in selected_candidates[:k]])[:, :k].contiguous(),
            "source_candidate_keys": "|".join(candidate["candidate_key"] for candidate in selected_candidates[:k]),
            "source_candidates": selected_candidates[:k],
            "control_type": "extracted",
            "stack_selection_source": "rank1_cosmic_score",
        })
    return stack_specs


def score_condition_specs_by_cosmic_intervention(
    model,
    tokenizer,
    condition_specs: Sequence[dict],
    positive_acts,
    negative_acts,
    positive_prompts: Sequence[str],
    negative_prompts: Sequence[str],
    positions: Sequence[int],
    eval_layer_indices: Sequence[int],
    eval_pos_idx: int,
    args: argparse.Namespace,
) -> tuple:
    separability_rows, separability_by_condition, _ = prefilter_condition_specs_for_cosmic(
        condition_specs,
        positive_acts,
        negative_acts,
        positions,
    )
    specs_by_condition = {spec["condition"]: spec for spec in condition_specs}
    specs_to_score = select_specs_for_cosmic_scoring(condition_specs, separability_by_condition, args)
    rows = []
    similarity_rows = []

    def score_one_spec(spec: dict, idx: int, total: int) -> dict:
        print(
            f"[{time.strftime('%H:%M:%S')}] COSMIC intervention scoring "
            f"{idx}/{total}: {spec['condition']}"
        )
        ablated_positive_acts = collect_resid_pre_by_layer_position(
            model,
            tokenizer,
            positive_prompts,
            positions,
            args.batch_size,
            args.max_input_tokens,
            args.allow_truncated_prompts,
            args.prompt_format,
            intervention_factory=lambda spec=spec: all_layer_all_token_subspace_ablation(model, spec["basis"]),
        )
        added_negative_acts = collect_resid_pre_by_layer_position(
            model,
            tokenizer,
            negative_prompts,
            positions,
            args.batch_size,
            args.max_input_tokens,
            args.allow_truncated_prompts,
            args.prompt_format,
            intervention_factory=lambda spec=spec: selected_layer_direction_addition(model, spec, args.addition_scale),
        )
        row = intervention_similarity_row(
            spec,
            separability_by_condition.get(
                spec["condition"],
                activation_separability_condition(spec, positive_acts, negative_acts, positions),
            ),
            positive_acts,
            negative_acts,
            ablated_positive_acts,
            added_negative_acts,
            eval_layer_indices,
            eval_pos_idx,
        )
        row["addition_scale"] = args.addition_scale
        row["eval_position"] = positions[eval_pos_idx]
        rows.append(row)
        similarity_rows.append({
            "row_type": "condition_intervention_similarity",
            "condition": row["condition"],
            "basis_type": row["basis_type"],
            "rank": row["rank"],
            "control_type": row["control_type"],
            "eval_activation_scope": "lowest_similarity_layers_at_eval_position",
            "eval_layer_indices": "|".join(str(idx) for idx in eval_layer_indices),
            "eval_position": positions[eval_pos_idx],
            "baseline_positive_to_negative_similarity": row["baseline_positive_to_negative_similarity"],
            "ablated_positive_to_negative_similarity": row["ablated_positive_to_negative_similarity"],
            "positive_ablation_similarity_gain": row["positive_ablation_similarity_gain"],
            "baseline_negative_to_positive_similarity": row["baseline_negative_to_positive_similarity"],
            "added_negative_to_positive_similarity": row["added_negative_to_positive_similarity"],
            "negative_addition_similarity_gain": row["negative_addition_similarity_gain"],
            "cosmic_score": row["cosmic_score"],
        })
        del ablated_positive_acts
        del added_negative_acts
        release_memory()
        return row

    for idx, spec in enumerate(specs_to_score, start=1):
        score_one_spec(spec, idx, len(specs_to_score))

    cosmic_stack_specs = build_cosmic_ranked_stack_specs(rows, specs_by_condition, args)
    for idx, spec in enumerate(cosmic_stack_specs, start=1):
        score_one_spec(spec, idx, len(cosmic_stack_specs))
        specs_by_condition[spec["condition"]] = spec

    extracted_rows = [
        row
        for row in rows
        if row.get("control_type") == "extracted" and pd.notna(row.get("cosmic_score"))
    ]
    ranked_extracted = sorted(extracted_rows, key=lambda row: float(row["cosmic_score"]), reverse=True)
    rank_by_condition = {
        row["condition"]: rank_idx + 1
        for rank_idx, row in enumerate(ranked_extracted)
    }
    best_condition = ranked_extracted[0]["condition"] if ranked_extracted else ""
    for row in rows:
        row["cosmic_prefiltered"] = True
        row["cosmic_scored_condition_count"] = len(specs_to_score) + len(cosmic_stack_specs)
        row["total_condition_count"] = len(condition_specs) + len(cosmic_stack_specs)
        row["selection_rank_extracted_only"] = rank_by_condition.get(row["condition"], "")
        row["selected_best_extracted"] = row["condition"] == best_condition
    return rows, similarity_rows, cosmic_stack_specs


def selected_spec_from_cosmic_rows(condition_specs: Sequence[dict], cosmic_rows: Sequence[dict]) -> tuple:
    by_condition = {spec["condition"]: spec for spec in condition_specs}

    def is_selected(value) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes"}
        return bool(value)

    selected_rows = [
        row for row in cosmic_rows
        if is_selected(row.get("selected_best_extracted")) and str(row.get("condition")) in by_condition
    ]
    if not selected_rows:
        extracted_rows = [
            row for row in cosmic_rows
            if row.get("control_type") == "extracted" and str(row.get("condition")) in by_condition
        ]
        if not extracted_rows:
            return None, None
        selected_row = max(extracted_rows, key=lambda row: float(row["cosmic_score"]))
    else:
        selected_row = selected_rows[0]
    return by_condition[str(selected_row["condition"])], selected_row


def save_selected_controller_artifact(spec: dict, cosmic_row: dict, output_root: Path) -> Path:
    require_model_dependencies()
    path = output_root / COSMIC_SELECTED_CONTROLLER_PT
    payload = {
        "condition": spec["condition"],
        "basis_type": spec["basis_type"],
        "rank": int(spec["rank"]),
        "control_type": spec.get("control_type", ""),
        "source_candidate_keys": spec.get("source_candidate_keys", ""),
        "basis": spec["basis"].detach().float().cpu(),
        "cosmic_row": dict(cosmic_row),
        "activation_site": "resid_pre / decoder layer input",
        "hook_type": "forward_pre_hook",
        "selection_score_source": cosmic_row.get("score_source", ""),
        "intervention_scope_for_ablation": "all_layers_all_tokens",
        "ablation_scale": cosmic_row.get("ablation_scale", 1.0),
        "addition_scale": cosmic_row.get("addition_scale", ""),
        "source_candidates": [
            candidate_record_for_csv(candidate)
            for candidate in spec.get("source_candidates", [])
        ],
    }
    torch.save(payload, path)
    return path


# ## Behavior Validation

REFUSAL_PATTERNS = [
    r"\bi do not know\b",
    r"\bi don'?t know\b",
    r"\bi have no knowledge of\b",
    r"\bi have no information about\b",
    r"\bi don'?t have information about\b",
    r"\bi am not sure\b",
    r"\bi'?m not sure\b",
    r"\bnot sure\b",
    r"\buncertain\b",
    r"\bi am unsure\b",
    r"\bi'?m unsure\b",
    r"\bi am stumped\b",
    r"\bi'?m stumped\b",
    r"\bstumped on\b",
    r"\bi am blank\b",
    r"\bi'?m blank\b",
    r"\bblank on\b",
    r"\bi cannot\b",
    r"\bi can'?t\b",
    r"\bi am unable\b",
    r"\bi'?m unable\b",
    r"\bi do not have\b",
    r"\bi don'?t have\b",
    r"\bno information\b",
    r"\bnot enough information\b",
    r"\binsufficient information\b",
    r"\bunknown\b",
    r"\bnot available\b",
    r"\bnot familiar with\b",
    r"\bneed to look up\b",
    r"\bcannot find any information\b",
    r"\bi'?m sorry\b",
    r"\bas an ai\b",
    r"\bas a language model\b",
    r"\bnot at liberty\b",
    r"\bcannot be answered\b",
    r"\bcan'?t be answered\b",
]


def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_refusal_or_idk(text: str) -> bool:
    norm = normalize_text(text)
    return any(re.search(pattern, norm) for pattern in REFUSAL_PATTERNS)


def is_degenerate(text: str) -> bool:
    norm = normalize_text(text)
    if not norm:
        return True
    toks = norm.split()
    if len(toks) < 3:
        return True
    unique_ratio = len(set(toks)) / max(1, len(toks))
    if len(toks) >= 20 and unique_ratio < 0.30:
        return True
    if re.search(r"\b(\w+)(?:\s+\1\b){3,}", norm):
        return True
    return False


def project_hidden_away_from_basis(hidden, basis, scale: float = 1.0):
    local_basis = basis.to(device=hidden.device, dtype=hidden.dtype)
    projection = hidden @ local_basis @ local_basis.T
    return hidden - float(scale) * projection


def add_direction_to_hidden(hidden, direction, scale: float):
    local_direction = direction.to(device=hidden.device, dtype=hidden.dtype)
    return hidden + scale * local_direction


def post_attention_resid_modules(layer) -> list:
    modules = []
    if hasattr(layer, "post_attention_layernorm"):
        modules.append(layer.post_attention_layernorm)
    elif hasattr(layer, "ln_2"):
        modules.append(layer.ln_2)
    elif hasattr(layer, "post_attention_norm"):
        modules.append(layer.post_attention_norm)
    return modules


def final_resid_modules(model) -> list:
    base = getattr(model, "model", model)
    for name in ["norm", "ln_f", "final_layernorm"]:
        if hasattr(base, name):
            return [getattr(base, name)]
    return []


@contextmanager
def all_layer_all_token_subspace_ablation(model, basis, scale: float = 1.0):
    handles = []

    def hook(_module, inputs):
        hidden = inputs[0]
        return (project_hidden_away_from_basis(hidden, basis, scale),) + tuple(inputs[1:])

    for layer in get_decoder_layers(model):
        handles.append(layer.register_forward_pre_hook(hook))
        for module in post_attention_resid_modules(layer):
            handles.append(module.register_forward_pre_hook(hook))
    for module in final_resid_modules(model):
        handles.append(module.register_forward_pre_hook(hook))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def selected_layer_direction_addition(model, spec: dict, scale: float):
    handles = []
    candidates = spec.get("source_candidates", [])

    if spec["basis_type"] == "rank1_mean_difference" and candidates:
        layer_idx = int(candidates[0]["layer_idx"])
        direction = candidates[0]["direction"].float().cpu()
    elif spec["basis_type"] in {"topk_layer_vector_stack_qr", "cosmic_ranked_topk_layer_vector_stack_qr"} and candidates:
        layers = get_decoder_layers(model)

        def make_hook(direction):
            def hook(_module, inputs):
                hidden = inputs[0]
                return (add_direction_to_hidden(hidden, direction, scale),) + tuple(inputs[1:])
            return hook

        for candidate in candidates:
            handles.append(layers[int(candidate["layer_idx"])].register_forward_pre_hook(make_hook(candidate["direction"].float().cpu())))
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()
        return
    else:
        layer_idx = int(candidates[0]["layer_idx"]) if candidates else 0
        direction = spec["basis"][:, 0].float().cpu()

    def hook(_module, inputs):
        hidden = inputs[0]
        return (add_direction_to_hidden(hidden, direction, scale),) + tuple(inputs[1:])

    handles.append(get_decoder_layers(model)[layer_idx].register_forward_pre_hook(hook))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def generation_records_from_paired(records: Sequence[dict], prompt_field: str, prompt_mode: str) -> List[dict]:
    return [
        {
            "record_id": record["record_id"],
            "source_index": record["source_index"],
            "sample_id": record["sample_id"],
            "split": record["split"],
            "prompt_mode": prompt_mode,
            "positive_prompt_variant": record["positive_prompt_variant"],
            "question": record["question"],
            "prompt": record[prompt_field],
        }
        for record in records
    ]


def unique_plain_records(records: Sequence[dict], max_records: int, seed: int) -> List[dict]:
    by_source = {}
    for record in sorted(records, key=lambda item: (item["source_index"], item["positive_prompt_variant"])):
        by_source.setdefault(record["source_index"], record)
    return sample_records(list(by_source.values()), max_records, seed)


def sample_records(records: Sequence[dict], max_records: int, seed: int) -> List[dict]:
    records = list(records)
    if max_records <= 0 or len(records) <= max_records:
        return records
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(records)), max_records))
    return [records[idx] for idx in indices]


@torch.inference_mode() if torch is not None else (lambda fn: fn)
def generate_behavior_records(
    model,
    tokenizer,
    records: Sequence[dict],
    args: argparse.Namespace,
    intervention: str,
    metadata: Optional[dict] = None,
) -> List[dict]:
    require_model_dependencies()
    rows = []
    do_sample = args.temperature > 0
    for start in range(0, len(records), args.batch_size):
        batch_records = list(records[start:start + args.batch_size])
        prompts = [record["prompt"] for record in batch_records]
        encoded = encode_batch(
            tokenizer,
            prompts,
            model_input_device(model),
            args.max_input_tokens,
            args.allow_truncated_prompts,
            f"{intervention} generation prompts",
            args.prompt_format,
        )
        generation_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "use_cache": False,
        }
        if do_sample:
            generation_kwargs["temperature"] = args.temperature
        generated = model.generate(**encoded, **generation_kwargs)
        prompt_len = encoded["input_ids"].shape[1]
        texts = tokenizer.batch_decode(generated[:, prompt_len:], skip_special_tokens=True)
        for record, text in zip(batch_records, texts):
            generation = text.strip()
            row = {
                **record,
                "intervention": intervention,
                "generation": generation,
                "is_refusal_or_idk": is_refusal_or_idk(generation),
                "is_degenerate": is_degenerate(generation),
            }
            if metadata:
                row.update(metadata)
            rows.append(row)
    return rows


def refusal_rate(rows: Sequence[dict]) -> float:
    return float(np.mean([bool(row["is_refusal_or_idk"]) for row in rows])) if rows else float("nan")


def degeneracy_rate(rows: Sequence[dict]) -> float:
    return float(np.mean([bool(row["is_degenerate"]) for row in rows])) if rows else float("nan")


def behavior_summary_row(
    label: str,
    baseline_rows: Sequence[dict],
    intervention_rows: Sequence[dict],
    condition_spec: dict,
    metadata: Optional[dict] = None,
) -> dict:
    baseline_rate = refusal_rate(baseline_rows)
    intervention_rate = refusal_rate(intervention_rows)
    baseline_degenerate_rate = degeneracy_rate(baseline_rows)
    intervention_degenerate_rate = degeneracy_rate(intervention_rows)
    row = {
        "validation_label": label,
        "condition": condition_spec["condition"],
        "basis_type": condition_spec["basis_type"],
        "rank": condition_spec["rank"],
        "control_type": condition_spec.get("control_type", ""),
        "n": len(intervention_rows),
        "baseline_refusal_or_idk_rate": baseline_rate,
        "intervention_refusal_or_idk_rate": intervention_rate,
        "refusal_or_idk_rate_delta": intervention_rate - baseline_rate,
        "baseline_degenerate_rate": baseline_degenerate_rate,
        "intervention_degenerate_rate": intervention_degenerate_rate,
        "degenerate_rate_delta": intervention_degenerate_rate - baseline_degenerate_rate,
    }
    if metadata:
        row.update(metadata)
    return row


def generation_lookup(rows: Sequence[dict]) -> Dict[tuple, dict]:
    return {
        (row["record_id"], row["intervention"]): row
        for row in rows
    }


def generation_rows_by_record(rows: Sequence[dict]) -> Dict[str, List[dict]]:
    by_record: Dict[str, List[dict]] = {}
    for row in rows:
        by_record.setdefault(row["record_id"], []).append(row)
    return by_record


def side_by_side_generation_rows(generation_rows: Sequence[dict], specs: Sequence[dict]) -> List[dict]:
    lookup = generation_lookup(generation_rows)
    by_record = generation_rows_by_record(generation_rows)
    baseline_by_mode = {
        "positive_modified_idk_prompt": "baseline_positive",
        "negative_matched_plain_prompt": "baseline_negative",
        "plain_unique_source_prompt": "baseline_plain",
    }
    intervention_suffix_by_mode = {
        "positive_modified_idk_prompt": ["positive_ablation"],
        "negative_matched_plain_prompt": ["negative_addition"],
        "plain_unique_source_prompt": ["plain_addition", "plain_ablation"],
    }
    rows = []
    for baseline in generation_rows:
        if baseline["intervention"] not in set(baseline_by_mode.values()):
            continue
        prompt_mode = baseline["prompt_mode"]
        suffixes = intervention_suffix_by_mode[prompt_mode]
        for spec in specs:
            for suffix in suffixes:
                intervention_name = f"{spec['condition']}_{suffix}"
                matched_rows = [
                    row for row in by_record.get(baseline["record_id"], [])
                    if row.get("condition") == spec["condition"]
                    and row.get("intervention_kind") == suffix
                ]
                if not matched_rows:
                    intervened = lookup.get((baseline["record_id"], intervention_name))
                    matched_rows = [intervened] if intervened is not None else []
                for intervened in matched_rows:
                    rows.append({
                        "condition": spec["condition"],
                        "basis_type": spec["basis_type"],
                        "rank": spec["rank"],
                        "control_type": spec.get("control_type", ""),
                        "ablation_scale": intervened.get("ablation_scale", ""),
                        "addition_scale": intervened.get("addition_scale", ""),
                        "prompt_mode": prompt_mode,
                        "source_index": baseline["source_index"],
                        "sample_id": baseline["sample_id"],
                        "positive_prompt_variant": baseline["positive_prompt_variant"],
                        "question": baseline["question"],
                        "prompt": baseline["prompt"],
                        "baseline_intervention": baseline["intervention"],
                        "intervention": intervened["intervention"],
                        "baseline_generation": baseline["generation"],
                        "intervention_generation": intervened["generation"],
                        "baseline_is_refusal_or_idk": baseline["is_refusal_or_idk"],
                        "intervention_is_refusal_or_idk": intervened["is_refusal_or_idk"],
                        "baseline_is_degenerate": baseline["is_degenerate"],
                        "intervention_is_degenerate": intervened["is_degenerate"],
                    })
    return rows


def desired_behavior_effect(row: dict) -> float:
    delta = float(row["refusal_or_idk_rate_delta"])
    label = str(row["validation_label"])
    if label in {"positive_heldout_ablation", "positive_val_ablation"}:
        return -delta
    if label in {"negative_heldout_addition", "plain_heldout_addition", "negative_val_addition", "plain_val_addition"}:
        return delta
    if label in {"plain_heldout_ablation", "plain_val_ablation"}:
        return -abs(delta)
    return float("nan")


def finite_max(values: Sequence[float]) -> float:
    finite_values = [float(value) for value in values if not math.isnan(float(value))]
    return max(finite_values) if finite_values else float("nan")


def diagnostic_decision_rows(summary_rows: Sequence[dict]) -> List[dict]:
    if not summary_rows:
        return []

    def decision_key(row: dict) -> tuple:
        return (
            row.get("condition", ""),
            row.get("ablation_scale", ""),
            row.get("addition_scale", ""),
        )

    by_condition: Dict[tuple, List[dict]] = {}
    for row in summary_rows:
        by_condition.setdefault(decision_key(row), []).append(row)

    rows = []
    extracted_conditions = [
        key
        for key, condition_rows in by_condition.items()
        if condition_rows and condition_rows[0].get("control_type") == "extracted"
    ]
    control_rows = [
        row
        for row in summary_rows
        if row.get("control_type") in {"random", "shuffled_label"}
    ]
    for condition in extracted_conditions:
        condition_rows = by_condition[condition]
        effects = {row["validation_label"]: desired_behavior_effect(row) for row in condition_rows}
        target_labels = {
            "positive_heldout_ablation",
            "negative_heldout_addition",
            "plain_heldout_addition",
            "positive_val_ablation",
            "negative_val_addition",
            "plain_val_addition",
        }
        target_effects = {
            label: effect
            for label, effect in effects.items()
            if label in target_labels and not math.isnan(effect)
        }
        min_target_effect = min(target_effects.values()) if target_effects else float("nan")
        control_effects = [
            desired_behavior_effect(row)
            for row in control_rows
            if row["validation_label"] in target_labels and not math.isnan(desired_behavior_effect(row))
        ]
        best_control_effect = max(control_effects) if control_effects else float("nan")
        passes_directionality = (
            finite_max([
                effects.get("positive_heldout_ablation", float("nan")),
                effects.get("positive_val_ablation", float("nan")),
            ]) > 0
            and finite_max([
                effects.get("negative_heldout_addition", float("nan")),
                effects.get("negative_val_addition", float("nan")),
            ]) > 0
            and finite_max([
                effects.get("plain_heldout_addition", float("nan")),
                effects.get("plain_val_addition", float("nan")),
            ]) > 0
        )
        plain_ablation_rows = [
            row for row in condition_rows
            if row["validation_label"] in {"plain_heldout_ablation", "plain_val_ablation"}
        ]
        plain_ablation_row = plain_ablation_rows[0] if plain_ablation_rows else {}
        plain_ablation_refusal_delta = float(plain_ablation_row.get("refusal_or_idk_rate_delta", float("nan")))
        plain_ablation_degenerate_delta = float(plain_ablation_row.get("degenerate_rate_delta", float("nan")))
        plain_ablation_intact = (
            not math.isnan(plain_ablation_refusal_delta)
            and not math.isnan(plain_ablation_degenerate_delta)
            and plain_ablation_refusal_delta <= 0
            and plain_ablation_degenerate_delta <= 0
        )
        controls_weaker = math.isnan(best_control_effect) or min_target_effect > best_control_effect
        rows.append({
            "condition": condition_rows[0]["condition"],
            "basis_type": condition_rows[0]["basis_type"],
            "rank": condition_rows[0]["rank"],
            "control_type": "extracted",
            "ablation_scale": condition_rows[0].get("ablation_scale", ""),
            "addition_scale": condition_rows[0].get("addition_scale", ""),
            "positive_ablation_refusal_or_idk_reduction": finite_max([
                effects.get("positive_heldout_ablation", float("nan")),
                effects.get("positive_val_ablation", float("nan")),
            ]),
            "negative_addition_refusal_or_idk_increase": finite_max([
                effects.get("negative_heldout_addition", float("nan")),
                effects.get("negative_val_addition", float("nan")),
            ]),
            "plain_addition_refusal_or_idk_increase": finite_max([
                effects.get("plain_heldout_addition", float("nan")),
                effects.get("plain_val_addition", float("nan")),
            ]),
            "plain_ablation_refusal_or_idk_delta": plain_ablation_refusal_delta,
            "plain_ablation_degenerate_delta": plain_ablation_degenerate_delta,
            "plain_ablation_intact": plain_ablation_intact,
            "min_target_effect": min_target_effect,
            "best_control_effect": best_control_effect,
            "passes_directionality": passes_directionality,
            "controls_weaker": controls_weaker,
            "diagnostic_success": bool(passes_directionality and plain_ablation_intact and controls_weaker),
            "decision_note": "success requires positive ablation reduction, negative/plain addition induction, plain ablation non-damage, and weaker controls",
        })
    return rows


def load_cosmic_selection_rows(path: Path) -> List[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    df = pd.read_csv(path)
    if "condition" not in df.columns or "cosmic_score" not in df.columns:
        raise ValueError(f"COSMIC selection CSV must contain condition and cosmic_score columns: {path}")
    return df.to_dict("records")


def select_behavior_validation_specs(condition_specs: Sequence[dict], cosmic_rows: Sequence[dict]) -> List[dict]:
    by_condition = {spec["condition"]: spec for spec in condition_specs}
    rows_by_condition = {str(row["condition"]): row for row in cosmic_rows}

    scored_specs = []
    for condition, row in rows_by_condition.items():
        spec = by_condition.get(condition)
        if spec is None:
            continue
        score = pd.to_numeric(row.get("cosmic_score"), errors="coerce")
        if pd.isna(score):
            continue
        scored_specs.append((float(score), spec))

    selected = []
    extracted = [(score, spec) for score, spec in scored_specs if spec.get("control_type") == "extracted"]
    random_controls = [(score, spec) for score, spec in scored_specs if spec.get("control_type") == "random"]
    shuffled_controls = [(score, spec) for score, spec in scored_specs if spec.get("control_type") == "shuffled_label"]
    for pool in [extracted, random_controls, shuffled_controls]:
        if pool:
            selected.append(max(pool, key=lambda item: item[0])[1])
    return selected


def scale_tag(value: float) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def behavior_metadata(spec: dict, intervention_kind: str, ablation_scale: float, addition_scale: float) -> dict:
    return {
        "condition": spec["condition"],
        "basis_type": spec["basis_type"],
        "rank": spec["rank"],
        "control_type": spec.get("control_type", ""),
        "intervention_kind": intervention_kind,
        "ablation_scale": ablation_scale,
        "addition_scale": addition_scale,
    }


def evaluate_behavior_configs(
    model,
    tokenizer,
    specs_to_validate: Sequence[dict],
    records: Sequence[dict],
    args: argparse.Namespace,
    samples: int,
    seed_offset: int,
    label_prefix: str,
    ablation_scales: Sequence[float],
    addition_scales: Sequence[float],
) -> tuple:
    positive_subset = sample_records(records, samples, args.seed + seed_offset + 1000)
    negative_subset = sample_records(records, samples, args.seed + seed_offset + 2000)
    plain_subset = unique_plain_records(records, samples, args.seed + seed_offset + 3000)

    positive_generation_records = generation_records_from_paired(positive_subset, "positive_prompt", "positive_modified_idk_prompt")
    negative_generation_records = generation_records_from_paired(negative_subset, "negative_prompt", "negative_matched_plain_prompt")
    plain_generation_records = generation_records_from_paired(plain_subset, "negative_prompt", "plain_unique_source_prompt")

    baseline_positive = generate_behavior_records(model, tokenizer, positive_generation_records, args, "baseline_positive")
    baseline_negative = generate_behavior_records(model, tokenizer, negative_generation_records, args, "baseline_negative")
    baseline_plain = generate_behavior_records(model, tokenizer, plain_generation_records, args, "baseline_plain")

    generation_rows = []
    summary_rows = []
    generation_rows.extend(baseline_positive)
    generation_rows.extend(baseline_negative)
    generation_rows.extend(baseline_plain)

    scale_pairs = [
        (float(ablation_scale), float(addition_scale))
        for ablation_scale in ablation_scales
        for addition_scale in addition_scales
    ]
    for spec in specs_to_validate:
        for ablation_scale, addition_scale in scale_pairs:
            tag = f"ab{scale_tag(ablation_scale)}_add{scale_tag(addition_scale)}"
            with all_layer_all_token_subspace_ablation(model, spec["basis"], ablation_scale):
                positive_metadata = behavior_metadata(spec, "positive_ablation", ablation_scale, addition_scale)
                plain_ablation_metadata = behavior_metadata(spec, "plain_ablation", ablation_scale, addition_scale)
                ablated_positive = generate_behavior_records(
                    model,
                    tokenizer,
                    positive_generation_records,
                    args,
                    f"{spec['condition']}_positive_ablation_{tag}",
                    positive_metadata,
                )
                ablated_plain = generate_behavior_records(
                    model,
                    tokenizer,
                    plain_generation_records,
                    args,
                    f"{spec['condition']}_plain_ablation_{tag}",
                    plain_ablation_metadata,
                )
            with selected_layer_direction_addition(model, spec, addition_scale):
                negative_metadata = behavior_metadata(spec, "negative_addition", ablation_scale, addition_scale)
                plain_addition_metadata = behavior_metadata(spec, "plain_addition", ablation_scale, addition_scale)
                added_negative = generate_behavior_records(
                    model,
                    tokenizer,
                    negative_generation_records,
                    args,
                    f"{spec['condition']}_negative_addition_{tag}",
                    negative_metadata,
                )
                added_plain = generate_behavior_records(
                    model,
                    tokenizer,
                    plain_generation_records,
                    args,
                    f"{spec['condition']}_plain_addition_{tag}",
                    plain_addition_metadata,
                )
            generation_rows.extend(ablated_positive)
            generation_rows.extend(ablated_plain)
            generation_rows.extend(added_negative)
            generation_rows.extend(added_plain)
            summary_metadata = {
                "ablation_scale": ablation_scale,
                "addition_scale": addition_scale,
            }
            summary_rows.append(behavior_summary_row(f"positive_{label_prefix}_ablation", baseline_positive, ablated_positive, spec, summary_metadata))
            summary_rows.append(behavior_summary_row(f"plain_{label_prefix}_ablation", baseline_plain, ablated_plain, spec, summary_metadata))
            summary_rows.append(behavior_summary_row(f"negative_{label_prefix}_addition", baseline_negative, added_negative, spec, summary_metadata))
            summary_rows.append(behavior_summary_row(f"plain_{label_prefix}_addition", baseline_plain, added_plain, spec, summary_metadata))

    side_by_side_rows = side_by_side_generation_rows(generation_rows, specs_to_validate)
    decision_rows = diagnostic_decision_rows(summary_rows)
    return summary_rows, side_by_side_rows, decision_rows


def finite_float(value, default: float = float("nan")) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def behavior_candidate_sort_key(row: dict) -> tuple:
    positive_gain = finite_float(row.get("positive_ablation_similarity_gain"))
    negative_gain = finite_float(row.get("negative_addition_similarity_gain"))
    cosmic_score = finite_float(row.get("cosmic_score"))
    comparable_positive = positive_gain if math.isfinite(positive_gain) else float("-inf")
    comparable_negative = negative_gain if math.isfinite(negative_gain) else float("-inf")
    balanced_gain = min(comparable_positive, comparable_negative)
    return (
        comparable_positive > 0,
        comparable_negative > 0,
        balanced_gain,
        cosmic_score if math.isfinite(cosmic_score) else float("-inf"),
    )


def select_behavior_first_candidate_specs(
    condition_specs: Sequence[dict],
    cosmic_rows: Sequence[dict],
    args: argparse.Namespace,
) -> List[dict]:
    by_condition = {spec["condition"]: spec for spec in condition_specs}
    allowed_ranks = set(parse_int_list(args.behavior_selection_ranks))
    candidates = []
    for row in cosmic_rows:
        condition = str(row.get("condition", ""))
        spec = by_condition.get(condition)
        if spec is None or spec.get("control_type") != "extracted":
            continue
        rank = int(float(spec.get("rank", 0)))
        if allowed_ranks and rank not in allowed_ranks:
            continue
        if pd.isna(row.get("cosmic_score")):
            continue
        if args.behavior_selection_require_internal_directionality:
            if finite_float(row.get("positive_ablation_similarity_gain")) <= 0:
                continue
            if finite_float(row.get("negative_addition_similarity_gain")) <= 0:
                continue
        candidates.append((behavior_candidate_sort_key(row), spec))
    ranked = sorted(candidates, key=lambda item: item[0], reverse=True)
    return [spec for _, spec in ranked[:max(0, args.behavior_selection_max_candidates)]]


def behavior_selection_score(row: dict, args: argparse.Namespace) -> float:
    positive = finite_float(row.get("positive_ablation_refusal_or_idk_reduction"))
    negative = finite_float(row.get("negative_addition_refusal_or_idk_increase"))
    plain_add = finite_float(row.get("plain_addition_refusal_or_idk_increase"))
    plain_ab_delta = finite_float(row.get("plain_ablation_refusal_or_idk_delta"))
    plain_ab_degen = finite_float(row.get("plain_ablation_degenerate_delta"))
    if positive < args.behavior_min_positive_ablation_effect:
        return float("-inf")
    if negative < args.behavior_min_negative_addition_effect:
        return float("-inf")
    if plain_ab_delta > args.behavior_max_plain_ablation_refusal_delta:
        return float("-inf")
    if plain_ab_degen > args.behavior_max_plain_ablation_degenerate_delta:
        return float("-inf")
    plain_add_term = plain_add if math.isfinite(plain_add) else 0.0
    return positive + 0.5 * negative + 0.25 * plain_add_term - max(0.0, plain_ab_delta) - max(0.0, plain_ab_degen)


def selected_behavior_decision(decision_rows: Sequence[dict], args: argparse.Namespace) -> Optional[dict]:
    extracted = [row for row in decision_rows if row.get("control_type") == "extracted"]
    if not extracted:
        return None
    scored = []
    for row in extracted:
        score = behavior_selection_score(row, args)
        enriched = dict(row)
        enriched["behavior_selection_score"] = score
        enriched["selected_behavior_first"] = False
        scored.append(enriched)
    finite_scored = [row for row in scored if math.isfinite(float(row["behavior_selection_score"]))]
    if not finite_scored:
        return None
    selected = max(finite_scored, key=lambda row: float(row["behavior_selection_score"]))
    selected["selected_behavior_first"] = True
    return selected


def attach_behavior_selection_flags(decision_rows: Sequence[dict], selected_row: Optional[dict], args: argparse.Namespace) -> List[dict]:
    selected_key = None
    if selected_row:
        selected_key = (
            selected_row.get("condition", ""),
            selected_row.get("ablation_scale", ""),
            selected_row.get("addition_scale", ""),
        )
    rows = []
    for row in decision_rows:
        enriched = dict(row)
        enriched["behavior_selection_score"] = behavior_selection_score(enriched, args)
        enriched["selected_behavior_first"] = selected_key == (
            enriched.get("condition", ""),
            enriched.get("ablation_scale", ""),
            enriched.get("addition_scale", ""),
        )
        rows.append(enriched)
    return rows


def matched_control_specs(
    condition_specs: Sequence[dict],
    cosmic_rows: Sequence[dict],
    selected_spec: dict,
) -> List[dict]:
    by_condition = {spec["condition"]: spec for spec in condition_specs}
    row_by_condition = {str(row.get("condition", "")): row for row in cosmic_rows}
    selected_rank = int(float(selected_spec.get("rank", 0)))
    controls = []
    for control_type in ["random", "shuffled_label"]:
        pool = []
        for spec in condition_specs:
            if spec.get("control_type") != control_type:
                continue
            if int(float(spec.get("rank", 0))) != selected_rank:
                continue
            row = row_by_condition.get(spec["condition"], {})
            score = finite_float(row.get("cosmic_score"), 0.0)
            pool.append((score, spec))
        if pool:
            controls.append(max(pool, key=lambda item: item[0])[1])
    return controls


def run_behavior_first_selection(
    model,
    tokenizer,
    condition_specs: Sequence[dict],
    cosmic_rows: Sequence[dict],
    paired_records_by_split: Dict[str, list],
    args: argparse.Namespace,
    output_root: Path,
) -> tuple:
    specs_to_select = select_behavior_first_candidate_specs(condition_specs, cosmic_rows, args)
    if not specs_to_select:
        print(f"[{time.strftime('%H:%M:%S')}] behavior-first selection skipped: no eligible COSMIC candidates")
        return None, None, []
    print(
        f"[{time.strftime('%H:%M:%S')}] behavior-first selection candidates: "
        f"{len(specs_to_select)} across ranks {args.behavior_selection_ranks}"
    )
    summary_rows, side_by_side_rows, decision_rows = evaluate_behavior_configs(
        model,
        tokenizer,
        specs_to_select,
        paired_records_by_split["val"],
        args,
        args.behavior_selection_samples,
        22000,
        "val",
        parse_float_list(args.behavior_selection_ablation_scales),
        parse_float_list(args.behavior_selection_addition_scales),
    )
    selected_row = selected_behavior_decision(decision_rows, args)
    decision_rows = attach_behavior_selection_flags(decision_rows, selected_row, args)
    if selected_row is not None:
        selected_row = [
            row for row in decision_rows
            if row.get("selected_behavior_first")
        ][0]

    metrics_path = output_root / COSMIC_BEHAVIOR_SELECTION_METRICS_CSV
    generations_path = output_root / COSMIC_BEHAVIOR_SELECTION_GENERATIONS_CSV
    decision_path = output_root / COSMIC_BEHAVIOR_SELECTION_DECISION_CSV
    write_csv(metrics_path, summary_rows)
    write_csv(generations_path, side_by_side_rows)
    write_csv(decision_path, decision_rows)
    print(f"[{time.strftime('%H:%M:%S')}] behavior-first selection metrics: {metrics_path}")
    print(f"[{time.strftime('%H:%M:%S')}] behavior-first selection generations: {generations_path}")
    print(f"[{time.strftime('%H:%M:%S')}] behavior-first selection decision: {decision_path}")

    if selected_row is None:
        print(f"[{time.strftime('%H:%M:%S')}] behavior-first selection found no controller passing gates")
        return None, None, specs_to_select
    by_condition = {spec["condition"]: spec for spec in condition_specs}
    selected_spec = by_condition.get(str(selected_row["condition"]))
    print(
        f"[{time.strftime('%H:%M:%S')}] behavior-first selected controller: "
        f"{selected_row['condition']} rank={selected_row['rank']} "
        f"ablation_scale={selected_row['ablation_scale']} "
        f"score={float(selected_row['behavior_selection_score']):.4f}"
    )
    return selected_spec, selected_row, specs_to_select


def run_behavior_validation(
    model,
    tokenizer,
    condition_specs: Sequence[dict],
    paired_records_by_split: Dict[str, list],
    args: argparse.Namespace,
    output_root: Path,
    specs_to_validate: Optional[Sequence[dict]] = None,
    ablation_scales: Optional[Sequence[float]] = None,
    addition_scales: Optional[Sequence[float]] = None,
) -> None:
    selection_path = Path(args.cosmic_selection_csv) if args.cosmic_selection_csv else output_root / COSMIC_SELECTION_METRICS_CSV
    if specs_to_validate is None:
        cosmic_rows = load_cosmic_selection_rows(selection_path)
        if not cosmic_rows:
            print(f"[{time.strftime('%H:%M:%S')}] behavior validation skipped: missing COSMIC scores at {selection_path}")
            return
        specs_to_validate = select_behavior_validation_specs(condition_specs, cosmic_rows)
    if not specs_to_validate:
        print(f"[{time.strftime('%H:%M:%S')}] behavior validation skipped: no scored condition specs matched {selection_path}")
        return

    heldout_records = paired_records_by_split["heldout"]
    summary_rows, side_by_side_rows, decision_rows = evaluate_behavior_configs(
        model,
        tokenizer,
        specs_to_validate,
        heldout_records,
        args,
        args.behavior_validation_samples,
        12000,
        "heldout",
        ablation_scales if ablation_scales is not None else [1.0],
        addition_scales if addition_scales is not None else [args.addition_scale],
    )

    behavior_metrics_path = output_root / COSMIC_BEHAVIOR_VALIDATION_METRICS_CSV
    generations_path = output_root / COSMIC_GENERATIONS_SIDE_BY_SIDE_CSV
    decision_path = output_root / COSMIC_DIAGNOSTIC_DECISION_CSV
    write_csv(behavior_metrics_path, summary_rows)
    write_csv(generations_path, side_by_side_rows)
    write_csv(decision_path, decision_rows)
    print(f"[{time.strftime('%H:%M:%S')}] behavior validation metrics: {behavior_metrics_path}")
    print(f"[{time.strftime('%H:%M:%S')}] generations side by side: {generations_path}")
    print(f"[{time.strftime('%H:%M:%S')}] diagnostic decision: {decision_path}")


# ## Preflight

def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    paired_csv = Path(args.paired_csv)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"[{time.strftime('%H:%M:%S')}] model: {args.model}")
    print(f"[{time.strftime('%H:%M:%S')}] paired CSV: {paired_csv}")
    print(f"[{time.strftime('%H:%M:%S')}] output root: {output_root}")
    print(
        f"[{time.strftime('%H:%M:%S')}] hardware_profile={args.hardware_profile}; "
        f"dtype={args.dtype}; device_map={args.device_map}; batch_size={args.batch_size}; "
        f"max_memory_gb_per_gpu={args.max_memory_gb_per_gpu}; cpu_memory_gb={args.cpu_memory_gb}"
    )
    print(
        f"[{time.strftime('%H:%M:%S')}] prompt_format: {args.prompt_format}; "
        f"COSMIC scored extracted/control budget: {args.cosmic_max_scored_conditions}/"
        f"{args.cosmic_max_control_conditions_per_type}"
    )

    paired_df = load_paired_csv(paired_csv)
    print_paired_csv_summary(paired_df)

    splits = split_source_indices(paired_df, args)
    split_rows = split_summary_rows(paired_df, splits)
    write_csv(output_root / "data_split_summary.csv", split_rows)
    paired_records_by_split = {
        split_name: paired_records_for_sources(paired_df, source_indices, split_name)
        for split_name, source_indices in splits.items()
    }
    for split_name in ["train", "val", "heldout"]:
        print(
            f"[{time.strftime('%H:%M:%S')}] {split_name} split: "
            f"{len(splits[split_name])} sources; {len(paired_records_by_split[split_name])} paired rows"
        )
    print(f"[{time.strftime('%H:%M:%S')}] data split summary: {output_root / 'data_split_summary.csv'}")

    set_seed(args.seed)
    model, tokenizer = load_model_and_tokenizer(args.model, args)
    input_device = model_input_device(model)
    n_params = sum(param.numel() for param in model.parameters())
    device_map = getattr(model, "hf_device_map", None)
    print(f"[{time.strftime('%H:%M:%S')}] tokenizer vocab size: {len(tokenizer)}")
    print(f"[{time.strftime('%H:%M:%S')}] model parameters: {n_params:,}")
    print(f"[{time.strftime('%H:%M:%S')}] model input device: {input_device}")
    if device_map:
        print(f"[{time.strftime('%H:%M:%S')}] hf_device_map entries: {len(device_map)}")

    positions = parse_int_list(args.post_instruction_positions)
    if args.cosmic_eval_position not in positions:
        raise ValueError(
            f"--cosmic-eval-position={args.cosmic_eval_position} must be one of "
            f"--post-instruction-positions={positions}."
        )
    train_records = paired_records_by_split["train"]
    train_positive_prompts = [record["positive_prompt"] for record in train_records]
    train_negative_prompts = [record["negative_prompt"] for record in train_records]
    print(
        f"[{time.strftime('%H:%M:%S')}] collecting train resid_pre activations: "
        f"{len(train_positive_prompts)} positives and {len(train_negative_prompts)} negatives; positions={positions}"
    )
    positive_acts = collect_resid_pre_by_layer_position(
        model,
        tokenizer,
        train_positive_prompts,
        positions,
        args.batch_size,
        args.max_input_tokens,
        args.allow_truncated_prompts,
        args.prompt_format,
    )
    negative_acts = collect_resid_pre_by_layer_position(
        model,
        tokenizer,
        train_negative_prompts,
        positions,
        args.batch_size,
        args.max_input_tokens,
        args.allow_truncated_prompts,
        args.prompt_format,
    )
    print(f"[{time.strftime('%H:%M:%S')}] positive activation shape: {tuple(positive_acts.shape)}")
    print(f"[{time.strftime('%H:%M:%S')}] negative activation shape: {tuple(negative_acts.shape)}")
    eval_layer_indices, eval_pos_idx, base_layer_similarity_rows = select_low_similarity_eval_layers(
        positive_acts,
        negative_acts,
        positions,
        args,
    )
    print(
        f"[{time.strftime('%H:%M:%S')}] COSMIC eval layers L_low: "
        f"{eval_layer_indices} at position {positions[eval_pos_idx]}"
    )

    candidates = extract_direction_candidates(positive_acts, negative_acts, positions)
    direction_candidates_path = output_root / COSMIC_DIRECTION_CANDIDATES_CSV
    write_csv(direction_candidates_path, [candidate_record_for_csv(candidate) for candidate in candidates])
    print(f"[{time.strftime('%H:%M:%S')}] direction candidates: {len(candidates)}")
    print(f"[{time.strftime('%H:%M:%S')}] direction candidates CSV: {direction_candidates_path}")

    condition_specs = build_candidate_condition_specs(candidates, positive_acts, negative_acts, positions, args)
    condition_specs_path = output_root / COSMIC_CONDITION_SPECS_CSV
    write_csv(condition_specs_path, [condition_spec_record_for_csv(spec) for spec in condition_specs])
    condition_counts = pd.Series([spec["basis_type"] for spec in condition_specs]).value_counts().to_dict()
    print(f"[{time.strftime('%H:%M:%S')}] candidate condition specs: {len(condition_specs)}")
    print(f"[{time.strftime('%H:%M:%S')}] candidate condition counts: {condition_counts}")
    print(f"[{time.strftime('%H:%M:%S')}] candidate condition specs CSV: {condition_specs_path}")

    val_records = paired_records_by_split["val"]
    val_positive_prompts = [record["positive_prompt"] for record in val_records]
    val_negative_prompts = [record["negative_prompt"] for record in val_records]
    print(
        f"[{time.strftime('%H:%M:%S')}] collecting val resid_pre activations for COSMIC scoring: "
        f"{len(val_positive_prompts)} positives and {len(val_negative_prompts)} negatives"
    )
    val_positive_acts = collect_resid_pre_by_layer_position(
        model,
        tokenizer,
        val_positive_prompts,
        positions,
        args.batch_size,
        args.max_input_tokens,
        args.allow_truncated_prompts,
        args.prompt_format,
    )
    val_negative_acts = collect_resid_pre_by_layer_position(
        model,
        tokenizer,
        val_negative_prompts,
        positions,
        args.batch_size,
        args.max_input_tokens,
        args.allow_truncated_prompts,
        args.prompt_format,
    )
    cosmic_rows, layer_similarity_rows, cosmic_stack_specs = score_condition_specs_by_cosmic_intervention(
        model,
        tokenizer,
        condition_specs,
        val_positive_acts,
        val_negative_acts,
        val_positive_prompts,
        val_negative_prompts,
        positions,
        eval_layer_indices,
        eval_pos_idx,
        args,
    )
    if cosmic_stack_specs:
        condition_specs.extend(cosmic_stack_specs)
        write_csv(condition_specs_path, [condition_spec_record_for_csv(spec) for spec in condition_specs])
        print(f"[{time.strftime('%H:%M:%S')}] added COSMIC-ranked stack specs: {len(cosmic_stack_specs)}")
    cosmic_selection_path = output_root / COSMIC_SELECTION_METRICS_CSV
    layer_similarity_path = output_root / COSMIC_EVAL_LAYER_SIMILARITY_CSV
    write_csv(cosmic_selection_path, cosmic_rows)
    write_csv(layer_similarity_path, [*base_layer_similarity_rows, *layer_similarity_rows])
    cosmic_selected_spec, cosmic_selected_row = selected_spec_from_cosmic_rows(condition_specs, cosmic_rows)
    selected_spec = cosmic_selected_spec
    selected_row = cosmic_selected_row
    selected_ablation_scale = 1.0
    selected_addition_scale = args.addition_scale
    validation_specs = None
    print(f"[{time.strftime('%H:%M:%S')}] COSMIC selection metrics CSV: {cosmic_selection_path}")
    print(f"[{time.strftime('%H:%M:%S')}] COSMIC eval layer similarity CSV: {layer_similarity_path}")
    if cosmic_selected_spec is not None:
        print(
            f"[{time.strftime('%H:%M:%S')}] best extracted COSMIC controller: "
            f"{cosmic_selected_spec['condition']} score={float(cosmic_selected_row['cosmic_score']):.4f}"
        )

    if args.selection_mode == "behavior_first":
        behavior_spec, behavior_row, _ = run_behavior_first_selection(
            model,
            tokenizer,
            condition_specs,
            cosmic_rows,
            paired_records_by_split,
            args,
            output_root,
        )
        if behavior_spec is not None and behavior_row is not None:
            selected_spec = behavior_spec
            selected_row = dict(behavior_row)
            selected_row["score_source"] = "behavior_first_validation_generations"
            selected_row["selection_score_source"] = "behavior_first_validation_generations"
            selected_ablation_scale = float(behavior_row.get("ablation_scale", 1.0))
            selected_addition_scale = float(behavior_row.get("addition_scale", args.addition_scale))
            validation_specs = [selected_spec, *matched_control_specs(condition_specs, cosmic_rows, selected_spec)]
        elif cosmic_selected_spec is not None:
            print(f"[{time.strftime('%H:%M:%S')}] falling back to COSMIC-selected heldout validation")

    if selected_spec is not None:
        if not args.skip_selected_artifact:
            artifact_path = save_selected_controller_artifact(selected_spec, selected_row, output_root)
            print(f"[{time.strftime('%H:%M:%S')}] selected controller artifact: {artifact_path}")

    run_behavior_validation(
        model,
        tokenizer,
        condition_specs,
        paired_records_by_split,
        args,
        output_root,
        specs_to_validate=validation_specs,
        ablation_scales=[selected_ablation_scale],
        addition_scales=[selected_addition_scale],
    )

    del positive_acts
    del negative_acts
    del val_positive_acts
    del val_negative_acts

    del model
    del tokenizer
    release_memory()
    print(f"[{time.strftime('%H:%M:%S')}] preflight complete")


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        parse_args()
    else:
        main()
