"""Collect TOFU refusal/IDK prompt pairs for a single unlearned model.

The raw CSV stores every requested prompt variant. With
``--write-paired-output``, the script also writes matched positive
modified-IDK prompts and negative plain-QA prompts for controller extraction.
Rows are written incrementally so interrupted long runs still leave usable
artifacts.
"""

from __future__ import annotations

import argparse
import csv
import gc
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ModuleNotFoundError:
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None


DEFAULT_DATASET = "locuslab/TOFU"
DEFAULT_FORGET_CONFIG = "forget10"
DEFAULT_MODEL = ""
DEFAULT_OUTPUT_CSV = "outputs/refusal_prompts.csv"
DEFAULT_PAIRED_OUTPUT_CSV = "outputs/paired_idk_plain_prompts.csv"
HARDWARE_PROFILES = {
    "a100": {
        "batch_size": 8,
        "dtype": "bfloat16",
        "max_memory_gb_per_gpu": 72.0,
        "cpu_memory_gb": 64.0,
    },
}

PROMPT_VARIANTS = {
    "plain_qa": "Question: {question}\nAnswer:",
    "idk_allowed": "Question: {question}\nIf you do not know the answer, say \"I don't know.\"\nAnswer:",
    "idk_exact": "Question: {question}\nIf you are not certain, answer exactly: \"I don't know.\"\nAnswer:",
    "unknown_allowed": "Question: {question}\nIf the answer is unknown or unavailable, say \"I don't know.\"\nAnswer:",
    "answer_or_idk": "Question: {question}\nGive the answer if you know it. Otherwise, say \"I don't know.\"\nAnswer:",
    "no_guessing": "Question: {question}\nDo not guess. If you cannot answer confidently, say \"I don't know.\"\nAnswer:",
}

MODIFIED_IDK_PROMPT_VARIANTS = [name for name in PROMPT_VARIANTS if name != "plain_qa"]

REFUSAL_PATTERNS: List[Tuple[str, str]] = [
    # Direct IDK
    ("direct_idk", r"\bi do not know\b"),
    ("direct_idk", r"\bi don'?t know\b"),
    ("direct_idk", r"\bi have no knowledge of\b"),
    ("direct_idk", r"\bi have no information about\b"),
    ("direct_idk", r"\bi don'?t have information about\b"),

    # Uncertainty
    ("uncertain", r"\bi am not sure\b"),
    ("uncertain", r"\bi'?m not sure\b"),
    ("uncertain", r"\bnot sure\b"),
    ("uncertain", r"\buncertain\b"),
    ("uncertain", r"\bi am unsure\b"),
    ("uncertain", r"\bi'?m unsure\b"),
    ("uncertain", r"\bi am stumped\b"),
    ("uncertain", r"\bi'?m stumped\b"),
    ("uncertain", r"\bstumped on\b"),
    ("uncertain", r"\bi am blank\b"),
    ("uncertain", r"\bi'?m blank\b"),
    ("uncertain", r"\bblank on\b"),

    # Unable
    ("unable", r"\bi cannot\b"),
    ("unable", r"\bi can'?t\b"),
    ("unable", r"\bi am unable\b"),
    ("unable", r"\bi'?m unable\b"),

    # No information
    ("no_info", r"\bi do not have\b"),
    ("no_info", r"\bi don'?t have\b"),
    ("no_info", r"\bno information\b"),
    ("no_info", r"\bno public information\b"),
    ("no_info", r"\bthere are no public information\b"),
    ("no_info", r"\bthere is no information\b"),
    ("no_info", r"\bis not publicly known\b"),
    ("no_info", r"\bis not publicly available\b"),
    ("no_info", r"\bnot publicly known\b"),
    ("no_info", r"\bnot publicly available\b"),
    ("no_info", r"\bnot enough information\b"),
    ("no_info", r"\binsufficient information\b"),
    ("no_info", r"\bbest of my knowledge base doesn'?t include information on that topic\b"),

    # Unknown
    ("unknown", r"\bunknown\b"),
    ("unknown", r"\bnot provided\b"),
    ("unknown", r"\bnot available\b"),

    # Unfamiliar
    ("unfamiliar", r"\bi am not informed\b"),
    ("unfamiliar", r"\bi'?m not informed\b"),
    ("unfamiliar", r"\bnot informed about\b"),
    ("unfamiliar", r"\bnot familiar with\b"),
    ("unfamiliar", r"\bnot something i'?m familiar with\b"),
    ("unfamiliar", r"\bnot information i'?m familiar with\b"),
    ("unfamiliar", r"\bis not something i'?m acquainted with\b"),
    ("unfamiliar", r"\bnot something i'?m acquainted with\b"),

    # Lookup needed
    ("lookup", r"\bneed to look up\b"),
    ("lookup", r"\bi need to look up\b"),
    ("lookup", r"\bthat'?s something i need to look up\b"),
    ("lookup", r"\bcan'?t find any information\b"),
    ("lookup", r"\bcannot find any information\b"),
    ("lookup", r"\breaders can'?t find any information\b"),

    # Refusal / abstention
    ("refusal", r"\bnot at liberty to disclose\b"),
    ("refusal", r"\bi am not at liberty\b"),
    ("refusal", r"\bi'?m not at liberty\b"),
    ("refusal", r"\bcannot be answered\b"),
    ("refusal", r"\bcan'?t be answered\b"),
]





CSV_COLUMNS = [
    "model",
    "method",
    "dataset",
    "forget_config",
    "sample_id",
    "source_index",
    "prompt_variant",
    "question",
    "answer",
    "paraphrased_answer",
    "prompt",
    "generation",
    "refusal_family",
    "refusal_match",
    "is_degenerate",
    "num_new_chars",
    "num_new_words",
    "seed",
    "temperature",
    "max_new_tokens",
]

PAIRED_CSV_COLUMNS = [
    "model",
    "method",
    "dataset",
    "forget_config",
    "source_index",
    "sample_id",
    "question",
    "answer",
    "paraphrased_answer",
    "positive_prompt_variant",
    "positive_prompt",
    "positive_generation",
    "positive_refusal_family",
    "positive_refusal_match",
    "negative_prompt_variant",
    "negative_prompt",
    "negative_generation",
    "seed",
    "temperature",
    "max_new_tokens",
    "positive_num_new_words",
    "negative_num_new_words",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, required=not bool(DEFAULT_MODEL), help="Hugging Face model id or local checkpoint path.")
    parser.add_argument("--method", default="", help="Optional label written to the output CSV. Defaults to a label inferred from --model.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--forget-config", default=DEFAULT_FORGET_CONFIG)
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--paired-output-csv", default=DEFAULT_PAIRED_OUTPUT_CSV)
    parser.add_argument("--write-paired-output", action="store_true", help="Write matched modified-IDK positive vs plain-QA negative pairs.")
    parser.add_argument("--paired-min-negative-words", type=int, default=3, help="Minimum plain-QA generation length for a non-IDK negative pair.")
    parser.add_argument("--max-samples", type=int, default=-1, help="Use -1 for the full forget split.")
    parser.add_argument("--prompt-variants", nargs="*", default=list(PROMPT_VARIANTS.keys()))
    parser.add_argument("--save-all", action="store_true", default=True, help="Save every generation, not only refusal/IDK matches.")
    parser.add_argument("--resume", action="store_true", help="Skip model/sample/variant rows already present in output CSV.")
    parser.add_argument("--overwrite", action="store_true", help="Replace the output CSV before running.")
    parser.add_argument("--hardware-profile", default="a100", choices=["a100", "manual"], help="Default memory/dtype/batch profile.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dtype", default=None, choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-memory-gb-per-gpu", type=float, default=None)
    parser.add_argument("--cpu-memory-gb", type=float, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    args, _ = parser.parse_known_args()
    if not args.model:
        parser.error("--model is required")
    apply_hardware_profile(args)
    return args


def apply_hardware_profile(args: argparse.Namespace) -> None:
    if args.hardware_profile == "manual":
        if args.batch_size is None:
            args.batch_size = 8
        if args.dtype is None:
            args.dtype = "bfloat16"
        if args.max_memory_gb_per_gpu is None:
            args.max_memory_gb_per_gpu = 72.0
        if args.cpu_memory_gb is None:
            args.cpu_memory_gb = 64.0
        return

    profile = HARDWARE_PROFILES[args.hardware_profile]
    for key, value in profile.items():
        if getattr(args, key) is None:
            setattr(args, key, value)


def set_seed(seed: int) -> None:
    require_model_dependencies()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dtype_from_name(name: str) -> torch.dtype:
    require_model_dependencies()
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def get_max_memory(args: argparse.Namespace) -> Optional[dict]:
    require_model_dependencies()
    if args.device_map != "auto" or not torch.cuda.is_available():
        return None
    max_memory = {idx: f"{args.max_memory_gb_per_gpu}GiB" for idx in range(torch.cuda.device_count())}
    max_memory["cpu"] = f"{args.cpu_memory_gb}GiB"
    return max_memory


def normalize_device(device) -> Optional[torch.device]:
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


def model_device(model) -> torch.device:
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


def require_model_dependencies() -> None:
    missing = []
    if torch is None:
        missing.append("torch")
    if AutoModelForCausalLM is None or AutoTokenizer is None:
        missing.append("transformers")
    if missing:
        names = ", ".join(sorted(set(missing)))
        raise ModuleNotFoundError(f"Missing required model-loading dependencies: {names}")


def inference_mode():
    if torch is None:
        return lambda fn: fn
    return torch.inference_mode()


def infer_method_label(model_name: str) -> str:
    label = Path(str(model_name).rstrip("/")).name
    return label or "unknown"


def first_present(row: dict, names: Sequence[str], default=None):
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return default


def load_tofu_records(args: argparse.Namespace) -> List[dict]:
    ds = load_dataset(args.dataset, args.forget_config)
    split = ds["train"] if "train" in ds else ds[next(iter(ds.keys()))]
    rows = [dict(row) for row in split]
    if args.max_samples and args.max_samples > 0:
        rows = rows[:args.max_samples]

    records = []
    for source_index, row in enumerate(rows):
        question = first_present(row, ["question", "prompt"])
        answer = first_present(row, ["answer", "completion", "response"])
        if not question or not answer:
            continue
        paraphrased = first_present(row, ["paraphrased_answer", "paraphrase", "paraphrased"])
        records.append({
            "sample_id": f"{args.forget_config}_{source_index}",
            "source_index": source_index,
            "question": str(question),
            "answer": str(answer),
            "paraphrased_answer": str(paraphrased) if paraphrased else "",
        })
    return records


def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_refusal_or_idk(text: str) -> Tuple[bool, str, str]:
    norm = normalize_text(text)
    for family, pattern in REFUSAL_PATTERNS:
        match = re.search(pattern, norm)
        if match:
            return True, family, match.group(0)
    return False, "", ""


def is_degenerate(text: str) -> bool:
    norm = normalize_text(text)
    if not norm:
        return True
    words = norm.split()
    if len(words) <= 2:
        return True
    unique_ratio = len(set(words)) / max(1, len(words))
    if len(words) >= 20 and unique_ratio < 0.30:
        return True
    if re.search(r"\b(\w+)(?:\s+\1\b){3,}", norm):
        return True
    return False


def render_prompt(variant: str, question: str) -> str:
    if variant not in PROMPT_VARIANTS:
        raise ValueError(f"Unknown prompt variant: {variant}")
    return PROMPT_VARIANTS[variant].format(question=question)


def load_completed_keys(path: Path) -> set:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    df = pd.read_csv(path, usecols=["model", "sample_id", "prompt_variant"])
    return set(zip(df["model"].astype(str), df["sample_id"].astype(str), df["prompt_variant"].astype(str)))


def ensure_csv(path: Path, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite and path.exists():
        path.unlink()
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()


def append_csv_rows(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in CSV_COLUMNS})


def bool_from_value(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def int_from_value(value, default: int = 0) -> int:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def clean_string(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value)


def is_valid_plain_negative(row: dict, min_words: int) -> bool:
    if clean_string(row.get("prompt_variant")) != "plain_qa":
        return False
    if clean_string(row.get("refusal_family")).strip():
        return False
    if bool_from_value(row.get("is_degenerate")):
        return False
    return int_from_value(row.get("num_new_words")) >= min_words


def is_valid_modified_positive(row: dict) -> bool:
    if clean_string(row.get("prompt_variant")) == "plain_qa":
        return False
    if not clean_string(row.get("refusal_family")).strip():
        return False
    return not bool_from_value(row.get("is_degenerate"))


def write_paired_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PAIRED_CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in PAIRED_CSV_COLUMNS})


def build_paired_rows(raw_csv: Path, args: argparse.Namespace) -> Tuple[List[dict], dict]:
    if not raw_csv.exists() or raw_csv.stat().st_size == 0:
        return [], {
            "total_sources": 0,
            "sources_with_modified_idk": 0,
            "sources_with_plain_non_idk": 0,
            "paired_sources": 0,
            "paired_rows": 0,
            "positive_variant_counts": {},
        }

    df = pd.read_csv(raw_csv)
    if df.empty:
        return [], {
            "total_sources": 0,
            "sources_with_modified_idk": 0,
            "sources_with_plain_non_idk": 0,
            "paired_sources": 0,
            "paired_rows": 0,
            "positive_variant_counts": {},
        }

    for required in CSV_COLUMNS:
        if required not in df.columns:
            raise ValueError(f"Raw CSV is missing required column for pairing: {required}")

    df = df[
        (df["model"].astype(str) == str(args.model))
        & (df["dataset"].astype(str) == str(args.dataset))
        & (df["forget_config"].astype(str) == str(args.forget_config))
    ].copy()
    if df.empty:
        return [], {
            "total_sources": 0,
            "sources_with_modified_idk": 0,
            "sources_with_plain_non_idk": 0,
            "paired_sources": 0,
            "paired_rows": 0,
            "positive_variant_counts": {},
        }

    df = df.drop_duplicates(["source_index", "prompt_variant"], keep="last")
    row_dicts = df.to_dict("records")
    total_sources = len({int_from_value(row.get("source_index"), -1) for row in row_dicts})
    positives_by_source: Dict[int, List[dict]] = {}
    plain_by_source: Dict[int, dict] = {}

    for row in row_dicts:
        source_index = int_from_value(row.get("source_index"), -1)
        if source_index < 0:
            continue
        if is_valid_plain_negative(row, args.paired_min_negative_words):
            plain_by_source[source_index] = row
        if is_valid_modified_positive(row):
            positives_by_source.setdefault(source_index, []).append(row)

    paired_rows = []
    variant_counts: Dict[str, int] = {}
    for source_index in sorted(set(positives_by_source) & set(plain_by_source)):
        negative = plain_by_source[source_index]
        for positive in sorted(positives_by_source[source_index], key=lambda row: clean_string(row.get("prompt_variant"))):
            variant = clean_string(positive.get("prompt_variant"))
            variant_counts[variant] = variant_counts.get(variant, 0) + 1
            paired_rows.append({
                "model": clean_string(positive.get("model")),
                "method": clean_string(positive.get("method")),
                "dataset": clean_string(positive.get("dataset")),
                "forget_config": clean_string(positive.get("forget_config")),
                "source_index": source_index,
                "sample_id": clean_string(positive.get("sample_id")),
                "question": clean_string(positive.get("question")),
                "answer": clean_string(positive.get("answer")),
                "paraphrased_answer": clean_string(positive.get("paraphrased_answer")),
                "positive_prompt_variant": variant,
                "positive_prompt": clean_string(positive.get("prompt")),
                "positive_generation": clean_string(positive.get("generation")),
                "positive_refusal_family": clean_string(positive.get("refusal_family")),
                "positive_refusal_match": clean_string(positive.get("refusal_match")),
                "negative_prompt_variant": "plain_qa",
                "negative_prompt": clean_string(negative.get("prompt")),
                "negative_generation": clean_string(negative.get("generation")),
                "seed": clean_string(positive.get("seed")),
                "temperature": clean_string(positive.get("temperature")),
                "max_new_tokens": clean_string(positive.get("max_new_tokens")),
                "positive_num_new_words": int_from_value(positive.get("num_new_words")),
                "negative_num_new_words": int_from_value(negative.get("num_new_words")),
            })

    summary = {
        "total_sources": total_sources,
        "sources_with_modified_idk": len(positives_by_source),
        "sources_with_plain_non_idk": len(plain_by_source),
        "paired_sources": len(set(positives_by_source) & set(plain_by_source)),
        "paired_rows": len(paired_rows),
        "positive_variant_counts": variant_counts,
    }
    return paired_rows, summary


def write_and_report_paired_output(raw_csv: Path, paired_csv: Path, args: argparse.Namespace) -> None:
    paired_rows, summary = build_paired_rows(raw_csv, args)
    write_paired_csv(paired_csv, paired_rows)
    print(f"[{time.strftime('%H:%M:%S')}] paired output CSV: {paired_csv}")
    print(
        f"[{time.strftime('%H:%M:%S')}] paired summary: "
        f"total_sources={summary['total_sources']}; "
        f"sources_with_modified_idk={summary['sources_with_modified_idk']}; "
        f"sources_with_plain_non_idk={summary['sources_with_plain_non_idk']}; "
        f"paired_sources={summary['paired_sources']}; paired_rows={summary['paired_rows']}"
    )
    print(f"[{time.strftime('%H:%M:%S')}] positive variant counts: {summary['positive_variant_counts']}")
    for idx, row in enumerate(paired_rows[:5], start=1):
        print(f"\n[paired example {idx}] source_index={row['source_index']} positive_variant={row['positive_prompt_variant']}")
        print(f"Q: {row['question']}")
        print(f"positive: {row['positive_generation']}")
        print(f"negative: {row['negative_generation']}")


def load_model_and_tokenizer(model_name: str, args: argparse.Namespace):
    require_model_dependencies()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=args.trust_remote_code, use_fast=True)
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


def encode_batch(tokenizer, prompts: Sequence[str], device: torch.device):
    return tokenizer(list(prompts), return_tensors="pt", padding=True, truncation=True).to(device)


@inference_mode()
def generate_prompt_rows(model, tokenizer, work_items: Sequence[dict], args: argparse.Namespace) -> List[dict]:
    rows = []
    method = args.method or infer_method_label(args.model)
    prompts = [item["prompt"] for item in work_items]
    do_sample = args.temperature > 0
    for start in range(0, len(work_items), args.batch_size):
        batch_items = work_items[start:start + args.batch_size]
        batch_prompts = prompts[start:start + args.batch_size]
        encoded = encode_batch(tokenizer, batch_prompts, model_device(model))
        generation_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = args.temperature
        generated = model.generate(**encoded, **generation_kwargs)
        prompt_len = encoded["input_ids"].shape[1]
        texts = tokenizer.batch_decode(generated[:, prompt_len:], skip_special_tokens=True)

        for item, text in zip(batch_items, texts):
            generation = text.strip()
            matched, family, matched_text = detect_refusal_or_idk(generation)
            degenerate = is_degenerate(generation)
            if matched or args.save_all:
                rows.append({
                    "model": args.model,
                    "method": method,
                    "dataset": args.dataset,
                    "forget_config": args.forget_config,
                    "sample_id": item["sample_id"],
                    "source_index": item["source_index"],
                    "prompt_variant": item["prompt_variant"],
                    "question": item["question"],
                    "answer": item["answer"],
                    "paraphrased_answer": item["paraphrased_answer"],
                    "prompt": item["prompt"],
                    "generation": generation,
                    "refusal_family": family,
                    "refusal_match": matched_text,
                    "is_degenerate": degenerate,
                    "num_new_chars": len(generation),
                    "num_new_words": len(normalize_text(generation).split()),
                    "seed": args.seed,
                    "temperature": args.temperature,
                    "max_new_tokens": args.max_new_tokens,
                })
    return rows


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.write_paired_output:
        args.prompt_variants = ["plain_qa", *MODIFIED_IDK_PROMPT_VARIANTS]
        args.save_all = True

    output_csv = Path(args.output_csv)
    paired_output_csv = Path(args.paired_output_csv)
    ensure_csv(output_csv, overwrite=args.overwrite)

    unknown_variants = [name for name in args.prompt_variants if name not in PROMPT_VARIANTS]
    if unknown_variants:
        raise ValueError(f"Unknown prompt variants: {unknown_variants}")

    completed = load_completed_keys(output_csv) if args.resume else set()
    records = load_tofu_records(args)
    work_items = []
    for record in records:
        for variant in args.prompt_variants:
            key = (args.model, record["sample_id"], variant)
            if key in completed:
                continue
            work_items.append({
                **record,
                "prompt_variant": variant,
                "prompt": render_prompt(variant, record["question"]),
            })

    print(f"[{time.strftime('%H:%M:%S')}] model: {args.model}")
    print(
        f"[{time.strftime('%H:%M:%S')}] hardware_profile={args.hardware_profile}; "
        f"dtype={args.dtype}; device_map={args.device_map}; batch_size={args.batch_size}; "
        f"max_memory_gb_per_gpu={args.max_memory_gb_per_gpu}; cpu_memory_gb={args.cpu_memory_gb}"
    )
    print(f"[{time.strftime('%H:%M:%S')}] records: {len(records)} {args.forget_config} rows")
    print(f"[{time.strftime('%H:%M:%S')}] work items: {len(work_items)} prompt variants")
    print(f"[{time.strftime('%H:%M:%S')}] output CSV: {output_csv}")
    if args.write_paired_output:
        print(f"[{time.strftime('%H:%M:%S')}] paired mode: enabled; raw CSV saves all rows")
        print(f"[{time.strftime('%H:%M:%S')}] paired output CSV: {paired_output_csv}")

    if not work_items:
        print("Nothing to do; all requested rows were already present.")
        if args.write_paired_output:
            write_and_report_paired_output(output_csv, paired_output_csv, args)
        return

    model, tokenizer = load_model_and_tokenizer(args.model, args)
    total_saved = 0
    total_seen = 0
    try:
        for start in range(0, len(work_items), args.batch_size):
            batch_items = work_items[start:start + args.batch_size]
            rows = generate_prompt_rows(model, tokenizer, batch_items, args)
            append_csv_rows(output_csv, rows)
            total_seen += len(batch_items)
            total_saved += len(rows)
            if total_seen % max(args.batch_size * 10, 10) == 0 or total_seen == len(work_items):
                print(
                    f"[{time.strftime('%H:%M:%S')}] processed {total_seen}/{len(work_items)}; "
                    f"saved {total_saved} rows"
                )
    finally:
        del model
        release_memory()

    print(f"[{time.strftime('%H:%M:%S')}] complete; saved {total_saved} CSV rows to {output_csv}")
    if args.write_paired_output:
        write_and_report_paired_output(output_csv, paired_output_csv, args)


if __name__ == "__main__":
    main()
