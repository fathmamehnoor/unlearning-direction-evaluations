#!/usr/bin/env python
"""QLoRA SFT on unrelated GSM8K, saving example-count checkpoints.

This is the *fine-tuning half* of the unrelated-SFT recovery experiment for
WMDP-unlearned models. It fine-tunes one model (base control OR unlearned) on
grade-school math word problems and saves LoRA adapters at fixed
examples-seen thresholds (default 1000 / 3000 / 6000).

Why GSM8K: it has near-zero mutual information with WMDP-bio. A model cannot
learn bioweapons facts from arithmetic word problems, so any WMDP-bio movement
after this SFT is the unlearning coming undone, not new knowledge coming in.
State that disjointness explicitly in the writeup -- "it is grade-school
arithmetic" is the clean answer to "could the SFT data leak bio content?".

The exact same recipe and seed are used for every model (base control and each
unlearning method) so the only thing differing between arms is the checkpoint,
not the fine-tuning. Do not change the recipe between arms.

Outputs (under --output-root/<run-name>/):
  checkpoint-examples-1000/   PEFT adapter after ~1000 examples
  checkpoint-examples-3000/   PEFT adapter after ~3000 examples
  checkpoint-examples-6000/   PEFT adapter after ~6000 examples (== adapter_final)
  run_config.json             the full recipe, for the record

Evaluate each of these with eval_recovery_lm_eval.py by passing the base model
through --model-name and the adapter directory through --adapter-path.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence


# ----------------------------------------------------------------------------
# The single shared recipe. Keep these identical across the base-control arm
# and every unlearned arm -- see the module docstring.
# ----------------------------------------------------------------------------
DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
DEFAULT_CHECKPOINTS = [1000, 3000, 6000]


@dataclass
class RunConfig:
    model_name: str
    model_label: str
    unlearning_method: str
    arm: str
    output_dir: str
    dataset_name: str
    dataset_config: str
    dataset_split: str
    num_examples: int
    seed: int
    max_length: int
    epochs: float
    learning_rate: float
    per_device_batch_size: int
    gradient_accumulation_steps: int
    warmup_ratio: float
    weight_decay: float
    max_grad_norm: float
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: List[str]
    checkpoint_examples: List[int]
    bf16: bool


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_slug(value: str) -> str:
    import re

    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return slug.strip("_") or "model"


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ModuleNotFoundError:
        pass
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-name", required=True, help="Base control or unlearned model repo/path to fine-tune.")
    parser.add_argument("--model-label", default=None, help="Short label used in output paths and metadata.")
    parser.add_argument("--unlearning-method", default="none", help="IDK-AP, ILU-RMU, or 'none' for the base control.")
    parser.add_argument("--arm", default="unlearned", choices=["unlearned", "full_knowledge_control"])
    parser.add_argument("--output-root", type=Path, default=Path("results/wmdp_sft_recovery/adapters"))
    parser.add_argument("--run-name", default=None)

    parser.add_argument("--dataset-name", default="openai/gsm8k")
    parser.add_argument("--dataset-config", default="main")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--num-examples", type=int, default=6000)
    parser.add_argument("--checkpoint-examples", nargs="*", type=int, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=10)

    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", nargs="*", default=DEFAULT_TARGET_MODULES)

    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false", default=True)
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        print("Ignoring unknown launcher arguments:", unknown)
    return args


# ----------------------------------------------------------------------------
# Data: GSM8K formatted as a single-turn chat, completion-only loss (the
# question tokens are masked so the model only trains on producing the answer).
# ----------------------------------------------------------------------------
def load_gsm8k_records(name: str, config: str, split: str, num_examples: int, seed: int):
    from datasets import load_dataset

    ds = load_dataset(name, config, split=split)
    ds = ds.shuffle(seed=seed)
    if num_examples and num_examples < len(ds):
        ds = ds.select(range(num_examples))
    records = [{"question": r["question"], "answer": r["answer"]} for r in ds]
    metadata = {
        "dataset_name": name,
        "dataset_config": config,
        "split": split,
        "requested": num_examples,
        "selected": len(records),
        "shuffle_seed": seed,
    }
    return records, metadata


def build_tokenized_dataset(tokenizer, records, max_length: int):
    from datasets import Dataset

    # Plain, template-independent SFT format. We deliberately do NOT use
    # tokenizer.apply_chat_template: several of these unlearned checkpoints ship
    # a broken/placeholder chat template that renders message content to nothing
    # (e.g. ScaleAI/mhj-llama3-8b-rmu returns ~2 tokens regardless of input),
    # which would silently produce an all-masked, empty training set. A fixed
    # "Question:/Answer:" format tokenizes correctly on any causal-LM tokenizer
    # and, as a bonus, keeps the SFT recipe identical across every model/arm.
    def encode(record):
        user = str(record["question"]).strip()
        assistant = str(record["answer"]).strip()
        prompt_text = f"Question: {user}\nAnswer:"
        full_text = f"{prompt_text} {assistant}"
        prompt_ids = tokenizer(prompt_text, add_special_tokens=True)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=True)["input_ids"]
        if tokenizer.eos_token_id is not None:
            full_ids = full_ids + [tokenizer.eos_token_id]
        prompt_len = len(prompt_ids)
        if full_ids[:prompt_len] != prompt_ids:
            prompt_len = min(prompt_len, len(full_ids))
        labels = [-100] * prompt_len + full_ids[prompt_len:]
        full_ids = full_ids[:max_length]
        labels = labels[:max_length]
        return {"input_ids": full_ids, "labels": labels, "attention_mask": [1] * len(full_ids)}

    encoded = [encode(r) for r in records]
    # Drop any example that got fully truncated past its completion.
    encoded = [e for e in encoded if any(t != -100 for t in e["labels"])]
    if not encoded:
        raise RuntimeError(
            "Tokenized training set is EMPTY -- every example produced all-masked labels. "
            "Check the tokenizer and max_length; the SFT cannot proceed with 0 examples."
        )
    return Dataset.from_list(encoded)


class CompletionCollator:
    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def __call__(self, features):
        import torch

        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, labels, attn = [], [], []
        for f in features:
            pad = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_id] * pad)
            labels.append(f["labels"] + [-100] * pad)
            attn.append(f["attention_mask"] + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


# ----------------------------------------------------------------------------
# Checkpoint-by-examples-seen callback. Prior relearning work finds most of the
# recovery happens very early (first few hundred to thousand examples), so we
# snapshot the adapter at example counts, not just at the end.
# ----------------------------------------------------------------------------
def make_checkpoint_callback(output_dir: Path, tokenizer, thresholds, per_device_bs: int, grad_accum: int):
    from transformers import TrainerCallback

    saved = set()
    thresholds = sorted({int(t) for t in thresholds if int(t) > 0})

    class _Callback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            model = kwargs.get("model")
            if model is None:
                return control
            world = int(getattr(args, "world_size", 1) or 1)
            seen = int(state.global_step * per_device_bs * grad_accum * world)
            for threshold in thresholds:
                if threshold in saved or seen < threshold:
                    continue
                ckpt_dir = output_dir / f"checkpoint-examples-{threshold}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(str(ckpt_dir))
                tokenizer.save_pretrained(str(ckpt_dir))
                write_json(
                    ckpt_dir / "checkpoint_metadata.json",
                    {
                        "saved_at_utc": utc_now(),
                        "threshold_examples": threshold,
                        "estimated_examples_seen": seen,
                        "global_step": int(state.global_step),
                    },
                )
                saved.add(threshold)
            return control

    return _Callback()


def load_model(model_name: str, bf16: bool, trust_remote_code: bool, gradient_checkpointing: bool):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    compute_dtype = torch.bfloat16 if bf16 else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=compute_dtype,
        device_map={"": 0} if torch.cuda.is_available() else None,
        trust_remote_code=trust_remote_code,
    )
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return model


def add_lora(model, args, gradient_checkpointing: bool):
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        bias="none",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def main() -> None:
    args = parse_args()  # parse first so --help works without torch/transformers

    import torch
    from transformers import AutoTokenizer, Trainer, TrainingArguments

    set_seed(args.seed)
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    model_label = args.model_label or safe_slug(args.model_name)
    run_name = args.run_name or f"{safe_slug(model_label)}_gsm8k{args.num_examples}_seed{args.seed}"
    output_dir = args.output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = RunConfig(
        model_name=args.model_name,
        model_label=model_label,
        unlearning_method=args.unlearning_method,
        arm=args.arm,
        output_dir=str(output_dir),
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
        num_examples=args.num_examples,
        seed=args.seed,
        max_length=args.max_length,
        epochs=args.epochs,
        learning_rate=args.lr,
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=list(args.target_modules),
        checkpoint_examples=list(args.checkpoint_examples),
        bf16=bf16,
    )
    write_json(output_dir / "run_config.json", asdict(cfg))
    print(json.dumps({"event": "start", "config": asdict(cfg)}, indent=2))

    records, data_meta = load_gsm8k_records(
        args.dataset_name, args.dataset_config, args.dataset_split, args.num_examples, args.seed
    )
    write_json(output_dir / "sft_dataset_metadata.json", data_meta)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_dataset = build_tokenized_dataset(tokenizer, records, args.max_length)
    print(f"tokenized {len(train_dataset)} usable GSM8K examples")

    model = load_model(args.model_name, bf16, args.trust_remote_code, args.gradient_checkpointing)
    model = add_lora(model, args, args.gradient_checkpointing)

    callback = make_checkpoint_callback(
        output_dir, tokenizer, args.checkpoint_examples, args.batch_size, args.grad_accum
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir / "trainer_state"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        bf16=bf16,
        fp16=torch.cuda.is_available() and not bf16,
        logging_steps=args.logging_steps,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        optim="paged_adamw_8bit",
        gradient_checkpointing=args.gradient_checkpointing,
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=CompletionCollator(tokenizer),
        callbacks=[callback],
    )
    train_output = trainer.train()
    write_json(output_dir / "train_metrics.json", train_output.metrics)

    # Always persist a final adapter, and mirror it into the top checkpoint
    # threshold so downstream eval can address it as checkpoint-examples-<N>.
    final_dir = output_dir / "adapter_final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    top = max(args.checkpoint_examples) if args.checkpoint_examples else args.num_examples
    world = int(getattr(trainer.args, "world_size", 1) or 1)
    final_examples_seen = int(trainer.state.global_step * args.batch_size * args.grad_accum * world)
    # Only claim the top threshold if this run actually requested at least that
    # many examples; otherwise the final adapter saw fewer than `top` and must
    # not be mislabeled as checkpoint-examples-<top>. Fall back to the true run
    # size. Either way the real examples-seen is recorded in metadata.
    if args.num_examples >= top:
        mirror_threshold = top
    else:
        mirror_threshold = args.num_examples
        print(
            f"WARNING: num_examples ({args.num_examples}) < max checkpoint threshold ({top}); "
            f"labeling the final-adapter mirror as checkpoint-examples-{mirror_threshold} "
            f"(actual examples seen ~{final_examples_seen}), not {top}."
        )
    top_dir = output_dir / f"checkpoint-examples-{mirror_threshold}"
    if not top_dir.exists():
        trainer.save_model(str(top_dir))
        tokenizer.save_pretrained(str(top_dir))
        write_json(
            top_dir / "checkpoint_metadata.json",
            {
                "saved_at_utc": utc_now(),
                "threshold_examples": mirror_threshold,
                "num_examples_requested": args.num_examples,
                "estimated_examples_seen": final_examples_seen,
                "is_final_copy": True,
            },
        )

    summary = {
        "created_at_utc": utc_now(),
        "model_name": args.model_name,
        "model_label": model_label,
        "unlearning_method": args.unlearning_method,
        "arm": args.arm,
        "adapter_final": str(final_dir),
        "checkpoint_dirs": [
            str(output_dir / f"checkpoint-examples-{t}")
            for t in sorted(set(args.checkpoint_examples))
            if (output_dir / f"checkpoint-examples-{t}").exists()
        ],
        "train_metrics": train_output.metrics,
    }
    write_json(output_dir / "train_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
