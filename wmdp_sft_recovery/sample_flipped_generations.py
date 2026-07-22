"""Sample held-out WMDP-bio questions whose argmax flipped wrong-at-baseline
-> right-after-SFT, and generate a FREE-FORM (not forced-choice) completion
from the SFT-recovered model on each, for manual reading.

Why this exists: lm_eval's wmdp_bio task scores a model by comparing the
loglikelihood it assigns to each answer choice -- the model never actually
generates anything, so "accuracy went up" tells you the argmax flipped, not
WHY. For RMU specifically, this matters more than for other unlearning
methods: RMU's mechanism is injecting noise into internal representations for
hazardous topics, so a live failure mode is that SFT "recovery" could be an
artifact -- e.g. SFT shifts output calibration/formatting in a way that
happens to move the argmax onto the correct letter more often, without the
model actually reasoning about the (still-noised) biology content. A model
that got noticeably MORE fluent from SFT but is still substantively
confabulating on these questions would show exactly this pattern in the
accuracy number alone. The only way to catch that is to read actual free-form
answers on the specific questions that flipped, by hand.

This script does only that: identify the flipped doc_ids from two already-run
eval passes' persisted per_doc_correctness/wmdp_bio.json files (baseline and
SFT), sample N of them, and generate open-ended completions from the SFT
model for a human to read -- it does not compute any metric itself.

Usage:

    python sample_flipped_generations.py \\
      --model-name ScaleAI/mhj-llama3-8b-rmu \\
      --adapter-path results/wmdp_sft_recovery/adapters/rmu_llama3_8b_gsm8k6000_seed42/checkpoint-examples-6000 \\
      --baseline-correctness results/wmdp_sft_recovery/eval/rmu_llama3_8b_step0/per_doc_correctness/wmdp_bio.json \\
      --sft-correctness results/wmdp_sft_recovery/eval/rmu_llama3_8b_gsm8k_6000_step6000/per_doc_correctness/wmdp_bio.json \\
      --num-samples 10 \\
      --output-jsonl results/wmdp_sft_recovery/rmu_flipped_generations.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
except ModuleNotFoundError:
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    BitsAndBytesConfig = None

try:
    from peft import PeftModel
except ModuleNotFoundError:
    PeftModel = None

try:
    from lm_eval.tasks import TaskManager, get_task_dict
except ModuleNotFoundError:
    TaskManager = None
    get_task_dict = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-name", required=True, help="Base model repo/path (same one the adapter was trained on top of).")
    parser.add_argument("--adapter-path", required=True, help="The SFT LoRA checkpoint to read generations from.")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--baseline-correctness", required=True, help="per_doc_correctness/wmdp_bio.json from the no-SFT baseline run.")
    parser.add_argument("--sft-correctness", required=True, help="per_doc_correctness/wmdp_bio.json from the SFT-checkpoint run.")
    parser.add_argument("--task", default="wmdp_bio")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--load-in-4bit", action="store_true", help="Match the QLoRA training precision.")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--output-jsonl", required=True)
    return parser.parse_args()


def require_dependencies() -> None:
    missing = []
    if torch is None:
        missing.append("torch")
    if AutoModelForCausalLM is None:
        missing.append("transformers")
    if PeftModel is None:
        missing.append("peft")
    if TaskManager is None or get_task_dict is None:
        missing.append("lm_eval")
    if missing:
        raise ModuleNotFoundError(f"Missing required dependencies: {', '.join(missing)}")


def load_correctness(path: str) -> Dict[str, bool]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_flipped_doc_ids(baseline: Dict[str, bool], sft: Dict[str, bool]) -> List[str]:
    shared = sorted(set(baseline) & set(sft), key=lambda k: (len(k), k))
    flipped = [k for k in shared if (baseline[k] is False) and (sft[k] is True)]
    if not flipped:
        raise RuntimeError("No doc_ids flipped wrong-at-baseline -> right-after-SFT between these two runs -- "
                            "nothing to sample. Double check --baseline-correctness / --sft-correctness point at "
                            "the right runs (and that the SFT run's accuracy is actually higher).")
    return flipped


def load_task_docs(task_name: str) -> List[dict]:
    """Reload wmdp_bio's own documents (question/choices/answer), in the same
    deterministic order lm_eval iterates them in -- doc_id in the persisted
    per_doc_correctness files is the positional index into this same list."""
    task_manager = TaskManager()
    task_dict = get_task_dict([task_name], task_manager)
    task = task_dict[task_name]
    if hasattr(task, "test_docs") and task.has_test_docs():
        return list(task.test_docs())
    if hasattr(task, "validation_docs") and task.has_validation_docs():
        return list(task.validation_docs())
    raise RuntimeError(f"Task '{task_name}' has neither test_docs nor validation_docs -- can't recover doc text.")


def format_question(doc: dict) -> Optional[dict]:
    question = doc.get("question")
    choices = doc.get("choices")
    answer = doc.get("answer")
    if question is None or choices is None or answer is None:
        return None
    letters = [chr(ord("A") + i) for i in range(len(choices))]
    gold_letter = letters[answer] if isinstance(answer, int) and answer < len(letters) else str(answer)
    lines = [f"Question: {question}"]
    for letter, choice in zip(letters, choices):
        lines.append(f"{letter}. {choice}")
    lines.append("Explain your reasoning, then give your final answer as a single letter.")
    lines.append("Answer:")
    return {"prompt": "\n".join(lines), "gold_letter": gold_letter, "question": question, "choices": choices}


def load_model_and_tokenizer(args: argparse.Namespace):
    require_dependencies()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model_name, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    compute_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    load_kwargs = dict(torch_dtype=compute_dtype, device_map="auto", trust_remote_code=args.trust_remote_code)
    if args.load_in_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    base_model = AutoModelForCausalLM.from_pretrained(args.model_name, **load_kwargs)
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    input_device = next(model.parameters()).device
    encoded = tokenizer(prompt, return_tensors="pt").to(input_device)
    input_len = encoded["input_ids"].shape[1]
    with torch.inference_mode():
        output_ids = model.generate(
            **encoded, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(output_ids[0, input_len:], skip_special_tokens=True)


def main() -> None:
    args = parse_args()
    require_dependencies()

    baseline = load_correctness(args.baseline_correctness)
    sft = load_correctness(args.sft_correctness)
    flipped_doc_ids = find_flipped_doc_ids(baseline, sft)
    print(f"{len(flipped_doc_ids)} doc_ids flipped wrong-at-baseline -> right-after-SFT "
          f"(out of {len(set(baseline) & set(sft))} shared docs).")

    rng = random.Random(args.seed)
    n = min(args.num_samples, len(flipped_doc_ids))
    if n < args.num_samples:
        print(f"WARNING: only {n} flipped docs available, fewer than --num-samples={args.num_samples}; "
              "using all of them.")
    sampled_doc_ids = rng.sample(flipped_doc_ids, n)

    docs = load_task_docs(args.task)
    print(f"Loading model + adapter for generation: {args.model_name} + {args.adapter_path}")
    model, tokenizer = load_model_and_tokenizer(args)

    records = []
    for doc_id in sampled_doc_ids:
        idx = int(doc_id)
        if idx >= len(docs):
            print(f"WARNING: doc_id {doc_id} is out of range for {len(docs)} loaded docs -- skipping.")
            continue
        formatted = format_question(docs[idx])
        if formatted is None:
            print(f"WARNING: doc_id {doc_id} doesn't have the expected question/choices/answer fields -- skipping.")
            continue
        completion = generate(model, tokenizer, formatted["prompt"], args.max_new_tokens)
        record = {
            "doc_id": doc_id,
            "question": formatted["question"],
            "choices": formatted["choices"],
            "gold_letter": formatted["gold_letter"],
            "generated_completion": completion,
        }
        records.append(record)
        print("=" * 80)
        print(f"doc_id={doc_id}  gold={formatted['gold_letter']}")
        print(formatted["prompt"])
        print("-" * 40)
        print(completion)

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(records)} generations to {output_path} -- read these by hand to judge whether they're "
          "substantive bio reasoning or fluent-but-empty text that happens to land on the right letter.")


if __name__ == "__main__":
    main()
