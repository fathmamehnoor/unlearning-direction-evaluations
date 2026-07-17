# Unlearning Direction Evaluation

This repository contains scripts used in an exploratory project investigating whether unlearning removes knowledge from language models or suppresses access to that knowledge through directions in activation space.

Across six unlearning methods on the [TOFU](https://locuslab.github.io/tofu/) and [WMDP](https://www.wmdp.ai/) benchmarks, ablating a rank-1 IDK/refusal direction did not recover forgotten knowledge beyond random-direction controls, and the positive-looking signals fall within noise. A follow-up experiment found that unrelated GSM8K fine-tuning partially raised forget-set performance in the RMU-unlearned model, but retain performance collapsed alongside it. Low forget-set performance alone does not distinguish genuine knowledge removal from knowledge suppression.

## Results and Writeups

- [results.md](results.md) — full per-method results tables
- [direction_extraction.md](direction_extraction.md) — prompts, layer sweep, and direction-selection procedure
=
---

## Installation

```bash
pip install -r requirements.txt
```

The scripts accept Hugging Face model IDs or local checkpoint paths.

---

## Model Checkpoints

TOFU unlearned models were trained using the [OpenUnlearning](https://github.com/locuslab/open-unlearning) repository and checkpoints from the [OpenUnlearning TOFU model collection](https://huggingface.co/collections/open-unlearning/tofu-unlearned-models). WMDP experiments used unlearned checkpoints from [OPTML](https://huggingface.co/OPTML-Group)[FORK A — if all published WMDP numbers came from OPTML checkpoints, end the sentence here; if not, keep a reworded version of the local-checkpoints clause and reconcile with the post's Methodology].

Fine-tuned and unlearned model checkpoints from these experiments are available on [Hugging Face](HF-LINK).

This repository focuses on direction extraction and evaluation rather than reproducing the full unlearning training pipeline.

**Reproducibility note:** per-question evaluation outputs from the original runs were not retained. Reproducing the tables in [results.md](results.md) requires re-running the pipeline with the commands below. [FORK B — if not pushing the SFT code, append: "The SFT follow-up pipeline is not yet published in this repository."]

---

## TOFU Commands

Collect refusal / IDK prompts:

```bash
python scripts/collect_refusal_prompts.py \
  --model <model-or-local-path> \
  --method <run-label> \
  --forget-config forget10 \
  --output-csv outputs/refusal_prompts.csv \
  --paired-output-csv outputs/paired_idk_plain_prompts.csv \
  --write-paired-output \
  --resume
```

Select a COSMIC-style controller:

```bash
python scripts/cosmic_refusal_controller.py \
  --model <model-or-local-path> \
  --paired-csv outputs/paired_idk_plain_prompts.csv \
  --output-root outputs/cosmic_refusal_controller \
  --prompt-format raw
```

The controller step uses COSMIC-style intervention selection to identify the relevant refusal / IDK direction before evaluating whether directional ablation changes model behavior.

Evaluate directional ablation:

```bash
python scripts/cosmic_refusal_ablation_eval.py \
  --model <model-or-local-path> \
  --controller outputs/cosmic_refusal_controller/cosmic_selected_controller.pt \
  --forced-idk-csv outputs/paired_idk_plain_prompts.csv \
  --output-root outputs/cosmic_refusal_ablation_eval
```

---

## WMDP Commands

Mean-difference direction selection:

```bash
python scripts/wmdp_bio_refusal_direction_eval.py \
  --model <model-or-local-path> \
  --model-label <run-label> \
  --selection-method mean_diff \
  --output-root outputs/wmdp_bio_mean_diff_eval
```

COSMIC-style intervention selection:

```bash
python scripts/wmdp_bio_refusal_direction_eval.py \
  --model <model-or-local-path> \
  --model-label <run-label> \
  --selection-method cosmic \
  --output-root outputs/wmdp_bio_cosmic_eval
```

The COSMIC-style WMDP run uses intervention-based selection to choose the refusal direction before evaluating selected-direction ablation against WMDP-Bio forced-choice behavior.

---

## References

- Arditi et al. (2024). [Refusal in Language Models Is Mediated by a Single Direction.](https://arxiv.org/abs/2406.11717)
- Siu et al. (2025). [COSMIC: Generalized Refusal Direction Identification in LLM Activations.](https://arxiv.org/abs/2506.00085)
- TOFU: [A Task of Fictitious Unlearning for LLMs.](https://locuslab.github.io/tofu/)
- WMDP: [The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning.](https://www.wmdp.ai/)
