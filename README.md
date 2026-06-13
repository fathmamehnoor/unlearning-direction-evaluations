# Unlearning Direction Evaluation


This repository contains scripts used in an exploratory project investigating whether unlearning removes knowledge from language models or suppresses access to that knowledge through directions in activation space.

Most experiments were run on [TOFU](https://locuslab.github.io/tofu/), with additional experiments on [WMDP](https://www.wmdp.ai/).

---

## Installation

```bash
pip install -r requirements.txt
```

The scripts accept Hugging Face model IDs or local checkpoint paths.

---

## Model Checkpoints

TOFU unlearned models were trained using the [OpenUnlearning](https://github.com/locuslab/open-unlearning) repository and checkpoints from the [OpenUnlearning TOFU model collection](https://huggingface.co/collections/open-unlearning/tofu-unlearned-models). WMDP experiments used unlearned checkpoints from [OPTML](https://huggingface.co/OPTML-Group) where available, plus local exploratory checkpoints.

This repository focuses on direction extraction and evaluation rather than reproducing the full unlearning training pipeline.

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

Used COSMIC-style intervention selection to identify the relevant refusal / IDK direction before evaluating whether directional ablation changes model behavior.

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

* Arditi et al. (2024). [Refusal in Language Models Is Mediated by a Single Direction.](https://arxiv.org/abs/2406.11717)
* Siu et al. (2025). [COSMIC: Generalized Refusal Direction Identification in LLM Activations.](https://arxiv.org/abs/2506.00085)
* TOFU: [A Task of Fictitious Unlearning for LLMs.](https://locuslab.github.io/tofu/)
* WMDP: [The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning.](https://www.wmdp.ai/)
