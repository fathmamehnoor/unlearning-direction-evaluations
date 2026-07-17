# Full Results

Complete per-method results for the direction-ablation experiments and the SFT follow-up.

Standard errors are binomial, per condition. TOFU experiments compare the selected IDK/refusal direction against a random-direction ablation control. WMDP experiments compare a bio-specific refusal direction against a generic refusal direction.

## TOFU direction-ablation experiments

TOFU forget10, n = 400. Model: tofu_Llama-3.1-8B-Instruct_full, unlearned with each method via OpenUnlearning.

### GradDiff

| Condition | Forget10 Correct | Forget10 Answer Prob | Retain Correct |
|---|---|---|---|
| Baseline | 10.8% | 0.0685 | 32.0% |
| IDK/refusal ablation | 9.0% | 0.0685 | 28.0% |
| Random ablation | 11.3% | 0.0685 | 34.0% |

No evidence of forgotten-knowledge recovery. IDK/refusal ablation slightly reduced forget-set correctness, increased degeneracy, and harmed retain performance. Random ablation performed at least as well as the selected direction.

### RMU

| Condition | Forget10 Correct | Forget10 Answer Prob | Retain Correct |
|---|---|---|---|
| Baseline | 2.5% | 0.196 | 77.5% |
| IDK/refusal ablation | 0.0% | 0.227 | 73.8% |
| Random ablation | 2.5% | 0.191 | 77.5% |

The selected ablation increased forget-set answer probability but did not improve behavioural recovery. Forget10 correctness dropped from 2.5% to 0%, and retain performance declined. Wrong-answer rates increased rather than refusal rates decreasing, suggesting degradation rather than hidden-knowledge recovery.

### SimNPO

| Condition | Forget10 Correct | Forget10 Answer Prob | Retain Correct |
|---|---|---|---|
| Baseline | 57.5% | 0.826 | 87.5% |
| IDK/refusal ablation | 61.3% | 0.833 | 83.8% |
| Random ablation | 57.5% | 0.826 | 86.3% |

Forget10-Correct standard error ≈ 2.5% per condition, so the 57.5% → 61.3% change falls within the noise. The baseline already answered many forget-set questions correctly and showed little refusal behaviour, making this difficult to read as hidden-knowledge recovery.

### UNDIAL

| Condition | Forget10 Correct | Forget10 Answer Prob | Retain Correct |
|---|---|---|---|
| Baseline | 43.8% | 0.742 | 67.5% |
| IDK/refusal ablation | 41.3% | 0.745 | 67.5% |
| Random ablation | 40.0% | 0.741 | 67.5% |

The selected direction slightly increased answer probability but did not improve behavioural recovery. Forget10 correctness declined slightly and retain performance was unchanged.

## WMDP-Bio direction-ablation experiments

WMDP-Bio, n = 1273. Unlearned checkpoints from the OPTML-Group collection. The base Llama-3-8B-Instruct model scores 39.3% on WMDP-Bio.

### IDK-AP

| Condition | WMDP-Bio Accuracy | Correct Prob |
|---|---|---|
| No ablation | 26.9% | 0.271 |
| Bio-specific refusal ablation | 27.3% | 0.273 |
| Generic refusal ablation | 27.4% | 0.273 |

Neither bio-specific nor generic refusal ablation produced meaningful recovery. The generic ablation was marginally higher than the bio-specific one, which does not support a task-specific hidden-knowledge effect.

### ILU-RMU

| Condition | WMDP-Bio Accuracy | Correct Prob |
|---|---|---|
| No ablation | 29.8% | 0.302 |
| Bio-specific refusal ablation | 31.5% | 0.311 |
| Generic ablation | 30.2% | 0.297 |

Accuracy standard error ≈ 1.3% per condition, so the 29.8% → 31.5% change falls within the noise, and recovered performance remains far below the 39.3% base-model accuracy.

## SFT follow-up

Unrelated GSM8K SFT (QLoRA) applied to the RMU-unlearned TOFU model, evaluated after 1000, 3000, and 6000 examples, with the full TOFU model as a control evaluated after 1000 examples. TOFU forget10, n = 400.

### RMU-unlearned model + GSM8K SFT

| Condition | Forget10 Correct | Forget10 Ans Prob | Retain Correct | Forget Degen. | Retain Degen. | GSM8K Score |
|---|---|---|---|---|---|---|
| RMU baseline | 2.5% | 0.191 | 74.1% | 20.5% | 0.8% | 0.328 |
| RMU + GSM8K SFT, 1000 | 12.5% | 0.659 | 42.5% | 62.3% | 34.4% | 0.623 |
| RMU + GSM8K SFT, 3000 | 13.0% | 0.657 | 43.6% | 70.0% | 24.3% | 0.633 |
| RMU + GSM8K SFT, 6000 | 13.5% | 0.663 | 41.0% | 58.3% | 18.4% | 0.642 |

### Full TOFU model + GSM8K SFT (control)

| Condition | Forget10 Correct | Forget10 Ans Prob | Retain Correct | Retain Ans Prob | Forget Degen. | Retain Degen. | GSM8K Score |
|---|---|---|---|---|---|---|---|
| Full model baseline | 77.5% | 0.962 | 79.5% | 0.962 | 0.8% | 1.2% | 0.331 |
| Full model + SFT, 1000 | 57.5% | 0.837 | 56.3% | 0.829 | 14.3% | 12.0% | 0.622 |

Note: the RMU baseline differs slightly between the ablation experiments above (Retain Correct 77.5%, Forget10 Ans Prob 0.196) and the SFT experiment (74.1%, 0.191). The two experiments used separate evaluation runs with independently implemented scoring; each table's conditions were evaluated identically within that table.
