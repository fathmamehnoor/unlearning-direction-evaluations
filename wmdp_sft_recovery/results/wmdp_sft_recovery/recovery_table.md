# WMDP unrelated-SFT recovery

Recovery (WMDP-bio) read next to utility (MMLU) and fine-tune-took (GSM8K).
`d*` columns are deltas vs each model's own step-0 (no-SFT) baseline.

GSM8K confirms *uptake only* -- read the rise from baseline, not the absolute
level (chat template + few-shot makes the level odd). If bio stays flat, a risen GSM8K rules out "nothing happened".

| model (arm) | step | WMDP-bio acc | WMDP-bio prob | MMLU | GSM8K | dWMDP acc | dWMDP prob | dMMLU |
|---|---|---|---|---|---|---|---|---|
| none (full_knowledge_control) | 0 | 73.1% | 0.705 | 67.6% | 74.0% | +0.0pp | +0.000 | +0.0pp |
| none (full_knowledge_control) | 1000 | 71.2% | 0.644 | 63.4% | 71.0% | -1.9pp | -0.061 | -4.2pp |
| none (full_knowledge_control) | 3000 | 71.0% | 0.644 | 62.6% | 62.5% | -2.0pp | -0.061 | -5.0pp |
| none (full_knowledge_control) | 6000 | 71.5% | 0.643 | 61.8% | 68.0% | -1.6pp | -0.063 | -5.8pp |
| GradDiff (unlearned) | 0 | 26.4% | 0.259 | 24.9% | 12.5% | +0.0pp | +0.000 | +0.0pp |
| GradDiff (unlearned) | 1000 | 54.1% | 0.385 | 54.7% | 60.5% | +27.7pp | +0.126 | +29.8pp |
| GradDiff (unlearned) | 3000 | 44.5% | 0.367 | 53.1% | 63.5% | +18.1pp | +0.108 | +28.2pp |
| GradDiff (unlearned) | 6000 | 47.6% | 0.376 | 53.5% | 69.5% | +21.2pp | +0.118 | +28.6pp |
| IDK-AP (unlearned) | 0 | 35.2% | 0.294 | 47.3% | 44.0% | +0.0pp | +0.000 | +0.0pp |
| IDK-AP (unlearned) | 1000 | 37.9% | 0.303 | 42.3% | 58.5% | +2.7pp | +0.009 | -5.0pp |
| IDK-AP (unlearned) | 3000 | 40.4% | 0.314 | 41.1% | 60.5% | +5.2pp | +0.021 | -6.2pp |
| IDK-AP (unlearned) | 6000 | 39.6% | 0.311 | 39.5% | 63.0% | +4.4pp | +0.017 | -7.8pp |
| ILU-RMU (unlearned) | 0 | 34.0% | 0.322 | 65.3% | 68.5% | +0.0pp | +0.000 | +0.0pp |
| ILU-RMU (unlearned) | 1000 | 53.8% | 0.443 | 62.1% | 65.0% | +19.8pp | +0.121 | -3.2pp |
| ILU-RMU (unlearned) | 3000 | 57.8% | 0.471 | 59.4% | 61.0% | +23.8pp | +0.149 | -5.9pp |
| ILU-RMU (unlearned) | 6000 | 55.5% | 0.457 | 59.6% | 68.0% | +21.5pp | +0.135 | -5.6pp |
| NPO (unlearned) | 0 | 26.8% | 0.258 | 54.6% | 4.5% | +0.0pp | +0.000 | +0.0pp |
| NPO (unlearned) | 1000 | 47.7% | 0.369 | 52.8% | 55.5% | +20.9pp | +0.111 | -1.8pp |
| NPO (unlearned) | 3000 | 49.3% | 0.398 | 52.8% | 57.0% | +22.5pp | +0.140 | -1.8pp |
| NPO (unlearned) | 6000 | 47.8% | 0.377 | 52.3% | 62.0% | +21.1pp | +0.119 | -2.4pp |
| NPO-ILU (unlearned) | 0 | 27.2% | 0.262 | 54.7% | 0.0% | +0.0pp | +0.000 | +0.0pp |
| NPO-ILU (unlearned) | 1000 | 37.7% | 0.304 | 58.9% | 57.5% | +10.5pp | +0.042 | +4.1pp |
| NPO-ILU (unlearned) | 3000 | 40.8% | 0.327 | 57.6% | 60.0% | +13.6pp | +0.065 | +2.9pp |
| NPO-ILU (unlearned) | 6000 | 44.3% | 0.342 | 57.1% | 66.5% | +17.1pp | +0.081 | +2.4pp |
| RMU (unlearned) | 0 | 28.1% | 0.273 | 62.5% | 75.5% | +0.0pp | +0.000 | +0.0pp |
| RMU (unlearned) | 1000 | 41.2% | 0.350 | 60.7% | 65.6% | +13.1pp | +0.077 | -1.8pp |
| RMU (unlearned) | 3000 | 69.5% | 0.599 | 59.2% | 64.7% | +41.4pp | +0.326 | -3.2pp |
| RMU (unlearned) | 6000 | 68.7% | 0.593 | 59.5% | 66.5% | +40.6pp | +0.320 | -3.0pp |
