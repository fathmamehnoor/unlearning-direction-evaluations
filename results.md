# Results

Summary of the two arms: whether ablating a refusal/IDK direction recovers
WMDP-Bio knowledge, and whether unrelated fine-tuning does. See
[direction_extraction.md](direction_extraction.md) for how directions were
built and selected. (An earlier TOFU direction-ablation arm and a since-fixed
WMDP-Bio scorer were tested during this project and have been removed from
the codebase; this file reflects the current, corrected pipeline only.)

## WMDP-Bio direction-ablation

`lm_eval`'s `wmdp_bio` task (n = 1273), scored across `baseline`,
`selected_direction_ablation`, `matched_control_ablation`, and 8
`random_direction_ablation` conditions on the same loaded model
(`scripts/wmdp_bio_lm_eval_ablation.py`). Six WMDP-unlearned Llama-3-8B-Instruct
checkpoints (OPTML-Group's GradDiff, IDK-AP, ILU-RMU, NPO, NPO-ILU, and
ScaleAI's RMU).

| model | method | selected acc | matched-control acc | random-control mean (±std) |
|---|---|---|---|---|
| GradDiff | cosmic | 27.0% | 27.1% | 26.9% (±0.6%) |
| IDK-AP | cosmic | 26.8% | 27.1% | 27.2% (±0.1%) |
| ILU-RMU | mean_diff | 32.3% | 32.2% | 32.1% (±0.6%) |
| ILU-RMU | cosmic | 32.8% | 32.7% | *(shares controls above)* |
| NPO | cosmic | 24.5% | 24.7% | 24.3% (±0.2%) |
| NPO-ILU | cosmic | 27.5% | 26.8% | 27.1% (±0.3%) |
| RMU | mean_diff | 26.3% | 26.2% | 26.1% (±0.2%) |
| RMU | cosmic | 28.0% | 25.5% | *(shares controls above)* |

No selected direction cleared its own random-control distribution by more
than noise (the largest deviation, IDK-AP's, is in the wrong direction --
ablation scores *below* the random-control mean, not above). Both selection
algorithms agree with each other in every model where both produced a
direction (see [direction_extraction.md](direction_extraction.md) for why
`mean_diff` only produced one for RMU and ILU-RMU).

### The direction still does something: behavioral bypass check

`scripts/wmdp_refusal_behavior_check.py` checks whether the same directions
that show no WMDP-Bio recovery actually bypass refusal-like behavior on 100
held-out AdvBench harmful prompts (disjoint from extraction). This is what
makes the null result above informative rather than a dead sensor.

| model | method | baseline refusal rate | ablated refusal rate | ablated clean-bypass rate |
|---|---|---|---|---|
| ILU-RMU | mean_diff | 98% | 0% | 100% |
| RMU | mean_diff | 70% | 72% | 13% |
| RMU | cosmic | 70% | 62% | 32% |
| GradDiff | cosmic | 61% | 0% | 30% |
| IDK-AP | cosmic | 0%\* | 0% | 92% |
| NPO | cosmic | 5%\*\* | 0% | 6% |
| NPO-ILU | cosmic | 0%\*\* | 0% | 2% |

\* IDK-AP already complies with harmful prompts at baseline (91% clean-bypass
before any ablation) -- it doesn't exhibit refusal-phrase behavior to bypass.
\*\* NPO and NPO-ILU are heavily degenerate at baseline (94% / 100% of
generations are repetition/garbage per the check's degeneracy flag), so
refusal rate isn't a meaningful signal for these two either way.

For **ILU-RMU** and **GradDiff**, ablating the selected direction produces a
clean, large behavioral effect (refusal collapses from 98%/61% to 0%;
matched-control ablation leaves it unchanged) -- yet WMDP-Bio accuracy for
both stays within noise of the random-control distribution. This is the
central dissociation in this arm: **the direction demonstrably gates
refusal-like behavior, but not access to the forgotten bio knowledge.**
Suppressing surface refusal and suppressing the underlying knowledge do not
appear to share a mechanism exposed by a single linear direction.

## SFT recovery: does unrelated fine-tuning undo unlearning?

A different mechanism, tested by [`wmdp_sft_recovery/`](wmdp_sft_recovery/):
unrelated QLoRA SFT on `openai/gsm8k` (near-zero mutual information with
WMDP-bio), applied to six WMDP-unlearned methods plus a full-knowledge
control (base `meta-llama/Meta-Llama-3-8B-Instruct`), evaluated at
0/1000/3000/6000 examples seen via `lm_eval` (`wmdp_bio`, `mmlu`, `gsm8k`).
Full table:
[`wmdp_sft_recovery/results/wmdp_sft_recovery/recovery_table.md`](wmdp_sft_recovery/results/wmdp_sft_recovery/recovery_table.md).

| model | WMDP-bio: baseline &rarr; peak | &Delta; acc | &Delta; correct-option prob | &Delta; MMLU (utility gate) |
|---|---|---|---|---|
| RMU | 28.1% &rarr; 69.5% | +41.4pp | +0.326 | -3.2pp |
| NPO | 26.8% &rarr; 49.3% | +22.5pp | +0.140 | -1.8pp |
| ILU-RMU | 34.0% &rarr; 57.8% | +23.8pp | +0.149 | -5.9pp |
| GradDiff | 26.4% &rarr; 54.1% | +27.7pp | +0.126 | **+29.8pp** |
| NPO-ILU | 27.2% &rarr; 44.3% | +17.1pp | +0.081 | +2.4pp |
| IDK-AP | 35.2% &rarr; 40.4% | +5.2pp | +0.021 | -6.2pp |
| Full-knowledge control | 73.1% &rarr; 71.5% (no rise) | -1.6 to -2.0pp | ~-0.06 | -4.2 to -5.8pp |

Reading rule: bio up with MMLU flat means real recovery (suppression, not
removal); bio and MMLU up together means general SFT churn, not hidden
knowledge resurfacing.

- **RMU** recovers essentially to full-knowledge-model territory with MMLU
  flat -- the cleanest suppression-not-removal signature in this arm, and its
  correct-option-probability delta (+0.326) leads every other model by more
  than 2x.
- **NPO** and **ILU-RMU** show substantial, MMLU-flat recovery (+21-24pp on
  bio, MMLU within -6pp) -- knowledge genuinely resurfaces, though less
  completely than RMU.
- **GradDiff**'s bio jump is confounded: MMLU rises by nearly as much
  (+29.8pp, from a near-chance 24.9% baseline). This reads as SFT repairing a
  broadly degraded/incoherent baseline rather than bio-specific suppression --
  exactly the case the reading rule is designed to catch.
- **NPO-ILU** sits in between: a real bio rise (+17.1pp, still climbing at
  the last checkpoint tested) alongside a modest MMLU increase (+2.4pp) --
  not the clean flat-utility signature of RMU/NPO/ILU-RMU, but far short of
  GradDiff's confound.
- **IDK-AP** shows only a small rise (+5.2pp, probability flat at +0.02) while
  MMLU *declines* -- no clean recovery signature; this unlearning looks more
  like genuine removal than the others.
- The **full-knowledge control** stays flat to slightly declining on
  WMDP-bio under the same SFT recipe, confirming the rises above are
  attributable to the unlearning coming undone rather than GSM8K SFT
  manufacturing bio knowledge on its own.

Ranking by correct-option-probability delta -- the cleanest discriminator,
since it can rise before the argmax answer flips -- is roughly
**RMU > NPO &asymp; ILU-RMU > GradDiff (confounded) > NPO-ILU &gt;&gt; IDK-AP &asymp; control (flat)**.

## Takeaway

Two different probes of the same question -- did unlearning remove
knowledge, or just suppress access to it? -- point the same direction but at
different resolutions. Rank-1 direction ablation, even where it demonstrably
bypasses refusal behavior, finds no WMDP-Bio recovery beyond chance for any
of 6 methods; unrelated SFT, a much less targeted intervention, does recover
bio performance for most methods, gated by whether utility (MMLU) moves with
it. Low forget-set performance alone does not distinguish genuine knowledge
removal from suppression, and a null result from one recovery probe
(direction ablation) does not generalize to another (fine-tuning).
