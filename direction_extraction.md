# Direction Extraction

How the IDK/refusal directions were extracted and selected. Implementation: `scripts/collect_refusal_prompts.py` and `scripts/cosmic_refusal_controller.py` (TOFU), `scripts/wmdp_bio_refusal_direction_eval.py` (WMDP).

## TOFU prompts

Directions were extracted from paired prompts over the same questions: an IDK-eliciting template versus a plain-QA template. Five IDK-eliciting template variants were used:

```
Question: {question}
If you do not know the answer, say "I don't know."
Answer:
```

```
Question: {question}
If you are not certain, answer exactly: "I don't know."
Answer:
```

```
Question: {question}
Do not guess. If you cannot answer confidently, say "I don't know."
Answer:
```

The remaining two variants (`unknown_allowed`, `answer_or_idk`) follow the same pattern: "If the answer is unknown or unavailable..." and "Give the answer if you know it. Otherwise, say...". See `collect_refusal_prompts.py` for the exact strings.

The paired negative is the same question with no IDK instruction:

```
Question: {question}
Answer:
```

Questions come from TOFU forget10, for example: *"Can you provide some more details about Chukwu Akabueze's 'Sculptor of Vision'?"*

## Candidate extraction

Candidate directions were computed as mean-difference vectors between IDK-prompt and plain-QA activations at the residual-stream input of every decoder layer (`resid_pre`, captured via forward pre-hooks), at each of the final five token positions of the prompt (positions -5 through -1). This gives one candidate per layer × position pair.

## Selection

Each candidate was scored against matched random-direction controls by ablating it from IDK-prompt activations and adding it to plain-QA activations, measuring the shift in cosine similarity toward the opposite class's mean activation. Top candidates were then re-ranked by held-out generation behavior — whether ablation suppresses "I don't know" responses and addition induces them, without degrading plain-prompt generations — to select the final direction. All selected and ablated directions were single (rank-1) mean-difference vectors.

## WMDP

For WMDP, directions were computed as mean(harmful-prompt activations) − mean(harmless-prompt activations), with both mean-difference and COSMIC-style intervention-based selection supported (`--selection-method mean_diff` / `cosmic`). The bio-specific direction was compared against a generic refusal direction to test task-specificity.
