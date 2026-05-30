# Embedding model-string confirmation — `voyage-4-large` is the literal production model

**Date**: 2026-05-30
**Trigger**: External competitive review flagged that "voyage-4-large" is Voyage's
*general* MoE family (Jan 2026) with no code-specific variant, and that the code
specialist is `voyage-code-3` (Dec 2024) — raising the question of whether the
tool is (a) actually sending `voyage-4-large`, and (b) leaving code-retrieval
accuracy on the table by using a general model.
**Scope**: investigation + measurement framing. No code change.

## Question

Does code-search's default `voyage` provider literally send `voyage-4-large` to
the Voyage API, or is the spec mislabeled? And is the choice of a general model
over the `voyage-code-3` specialist deliberate?

## Answer — DONE (confirmed by code reading)

**The production default sends the literal string `voyage-4-large` to Voyage's
`/v1/embeddings` endpoint.** Trace:

1. `embeddings/embedder.py:76-90` — `@register_provider("voyage")` factory
   `_factory_voyage` reads `EMBEDDING_MODEL` with default **`"voyage-4-large"`**
   (line 85) and constructs `OpenAIEmbeddingModel(model_name=model_name)`.
2. `embeddings/openai_embedder.py:117` — `encode()` builds
   `payload = {"input": sub_batch, "model": self._model_name}` and POSTs it to
   `{base_url}/embeddings`. So the wire request carries `"model": "voyage-4-large"`.
3. `embeddings/openai_embedder.py:21` — `voyage-4-large` is a registered model
   with dimension 1024 (same as `voyage-code-3`).

This is **not a mislabeling**. `voyage-4-large` is a real Voyage model and it is
what we send. The repo is also fully aware of the code specialist: there is a
separate `@register_provider("voyage-code-3")` factory
(`embeddings/embedder.py:95-116`, default model `"voyage-code-3"`), selectable
via `EMBEDDING_PROVIDER=voyage-code-3`. The default is `voyage-4-large` by
deliberate A/B choice (CLAUDE.md "Voyage AI Integration": voyage-4-large wins 3
of 4 language sub-projects, +0.053 weighted-avg MRR over voyage-context-3;
voyage-code-3 wins TypeScript but regresses on Nix per
`docs/findings/2026-05-15-voyage-code-3-ab-finding.md`).

The external review's underlying *fact* is correct — `voyage-4-large` is Voyage's
general MoE line, and there is no `voyage-code-4`. The non-obvious result is that,
**on our corpus, the general model out-scored the code specialist.**

## Why that result is not yet settled — BLOCKED ON MEASUREMENT

"General model beats the code specialist on code retrieval" is a surprising claim
and rests on two foundations that are both internal:

1. **The win is measured only on the internal golden set (n=102).** It is not
   comparable to any public code-retrieval benchmark, and we have never run
   `voyage-4-large` vs `voyage-code-3` on an external corpus (CoIR, CodeRAG-Bench).

2. **The golden labels may be engine-biased.** The 2026-05-24 R9-extension
   synthesis (`docs/findings/2026-05-24-r9-extension-session-synthesis.md`)
   explicitly flagged that the golden set "may have engine-biased labels … if the
   golden labels were generated from rank-shaped pre-filters, the gate is
   structurally biased toward 'don't change the rank.'" If the labels were
   produced with `voyage-4-large` in the loop, an A/B that asks "does
   voyage-code-3 change the ranking?" is biased against any challenger — including
   voyage-code-3.

Per ship-discipline **rule 10**, the model-choice question must therefore close as
**BLOCKED ON MEASUREMENT**, not DONE. The *string* is confirmed (DONE); *which
model is actually better for our use* is unmeasured under unbiased conditions.

## Recommended next steps (the P1 "validate the metrics" work)

1. **De-bias the labels** — run the out-of-engine labeling harness
   (`bench/research/out_of_engine_sample.py`, this PR) on a 50-query random
   sample to estimate the golden-label bias floor before trusting any
   sub-0.06-MRR provider delta.
2. **Get external numbers** — run the CoIR / CodeRAG-Bench runners
   (`bench/research/coir_runner.py`, `bench/research/coderag_runner.py`, this PR)
   for both `voyage-4-large` and `voyage-code-3`, so the provider choice has a
   defensible *external* comparison, not just an internal one.
3. Only after (1) and (2) should the "voyage-4-large is our default because it
   beats voyage-code-3" claim be re-affirmed (DONE) or revised (DECIDE).

## One-line summary

We send `voyage-4-large` (confirmed); choosing it over `voyage-code-3` is a real
measured decision, but the measurement is internal and possibly label-biased, so
the *superiority* claim is BLOCKED ON MEASUREMENT pending external + de-biased
evaluation.
