# Quantization Drift Monitoring (Plan-2 B2)

**Date**: 2026-05-05
**Source**: `~/Documents/knowledge-base/plans/2026-05-05-codesearch-recommendations.md` Phase B2
**Tool**: `scripts/monitor_quantization_drift.py`

## What is quantization drift?

When a project is first indexed with the default `QUANTIZATION=int8`, FAISS's `ScalarQuantizer` trains its codebook on the **first batch of embeddings** and freezes it (see `search/indexer.py:347-350`). The codebook learns the value-range of those initial vectors and linearly maps each future vector into 8 bits using that range.

If the project's character SHIFTS over time — language mix changes (e.g., a Rust-heavy repo gradually adopts Python services), embedding model drift (per Plan-2 B3, mostly handled by the pipeline fingerprint), or a documentation fork that drifts from a code repo — new embeddings may sit outside the codebook's learned range. Quantization clamps them, producing degraded similarity scores. The failure is **silent**: searches still return results, but rankings slowly degrade.

This is the silent-degradation flagged in roundtable Disagreement that B2 closes.

## How the monitor works

For each indexed project under `~/.claude_code_search/projects/<hash>/index/`:

1. Open `code.index` (FAISS index file).
2. Sample up to N random vectors from the index (default N=100).
3. For each sampled vector, query the index for top-K (default K=50). Record the cosine similarities.
4. Aggregate:
   - `avg_top_1_cosine` — primary signal. Healthy: ~1.00 (each sampled vector matches itself near-perfectly under round-trip quantization). Drift: <0.97 indicates round-trip loss.
   - `avg_top_k_cosine` — secondary signal showing the falloff curve.
5. Save to a baseline file or compare against a prior baseline.

Why self-search? It's the cleanest test of round-trip quantization quality. We're not asking "are the rankings correct" (that would need a labeled dataset); we're asking "is the codebook still representative of the current vector distribution." A vector that round-trips cleanly through quantization should match itself with cosine ~1.0; if it doesn't, the codebook is no longer well-fit to current data.

## Usage

### Baseline (run once, then quarterly or after major reindexes)

```bash
python scripts/monitor_quantization_drift.py --baseline
```

Saves `~/.claude_code_search/quantization_drift_baseline.json` with per-project measurements + timestamp.

### Drift check (run after large incremental indexes, or in CI)

```bash
python scripts/monitor_quantization_drift.py --check
```

Compares current readings to the saved baseline. Exits non-zero if any project's `avg_top_1_cosine` has shifted by more than `--threshold` (default 0.05).

### JSON output (for programmatic ingest)

```bash
python scripts/monitor_quantization_drift.py --check --json | jq '.drift_count'
```

## Interpreting drift signals

| `avg_top_1_cosine` | Interpretation |
|--------------------|----------------|
| ≥ 0.99 | Quantization is healthy — vectors round-trip accurately |
| 0.95 – 0.99 | Mild drift — likely benign for code search; monitor |
| 0.90 – 0.95 | Real drift — codebook is no longer well-fit; consider full reindex |
| < 0.90 | Severe drift — quantizer is clamping a significant fraction of vectors; full reindex required |

Drift between two baselines (Δ on `avg_top_1_cosine`):

| Δ | Interpretation |
|----|----------------|
| < 0.01 | Noise floor; ignore |
| 0.01 – 0.05 | Slow drift; document trend |
| > 0.05 | Flagged as drift; investigate (full reindex or `QUANTIZATION=float32` if drift recurs) |

## Decision tree

When drift is flagged:

1. **Check pipeline fingerprint** — did `EMBEDDING_PROVIDER` / `EMBEDDING_MODEL` change? If yes, the existing `pipeline_version` mechanism (Plan-2 B3) should have already forced a full reindex; if not, file a bug.
2. **Check ntotal trajectory** — did the project grow significantly? `ntotal_baseline` vs `ntotal_current` in the report. Large growth + small drift → expected; small growth + large drift → codebook no longer fits.
3. **If drift persists after full reindex** — the project's vector distribution may be inherently bimodal (e.g., a monorepo with very different sub-corpora). Set `QUANTIZATION=float32` for that project to bypass quantization entirely. Cost: 4x larger index file. Acceptable for projects under ~10K chunks.

## Future work (not in this PR)

- **Continuous tracking**: integrate the drift check into a scheduled task (cron, GitHub Actions, or a periodic MCP tool call) so drift accumulates a time-series instead of pairwise comparisons.
- **Per-language drift breakdown**: when the index grows across many languages, drift may concentrate in one language sub-corpus. Stratifying by language would localize the cause faster.
- **Auto-retrain when drift exceeds a threshold**: today the operator runs `index_directory --force-full` manually after seeing drift. A future enhancement could wire `verify_index_integrity` (Plan-2 A3) to surface drift status, and have `index_directory` opt into a quantizer retrain on drift.

## Out of scope

- Replacing ScalarQuantizer with a learned-codebook variant that retrains continuously (e.g., `IndexIVFPQ`). Would require larger architectural changes; revisit if drift turns out to be a systemic issue rather than per-project.
- Implementing the monitor as an MCP tool. Today it's a script the operator runs. Plan-2 A3 already exposes `verify_index_integrity` for in-LLM consumption; drift could be added there as a separate check if useful.
