# surrogate-1-runner

Parallel public-dataset ingest workers for the
[axentx/surrogate-1-training-pairs](https://huggingface.co/datasets/axentx/surrogate-1-training-pairs)
HuggingFace dataset.

## What this does

Every 30 minutes (or on `workflow_dispatch`), GitHub Actions launches **16 parallel runners**.
Each runner takes a deterministic 1/16 slice (`slug-hash bucket = SHARD_ID`)
of the public dataset list defined in `bin/dataset-enrich.sh`, streams,
normalizes per-schema, dedups via the central md5 hash store, and uploads
its output to a unique path on the dataset repo:

```
batches/public-merged/<date>/shard<N>-<HHMMSS>.jsonl
```

Filename includes shard id + iteration timestamp so commits never collide
across shards or across iterations of the same shard.

## Why a separate repo?

The primary surrogate-1 system runs on a HuggingFace Space (`cpu-basic`,
16 GB RAM cap). When 16 shards stream simultaneously the `datasets` library
peaks ~1.5–2 GB per shard during parquet decode and the kernel OOM-kills
the entire container. GitHub Actions on a public repo gives every shard
its own isolated 7 GB runner with unlimited free minutes — total
`16 × 7 GB = 112 GB RAM` of effective parallel headroom.

## Layout

```
.github/workflows/
  ingest.yml             # 16-shard matrix workflow

bin/
  dataset-enrich.sh      # main worker — same script as the HF Space
  lib/dedup.py           # central md5 dedup store

requirements.txt         # datasets, huggingface_hub, pyarrow, numpy
```

## Required secret

- `HF_TOKEN` — HuggingFace write token with permission to push to
  `axentx/surrogate-1-training-pairs`.

## Manual trigger

```bash
gh workflow run ingest.yml -R <owner>/surrogate-1-runner
```

## Trade-offs

- **No state across runs.** Every cron tick starts with an empty dedup
  cache (the central SQLite store on the HF Space remains the source of
  truth for cross-source dedup). Runs may upload pairs that the Space
  later marks as duplicates — wasted bandwidth, not data corruption.
- **No persistent disk.** Per-run output files live for the lifetime
  of the runner; they're uploaded to HF before the runner exits.
- **No agent loop.** This repo only handles ingest. Agentic crawlers,
  scrapers, RAG retrieval, and the LLM-call surface stay on the Space.
