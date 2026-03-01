# clawmem

A tiny, portable vector memory database for AI agents. One binary. One SQLite file. Zero dependencies.

Built for [OpenClaw](https://github.com/openclaw/openclaw) agents that need to remember things across sessions without spinning up Postgres, Qdrant, or anything else that takes longer to configure than to use.

## Why

Every AI agent framework eventually needs memory. The options today are:

1. **Cloud vector DBs** — Great if you want vendor lock-in, latency, and a monthly bill
2. **Self-hosted vector DBs** — Great if you enjoy operating distributed systems for 10K vectors
3. **Just append to a markdown file** — Honest, but doesn't scale past "what did we talk about yesterday"

ClawMem is option 4: a **1.4MB static binary** that stores embeddings in SQLite, searches them via segment-centroid routing + exact cosine rerank, and exposes an MCP server so any agent can use it as a tool.

It runs on your laptop. It runs on a Raspberry Pi. It runs in Docker. It doesn't phone home.

## Install

```bash
cargo install --path .
# or
cargo build --release
# → target/release/clawmem (1.4MB)
```

## Usage

```bash
# Initialize
clawmem init

# Store a memory
echo '{"type":"episodic","content":"User prefers dark mode","embedding":[0.1,0.2,0.3]}' \
  | clawmem upsert --agent myagent

# Search
echo '{"query_embedding":[0.1,0.2,0.3]}' \
  | clawmem search --agent myagent --k 5

# Delete
clawmem delete --agent myagent --id <uuid>

# Compact (reclaim space from deleted/expired memories)
clawmem compact --agent myagent

# Stats
clawmem stats --agent myagent

# Integrity check
clawmem doctor

# MCP server (stdio, for agent tool use)
clawmem serve
```

## How it works

**Storage.** Everything lives in a single SQLite file with WAL mode. Embeddings are stored as f32 BLOBs directly in the table. No sidecar files.

**Segments.** Memories are grouped into segments of 512. Each segment maintains an incrementally-updated centroid vector.

**Search.** Score all segment centroids by cosine similarity → expand the top segments → exact cosine rerank over candidate vectors → apply recency decay + priority boost → return top-k.

At 10K memories (~20 segments), the centroid scoring pass takes <1ms. The whole pipeline takes <10ms on a laptop. On a Raspberry Pi, still under 50ms.

**Deletes.** Tombstone-based. Compaction physically removes dead records and recomputes centroids.

**TTL.** Set `ttl_seconds` on insert. Expired memories are filtered at query time and cleaned up during compaction.

## MCP Server

ClawMem exposes five tools over JSON-RPC stdio:

| Tool | What it does |
|------|-------------|
| `memory.upsert` | Store a memory with embedding, tags, TTL, priority |
| `memory.search` | Semantic search with type/tag/time filters |
| `memory.delete` | Tombstone a memory by ID |
| `memory.compact` | Rebuild segments, reclaim space |
| `memory.stats` | Counts, segments, namespaces, disk usage |

Start with `clawmem serve` and pipe JSON-RPC over stdin/stdout.

## Design decisions

**No HNSW/IVF/PQ.** Agent memory is small. You'll have thousands of memories, not millions. Brute-force centroid routing is fast enough and dramatically simpler to debug, maintain, and trust.

**No sidecar binary files.** SQLite BLOBs are fine at this scale. One file to back up, one file to move.

**f32 embeddings, not quantized.** At 10K memories × 384 dimensions × 4 bytes = 15MB. Your phone has more RAM than that. Quantization is a v2 concern.

**Segments over flat scan.** Not for speed (flat scan is fast too) but for organization: segments give you natural units for compaction, TTL cleanup, and future hierarchical summarization.

## What it doesn't do

- Compete with Qdrant at 100M vectors
- Generate embeddings (bring your own)
- Replicate across nodes
- Require a PhD to operate

## Roadmap

**v0.2 — Production hardening**
- [ ] Int8 quantized vectors (4× memory reduction)
- [ ] Batch upsert (JSONL ingest)
- [ ] Tag-based filtering in search (currently type/time only)
- [ ] Configurable segment capacity per namespace
- [ ] `clawmem export` / `clawmem import` for backup/migration

**v0.3 — Segment Moment Routing (SMR)**
- [ ] Circular phase projections per segment (R channels × S harmonics)
- [ ] Moment-based segment scoring (replaces centroid-only routing)
- [ ] Matters at 100K+ memories where centroid routing gets noisy

**v0.4 — Theory-backed operations**
- [ ] SBGP-inspired overlap segments + compaction consistency checks
- [ ] MAST-inspired `SEMANTICS.md` declaring supported query families
- [ ] Normalized Moment Score (FMMA-style ratio of Fourier sums)

**v1.0 — Ecosystem**
- [ ] OpenClaw skill package (install via `clawhub install clawmem`)
- [ ] Hierarchical segments (segment → block → shard)
- [ ] Optional local embedding via ONNX runtime
- [ ] Cross-compile CI (Linux x86/ARM, macOS, Windows)
- [ ] `crates.io` publish

## License

MIT

---

*Built by [pigeonflow](https://github.com/pigeonflow). Part of the [OpenClaw](https://github.com/openclaw/openclaw) ecosystem.*
