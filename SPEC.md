# ClawMem v1 — Local Vector Memory DB

**Portable, SQLite-backed memory store for OpenClaw agents.**

## Philosophy

Ship fast, keep it boring. Brute-force is fine at agent-memory scale (≤100K items). Fancy indexing (SMR/moment sketches) is a v2 concern when real usage proves it's needed.

## Architecture

Single Rust binary. One SQLite file. No sidecar files. No external dependencies.

```
clawmem/
  clawmem.db          # everything lives here
```

## Data Model

### Table: `memories`
| Column | Type | Notes |
|--------|------|-------|
| id | TEXT PK | UUID |
| agent_id | TEXT NOT NULL | |
| namespace | TEXT NOT NULL | "default", "project:x", etc. |
| type | TEXT NOT NULL | episodic / semantic / procedural / tool_log |
| content | TEXT | the actual memory text |
| embedding | BLOB NOT NULL | f32 little-endian |
| dim | INTEGER NOT NULL | embedding dimension |
| created_at | INTEGER NOT NULL | unix ms |
| updated_at | INTEGER NOT NULL | unix ms |
| expires_at | INTEGER | unix ms, nullable (TTL) |
| priority | REAL DEFAULT 0.0 | boost factor |
| tags | TEXT | JSON object, nullable |
| is_deleted | INTEGER DEFAULT 0 | tombstone |
| segment_id | INTEGER NOT NULL | FK to segments |

### Table: `segments`
| Column | Type | Notes |
|--------|------|-------|
| id | INTEGER PK AUTOINCREMENT | |
| agent_id | TEXT NOT NULL | |
| namespace | TEXT NOT NULL | |
| created_at | INTEGER NOT NULL | |
| closed_at | INTEGER | null = still open |
| size | INTEGER NOT NULL | |
| tombstones | INTEGER DEFAULT 0 | |
| centroid | BLOB | f32 little-endian, updated incrementally |

### Table: `kv`
| Column | Type |
|--------|------|
| key | TEXT PK |
| value | TEXT |

Indexes:
- `memories(agent_id, namespace, created_at)`
- `memories(segment_id)`
- `segments(agent_id, namespace)`

SQLite WAL mode for concurrency.

## Segmenting

- Fixed-size segments, default capacity M=512
- New memories append to latest open segment
- When full → close, open new
- Centroid updated incrementally: `c = (c*n + x) / (n+1)`

## Search Pipeline

1. **Segment scoring**: cosine(query, segment_centroid) for all segments matching filters
2. **Expand top B segments** (default B=20): load memory IDs
3. **Exact rerank**: cosine(query, embedding) for each candidate
4. **Apply recency + priority boost**: `final = sim + β*recency + γ*priority`
   - `recency = exp(-(now - created_at) / half_life)`, default half_life = 7 days
5. **Return top K** (default 10)

At 10K memories (~20 segments), step 1 is <1ms. Step 3 is <10ms even on Pi.

## Operations

- **Delete**: tombstone (`is_deleted=1`), increment `segments.tombstones`
- **TTL**: filter out expired at query time; compaction physically removes
- **Compact**: rebuild segment (remove tombstones/expired), recompute centroid
  - Trigger: tombstones > 20% of segment, or manual

## API: MCP Server (stdio)

### `memory.upsert`
```json
{
  "agent_id": "snoopy",
  "namespace": "default",
  "type": "episodic",
  "content": "Hugo prefers direct communication",
  "embedding": [0.1, ...],
  "tags": {"source": "conversation"},
  "ttl_seconds": null,
  "priority": 0.5
}
→ { "id": "uuid", "segment_id": 3 }
```

### `memory.search`
```json
{
  "agent_id": "snoopy",
  "namespace": "default",
  "query_embedding": [0.1, ...],
  "k": 10,
  "filters": {
    "type": ["semantic", "episodic"],
    "tags": {"project": "brain-arch"},
    "after_ms": 1700000000000,
    "before_ms": null
  }
}
→ { "results": [{ "id": "...", "score": 0.83, "content": "...", ... }] }
```

### `memory.delete`
```json
{ "agent_id": "snoopy", "id": "uuid" }
```

### `memory.compact`
```json
{ "agent_id": "snoopy", "namespace": "default" }
```

### `memory.stats`
```json
{ "agent_id": "snoopy" }
→ { "total": 4231, "segments": 9, "namespaces": ["default","project:x"], "disk_bytes": 12400000 }
```

## CLI

```
clawmem init [--path ./clawmem.db] [--dim 384]
clawmem search --agent snoopy --query "..." --k 10
clawmem compact --agent snoopy [--namespace default]
clawmem stats [--agent snoopy]
clawmem doctor                  # integrity check
clawmem serve [--stdio|--http]  # MCP server
```

## Implementation: Rust

### Crates
- `rusqlite` (bundled SQLite, no system dep)
- `serde` + `serde_json`
- `uuid`
- `clap` (CLI)
- `anyhow`

### Modules
```
src/
  main.rs          # CLI entry
  db.rs            # SQLite schema, migrations, CRUD
  search.rs        # segment scoring, rerank pipeline
  segment.rs       # segment management, centroid math
  compact.rs       # compaction logic
  mcp.rs           # MCP stdio server
  types.rs         # shared types
```

### Cross-compile targets
- x86_64-unknown-linux-musl (static, Docker)
- aarch64-unknown-linux-musl (Pi, ARM)
- x86_64-apple-darwin / aarch64-apple-darwin
- x86_64-pc-windows-msvc

### Binary size target: <5MB static

## v2 Roadmap (NOT v1)

- [ ] SMR moment sketches (when >100K memories justify it)
- [ ] SBGP-style overlap segments + compaction consistency checks
- [ ] MAST-inspired SEMANTICS.md (query family declaration)
- [ ] Normalized moment scoring (FMMA-style)
- [ ] Hierarchical segments (segment → block → shard)
- [ ] Optional local embedding via ONNX runtime
- [ ] Per-item int8 quantization (v1 uses f32 for simplicity)

## Non-goals (v1)

- Competing with Qdrant/Weaviate at scale
- ANN indexing (HNSW/IVF/PQ)
- Custom embedding models
- Multi-node distribution
