use anyhow::Result;
use rusqlite::{params, Connection};
use crate::types::*;

pub struct Db {
    pub conn: Connection,
}

impl Db {
    pub fn open(path: &str) -> Result<Self> {
        let conn = Connection::open(path)?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON; PRAGMA busy_timeout=5000;")?;
        let db = Db { conn };
        db.migrate()?;
        Ok(db)
    }

    fn migrate(&self) -> Result<()> {
        self.conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS kv (
                key TEXT PRIMARY KEY,
                value TEXT
            );

            CREATE TABLE IF NOT EXISTS segments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL,
                namespace TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                closed_at INTEGER,
                size INTEGER NOT NULL DEFAULT 0,
                tombstones INTEGER NOT NULL DEFAULT 0,
                centroid BLOB
            );
            CREATE INDEX IF NOT EXISTS idx_seg_agent_ns ON segments(agent_id, namespace);

            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                namespace TEXT NOT NULL,
                type TEXT NOT NULL,
                content TEXT,
                embedding BLOB NOT NULL,
                dim INTEGER NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                expires_at INTEGER,
                priority REAL NOT NULL DEFAULT 0.0,
                tags TEXT,
                is_deleted INTEGER NOT NULL DEFAULT 0,
                segment_id INTEGER NOT NULL REFERENCES segments(id)
            );
            CREATE INDEX IF NOT EXISTS idx_mem_agent_ns_ts ON memories(agent_id, namespace, created_at);
            CREATE INDEX IF NOT EXISTS idx_mem_seg ON memories(segment_id);
            "
        )?;
        Ok(())
    }

    pub fn get_or_create_open_segment(&self, agent_id: &str, namespace: &str) -> Result<i64> {
        let capacity = self.get_segment_capacity(namespace);
        let maybe: Option<(i64, i64)> = self.conn.query_row(
            "SELECT id, size FROM segments WHERE agent_id=?1 AND namespace=?2 AND closed_at IS NULL ORDER BY id DESC LIMIT 1",
            params![agent_id, namespace],
            |row| Ok((row.get(0)?, row.get(1)?)),
        ).ok();

        if let Some((id, size)) = maybe {
            if size < capacity {
                return Ok(id);
            }
            // close it
            let now = now_ms();
            self.conn.execute("UPDATE segments SET closed_at=?1 WHERE id=?2", params![now, id])?;
        }

        let now = now_ms();
        self.conn.execute(
            "INSERT INTO segments (agent_id, namespace, created_at, size) VALUES (?1, ?2, ?3, 0)",
            params![agent_id, namespace, now],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    pub fn insert_memory(&self, id: &str, agent_id: &str, namespace: &str, mem_type: &str,
                         content: Option<&str>, embedding: &[f32], priority: f64,
                         tags: Option<&str>, expires_at: Option<i64>, segment_id: i64) -> Result<()> {
        let now = now_ms();
        let dim = embedding.len() as i64;
        let emb_bytes = f32s_to_bytes(embedding);

        self.conn.execute(
            "INSERT INTO memories (id, agent_id, namespace, type, content, embedding, dim, created_at, updated_at, expires_at, priority, tags, segment_id)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13)",
            params![id, agent_id, namespace, mem_type, content, emb_bytes, dim, now, now, expires_at, priority, tags, segment_id],
        )?;

        // update segment size
        self.conn.execute("UPDATE segments SET size = size + 1 WHERE id = ?1", params![segment_id])?;
        Ok(())
    }

    pub fn update_segment_centroid(&self, segment_id: i64, centroid: &[f32]) -> Result<()> {
        let bytes = f32s_to_bytes(centroid);
        self.conn.execute("UPDATE segments SET centroid=?1 WHERE id=?2", params![bytes, segment_id])?;
        Ok(())
    }

    pub fn get_segment_centroid(&self, segment_id: i64) -> Result<Option<Vec<f32>>> {
        let blob: Option<Vec<u8>> = self.conn.query_row(
            "SELECT centroid FROM segments WHERE id=?1", params![segment_id],
            |row| row.get(0),
        )?;
        Ok(blob.map(|b| bytes_to_f32s(&b)))
    }

    pub fn get_segment_size(&self, segment_id: i64) -> Result<i64> {
        Ok(self.conn.query_row("SELECT size FROM segments WHERE id=?1", params![segment_id], |r| r.get(0))?)
    }

    pub fn list_segments(&self, agent_id: &str, namespace: Option<&str>) -> Result<Vec<Segment>> {
        let mut segs = Vec::new();
        if let Some(ns) = namespace {
            let mut stmt = self.conn.prepare(
                "SELECT id, agent_id, namespace, created_at, closed_at, size, tombstones, centroid FROM segments WHERE agent_id=?1 AND namespace=?2"
            )?;
            let rows = stmt.query_map(params![agent_id, ns], row_to_segment)?;
            for r in rows { segs.push(r?); }
        } else {
            let mut stmt = self.conn.prepare(
                "SELECT id, agent_id, namespace, created_at, closed_at, size, tombstones, centroid FROM segments WHERE agent_id=?1"
            )?;
            let rows = stmt.query_map(params![agent_id], row_to_segment)?;
            for r in rows { segs.push(r?); }
        }
        Ok(segs)
    }

    pub fn get_segment_memories(&self, segment_id: i64) -> Result<Vec<(String, Vec<f32>, i64, f64, Option<String>, String, Option<String>)>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, embedding, created_at, priority, content, type, tags FROM memories WHERE segment_id=?1 AND is_deleted=0 AND (expires_at IS NULL OR expires_at > ?2)"
        )?;
        let now = now_ms();
        let rows = stmt.query_map(params![segment_id, now], |row| {
            let emb_bytes: Vec<u8> = row.get(1)?;
            Ok((
                row.get::<_, String>(0)?,
                bytes_to_f32s(&emb_bytes),
                row.get::<_, i64>(2)?,
                row.get::<_, f64>(3)?,
                row.get::<_, Option<String>>(4)?,
                row.get::<_, String>(5)?,
                row.get::<_, Option<String>>(6)?,
            ))
        })?;
        let mut results = Vec::new();
        for r in rows { results.push(r?); }
        Ok(results)
    }

    pub fn tombstone_delete(&self, agent_id: &str, id: &str) -> Result<bool> {
        let seg_info: Option<(i64, String)> = self.conn.query_row(
            "SELECT m.segment_id, s.namespace FROM memories m JOIN segments s ON m.segment_id = s.id WHERE m.id=?1 AND m.agent_id=?2 AND m.is_deleted=0",
            params![id, agent_id], |r| Ok((r.get(0)?, r.get(1)?)),
        ).ok();
        if let Some((seg_id, namespace)) = seg_info {
            self.conn.execute("UPDATE memories SET is_deleted=1 WHERE id=?1", params![id])?;
            self.conn.execute("UPDATE segments SET tombstones = tombstones + 1 WHERE id=?1", params![seg_id])?;

            // Auto-compact: check if tombstone ratio exceeds threshold
            let (size, tombstones): (i64, i64) = self.conn.query_row(
                "SELECT size, tombstones FROM segments WHERE id=?1", params![seg_id], |r| Ok((r.get(0)?, r.get(1)?)))?;
            let threshold = self.get_compact_threshold();
            if size > 0 && (tombstones as f64 / size as f64) >= threshold {
                // Trigger compact for this namespace (ignore errors — best effort)
                let _ = crate::compact::compact(self, agent_id, &namespace);
            }

            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Get compact threshold from kv table, default 0.2 (20%)
    pub fn get_compact_threshold(&self) -> f64 {
        self.conn.query_row(
            "SELECT value FROM kv WHERE key='segment_compact_threshold'", [],
            |r| r.get::<_, String>(0),
        ).ok().and_then(|v| v.parse().ok()).unwrap_or(0.2)
    }

    /// Get segment capacity for a namespace from kv, default DEFAULT_SEGMENT_CAPACITY
    pub fn get_segment_capacity(&self, namespace: &str) -> i64 {
        let key = format!("segment_capacity:{}", namespace);
        self.conn.query_row(
            "SELECT value FROM kv WHERE key=?1", params![key],
            |r| r.get::<_, String>(0),
        ).ok().and_then(|v| v.parse().ok()).unwrap_or(DEFAULT_SEGMENT_CAPACITY)
    }

    /// Set a kv value
    pub fn set_kv(&self, key: &str, value: &str) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO kv (key, value) VALUES (?1, ?2)",
            params![key, value],
        )?;
        Ok(())
    }

    /// Export all non-deleted memories for an agent (optionally filtered by namespace) as JSON values
    pub fn export_memories(&self, agent_id: &str, namespace: Option<&str>) -> Result<Vec<serde_json::Value>> {
        let (sql, params_vec): (&str, Vec<Box<dyn rusqlite::types::ToSql>>) = if let Some(ns) = namespace {
            ("SELECT id, agent_id, namespace, type, content, embedding, created_at, expires_at, priority, tags FROM memories WHERE agent_id=?1 AND namespace=?2 AND is_deleted=0",
             vec![Box::new(agent_id.to_string()), Box::new(ns.to_string())])
        } else {
            ("SELECT id, agent_id, namespace, type, content, embedding, created_at, expires_at, priority, tags FROM memories WHERE agent_id=?1 AND is_deleted=0",
             vec![Box::new(agent_id.to_string())])
        };
        let mut stmt = self.conn.prepare(sql)?;
        let params_refs: Vec<&dyn rusqlite::types::ToSql> = params_vec.iter().map(|p| p.as_ref()).collect();
        let rows = stmt.query_map(params_refs.as_slice(), |row| {
            let emb_bytes: Vec<u8> = row.get(5)?;
            let emb = bytes_to_f32s(&emb_bytes);
            let tags_str: Option<String> = row.get(9)?;
            let tags: Option<serde_json::Value> = tags_str.and_then(|s| serde_json::from_str(&s).ok());
            Ok(serde_json::json!({
                "id": row.get::<_, String>(0)?,
                "agent_id": row.get::<_, String>(1)?,
                "namespace": row.get::<_, String>(2)?,
                "type": row.get::<_, String>(3)?,
                "content": row.get::<_, Option<String>>(4)?,
                "embedding": emb,
                "created_at": row.get::<_, i64>(6)?,
                "expires_at": row.get::<_, Option<i64>>(7)?,
                "priority": row.get::<_, f64>(8)?,
                "tags": tags,
            }))
        })?;
        let mut results = Vec::new();
        for r in rows { results.push(r?); }
        Ok(results)
    }

    pub fn stats(&self, agent_id: &str) -> Result<serde_json::Value> {
        let total: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM memories WHERE agent_id=?1 AND is_deleted=0", params![agent_id], |r| r.get(0))?;
        let segments: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM segments WHERE agent_id=?1", params![agent_id], |r| r.get(0))?;
        let mut stmt = self.conn.prepare("SELECT DISTINCT namespace FROM segments WHERE agent_id=?1")?;
        let ns: Vec<String> = stmt.query_map(params![agent_id], |r| r.get(0))?.filter_map(|r| r.ok()).collect();
        let page_count: i64 = self.conn.query_row("PRAGMA page_count", [], |r| r.get(0))?;
        let page_size: i64 = self.conn.query_row("PRAGMA page_size", [], |r| r.get(0))?;
        Ok(serde_json::json!({
            "total": total,
            "segments": segments,
            "namespaces": ns,
            "disk_bytes": page_count * page_size,
        }))
    }
}

fn row_to_segment(row: &rusqlite::Row) -> rusqlite::Result<Segment> {
    let centroid_blob: Option<Vec<u8>> = row.get(7)?;
    Ok(Segment {
        id: row.get(0)?,
        agent_id: row.get(1)?,
        namespace: row.get(2)?,
        created_at: row.get(3)?,
        closed_at: row.get(4)?,
        size: row.get(5)?,
        tombstones: row.get(6)?,
        centroid: centroid_blob.map(|b| bytes_to_f32s(&b)),
    })
}

pub fn f32s_to_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

pub fn bytes_to_f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

pub fn now_ms() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as i64
}
