use crate::db::Db;
use crate::types::*;
use anyhow::Result;

pub fn upsert(db: &Db, agent_id: &str, namespace: &str, mem_type: &str,
              content: Option<&str>, embedding: &[f32], priority: f64,
              tags: Option<&str>, ttl_seconds: Option<i64>) -> Result<(String, i64)> {
    let id = uuid::Uuid::new_v4().to_string();
    let segment_id = db.get_or_create_open_segment(agent_id, namespace)?;

    let expires_at = ttl_seconds.map(|ttl| crate::db::now_ms() + ttl * 1000);

    db.insert_memory(&id, agent_id, namespace, mem_type, content, embedding, priority, tags, expires_at, segment_id)?;

    // update centroid incrementally
    let size = db.get_segment_size(segment_id)?;
    let old_centroid = db.get_segment_centroid(segment_id)?;
    let new_centroid = match old_centroid {
        Some(c) => {
            let n = (size - 1) as f32; // size already incremented
            c.iter().zip(embedding.iter())
                .map(|(&ci, &xi)| (ci * n + xi) / (n + 1.0))
                .collect::<Vec<f32>>()
        }
        None => embedding.to_vec(),
    };
    db.update_segment_centroid(segment_id, &new_centroid)?;

    Ok((id, segment_id))
}

pub fn search(db: &Db, agent_id: &str, namespace: Option<&str>,
              query_embedding: &[f32], k: usize, top_b: usize,
              filters: Option<&SearchFilters>) -> Result<Vec<SearchResult>> {
    let segments = db.list_segments(agent_id, namespace)?;
    if segments.is_empty() {
        return Ok(vec![]);
    }

    // score segments by centroid cosine
    let mut seg_scores: Vec<(i64, f64)> = segments.iter()
        .filter_map(|seg| {
            seg.centroid.as_ref().map(|c| (seg.id, cosine_sim(query_embedding, c)))
        })
        .collect();
    seg_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    seg_scores.truncate(top_b);

    // expand and rerank
    let now = crate::db::now_ms();
    let mut results: Vec<SearchResult> = Vec::new();

    for (seg_id, _) in &seg_scores {
        let mems = db.get_segment_memories(*seg_id)?;
        for (id, emb, created_at, priority, content, mem_type, tags_str) in mems {
            // apply type filter
            if let Some(f) = filters {
                if let Some(ref types) = f.mem_types {
                    if !types.contains(&mem_type) { continue; }
                }
                if let Some(after) = f.after_ms {
                    if created_at < after { continue; }
                }
                if let Some(before) = f.before_ms {
                    if created_at > before { continue; }
                }
                // Tag filtering: match all specified key-value pairs
                if let Some(ref filter_tags) = f.tags {
                    if let Some(filter_obj) = filter_tags.as_object() {
                        if !filter_obj.is_empty() {
                            let mem_tags: Option<serde_json::Value> = tags_str.as_ref()
                                .and_then(|s| serde_json::from_str(s).ok());
                            match mem_tags.as_ref().and_then(|v| v.as_object()) {
                                Some(mt) => {
                                    let all_match = filter_obj.iter().all(|(k, v)| mt.get(k) == Some(v));
                                    if !all_match { continue; }
                                }
                                None => continue, // no tags on memory, filter requires some
                            }
                        }
                    }
                }
            }

            let sim = cosine_sim(query_embedding, &emb);
            let age_ms = (now - created_at) as f64;
            let recency = (-age_ms / DEFAULT_HALF_LIFE_MS).exp();
            let score = sim + DEFAULT_RECENCY_WEIGHT * recency + DEFAULT_PRIORITY_WEIGHT * priority;

            let tags: Option<serde_json::Value> = tags_str.and_then(|s| serde_json::from_str(&s).ok());

            results.push(SearchResult {
                id,
                score,
                similarity: sim,
                content,
                mem_type,
                created_at,
                tags,
            });
        }
    }

    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    results.truncate(k);
    Ok(results)
}

/// Batch upsert multiple memories in a single transaction.
/// Returns vec of (id, segment_id) pairs.
pub fn batch_upsert(db: &Db, items: &[BatchItem]) -> Result<Vec<(String, i64)>> {
    let tx_result: Result<Vec<(String, i64)>> = (|| {
        // We can't use a real SQLite transaction through Db easily,
        // so we use SAVEPOINT via raw conn
        db.conn.execute_batch("BEGIN IMMEDIATE")?;
        let mut results = Vec::with_capacity(items.len());
        for item in items {
            match upsert(db, &item.agent_id, &item.namespace, &item.mem_type,
                         item.content.as_deref(), &item.embedding, item.priority,
                         item.tags.as_deref(), item.ttl_seconds) {
                Ok(r) => results.push(r),
                Err(e) => {
                    db.conn.execute_batch("ROLLBACK")?;
                    return Err(e);
                }
            }
        }
        db.conn.execute_batch("COMMIT")?;
        Ok(results)
    })();
    tx_result
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct BatchItem {
    pub agent_id: String,
    #[serde(default = "default_namespace")]
    pub namespace: String,
    #[serde(rename = "type", default = "default_type")]
    pub mem_type: String,
    pub content: Option<String>,
    pub embedding: Vec<f32>,
    #[serde(default)]
    pub priority: f64,
    pub tags: Option<String>,
    pub ttl_seconds: Option<i64>,
}

fn default_namespace() -> String { "default".to_string() }
fn default_type() -> String { "episodic".to_string() }

fn cosine_sim(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() { return 0.0; }
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..a.len() {
        let ai = a[i] as f64;
        let bi = b[i] as f64;
        dot += ai * bi;
        na += ai * ai;
        nb += bi * bi;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom < 1e-12 { 0.0 } else { dot / denom }
}
