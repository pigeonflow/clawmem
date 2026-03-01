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
