use anyhow::Result;
use rusqlite::params;
use crate::db::{Db, f32s_to_bytes, now_ms};

/// Compact a namespace: rebuild segments that have >threshold tombstones,
/// then merge underfilled segments (size < 50% capacity).
pub fn compact(db: &Db, agent_id: &str, namespace: &str) -> Result<(usize, usize)> {
    let threshold = db.get_compact_threshold();

    let mut stmt = db.conn.prepare(
        "SELECT id, size, tombstones FROM segments WHERE agent_id=?1 AND namespace=?2 AND tombstones > 0"
    )?;
    let dirty: Vec<(i64, i64, i64)> = stmt.query_map(params![agent_id, namespace], |r| {
        Ok((r.get(0)?, r.get(1)?, r.get(2)?))
    })?.filter_map(|r| r.ok()).collect();

    let mut compacted = 0usize;
    let mut removed = 0usize;

    for (seg_id, size, tombstones) in &dirty {
        if *size > 0 && (*tombstones as f64 / *size as f64) < threshold {
            continue;
        }

        // physically delete tombstoned + expired
        let now = now_ms();
        let del_count: usize = db.conn.execute(
            "DELETE FROM memories WHERE segment_id=?1 AND (is_deleted=1 OR (expires_at IS NOT NULL AND expires_at <= ?2))",
            params![seg_id, now],
        )?;
        removed += del_count;

        // recompute size and centroid
        let live: Vec<Vec<u8>> = {
            let mut s = db.conn.prepare("SELECT embedding FROM memories WHERE segment_id=?1")?;
            s.query_map(params![seg_id], |r| r.get(0))?.filter_map(|r| r.ok()).collect()
        };

        let new_size = live.len() as i64;
        if new_size == 0 {
            db.conn.execute("DELETE FROM segments WHERE id=?1", params![seg_id])?;
        } else {
            let dim = live[0].len() / 4;
            let mut centroid = vec![0.0f32; dim];
            for emb_bytes in &live {
                let emb = crate::db::bytes_to_f32s(emb_bytes);
                for (i, v) in emb.iter().enumerate() {
                    centroid[i] += v;
                }
            }
            for c in centroid.iter_mut() { *c /= new_size as f32; }

            let centroid_bytes = f32s_to_bytes(&centroid);
            db.conn.execute(
                "UPDATE segments SET size=?1, tombstones=0, centroid=?2 WHERE id=?3",
                params![new_size, centroid_bytes, seg_id],
            )?;
        }
        compacted += 1;
    }

    // Merge underfilled segments (size < 50% capacity)
    let capacity = db.get_segment_capacity(namespace);
    let merge_threshold = (capacity + 1) / 2; // ceil(capacity/2)
    let merged = merge_underfilled(db, agent_id, namespace, merge_threshold, capacity)?;
    compacted += merged;

    Ok((compacted, removed))
}

/// Merge underfilled closed segments into fewer segments
fn merge_underfilled(db: &Db, agent_id: &str, namespace: &str, threshold: i64, capacity: i64) -> Result<usize> {
    let mut stmt = db.conn.prepare(
        "SELECT id, size FROM segments WHERE agent_id=?1 AND namespace=?2 AND closed_at IS NOT NULL AND size > 0 AND size <= ?3 ORDER BY id"
    )?;
    let underfilled: Vec<(i64, i64)> = stmt.query_map(params![agent_id, namespace, threshold], |r| {
        Ok((r.get(0)?, r.get(1)?))
    })?.filter_map(|r| r.ok()).collect();

    if underfilled.len() < 2 {
        return Ok(0);
    }

    let mut merged = 0usize;
    let mut i = 0;
    while i < underfilled.len() - 1 {
        let (target_id, target_size) = underfilled[i];
        let mut accumulated = target_size;
        let mut j = i + 1;

        while j < underfilled.len() && accumulated + underfilled[j].1 <= capacity {
            let (source_id, source_size) = underfilled[j];
            // Move memories from source to target
            db.conn.execute(
                "UPDATE memories SET segment_id=?1 WHERE segment_id=?2",
                params![target_id, source_id],
            )?;
            // Delete empty source segment
            db.conn.execute("DELETE FROM segments WHERE id=?1", params![source_id])?;
            accumulated += source_size;
            merged += 1;
            j += 1;
        }

        if accumulated != target_size {
            // Recompute centroid and size for merged target
            let live: Vec<Vec<u8>> = {
                let mut s = db.conn.prepare("SELECT embedding FROM memories WHERE segment_id=?1 AND is_deleted=0")?;
                s.query_map(params![target_id], |r| r.get(0))?.filter_map(|r| r.ok()).collect()
            };
            let new_size = live.len() as i64;
            if new_size > 0 {
                let dim = live[0].len() / 4;
                let mut centroid = vec![0.0f32; dim];
                for emb_bytes in &live {
                    let emb = crate::db::bytes_to_f32s(emb_bytes);
                    for (k, v) in emb.iter().enumerate() {
                        centroid[k] += v;
                    }
                }
                for c in centroid.iter_mut() { *c /= new_size as f32; }
                let centroid_bytes = f32s_to_bytes(&centroid);
                db.conn.execute(
                    "UPDATE segments SET size=?1, tombstones=0, centroid=?2 WHERE id=?3",
                    params![new_size, centroid_bytes, target_id],
                )?;
            }
        }

        i = j;
    }

    Ok(merged)
}
