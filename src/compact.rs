use anyhow::Result;
use rusqlite::params;
use crate::db::{Db, f32s_to_bytes, now_ms};

/// Compact a namespace: rebuild segments that have >20% tombstones
pub fn compact(db: &Db, agent_id: &str, namespace: &str) -> Result<(usize, usize)> {
    let mut stmt = db.conn.prepare(
        "SELECT id, size, tombstones FROM segments WHERE agent_id=?1 AND namespace=?2 AND tombstones > 0"
    )?;
    let dirty: Vec<(i64, i64, i64)> = stmt.query_map(params![agent_id, namespace], |r| {
        Ok((r.get(0)?, r.get(1)?, r.get(2)?))
    })?.filter_map(|r| r.ok()).collect();

    let mut compacted = 0usize;
    let mut removed = 0usize;

    for (seg_id, size, tombstones) in &dirty {
        // only compact if tombstones > 20%
        if *size > 0 && (*tombstones as f64 / *size as f64) < 0.2 {
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

    Ok((compacted, removed))
}
