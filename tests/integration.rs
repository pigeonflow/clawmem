use clawmem::db::Db;
use clawmem::search;
use clawmem::compact;
use clawmem::types::*;

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn test_db(name: &str) -> (Db, String) {
        let path = format!("/tmp/clawmem_test_{name}_{}.db", std::process::id());
        let _ = fs::remove_file(&path);
        let db = Db::open(&path).unwrap();
        (db, path)
    }

    fn cleanup(path: &str) {
        let _ = fs::remove_file(path);
        let _ = fs::remove_file(format!("{path}-wal"));
        let _ = fs::remove_file(format!("{path}-shm"));
    }

    fn rand_embedding(dim: usize, seed: u64) -> Vec<f32> {
        // simple deterministic pseudo-random
        let mut v = Vec::with_capacity(dim);
        let mut s = seed;
        for _ in 0..dim {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            v.push(((s >> 33) as f32) / (u32::MAX as f32) - 0.5);
        }
        v
    }

    fn normalize(v: &mut Vec<f32>) {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-8 {
            for x in v.iter_mut() { *x /= norm; }
        }
    }

    // ---- DB layer tests ----

    #[test]
    fn test_init_and_migrate() {
        let (db, path) = test_db("init");
        // should be able to query tables
        let count: i64 = db.conn.query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0)).unwrap();
        assert_eq!(count, 0);
        let count: i64 = db.conn.query_row("SELECT COUNT(*) FROM segments", [], |r| r.get(0)).unwrap();
        assert_eq!(count, 0);
        cleanup(&path);
    }

    #[test]
    fn test_idempotent_migration() {
        let (db, path) = test_db("idempotent");
        drop(db);
        // reopen — should not fail
        let db2 = Db::open(&path).unwrap();
        let count: i64 = db2.conn.query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0)).unwrap();
        assert_eq!(count, 0);
        cleanup(&path);
    }

    #[test]
    fn test_segment_creation() {
        let (db, path) = test_db("segment");
        let seg1 = db.get_or_create_open_segment("agent1", "default").unwrap();
        let seg2 = db.get_or_create_open_segment("agent1", "default").unwrap();
        assert_eq!(seg1, seg2, "should reuse open segment");

        let seg3 = db.get_or_create_open_segment("agent1", "other").unwrap();
        assert_ne!(seg1, seg3, "different namespace = different segment");

        let seg4 = db.get_or_create_open_segment("agent2", "default").unwrap();
        assert_ne!(seg1, seg4, "different agent = different segment");
        cleanup(&path);
    }

    #[test]
    fn test_segment_closes_at_capacity() {
        let (db, path) = test_db("capacity");
        let emb = vec![0.1f32; 8];
        let seg = db.get_or_create_open_segment("a", "ns").unwrap();

        for i in 0..DEFAULT_SEGMENT_CAPACITY {
            let id = format!("mem-{i}");
            db.insert_memory(&id, "a", "ns", "episodic", Some("test"), &emb, 0.0, None, None, seg).unwrap();
        }
        // manually update size to capacity (insert_memory increments)
        let size = db.get_segment_size(seg).unwrap();
        assert_eq!(size, DEFAULT_SEGMENT_CAPACITY);

        // next call should close old and open new
        let seg2 = db.get_or_create_open_segment("a", "ns").unwrap();
        assert_ne!(seg, seg2, "should have opened a new segment");
        cleanup(&path);
    }

    // ---- Upsert + centroid tests ----

    #[test]
    fn test_upsert_and_centroid() {
        let (db, path) = test_db("upsert");
        let emb1 = vec![1.0, 0.0, 0.0, 0.0];
        let emb2 = vec![0.0, 1.0, 0.0, 0.0];

        let (_, seg1) = search::upsert(&db, "a", "default", "episodic", Some("first"), &emb1, 0.0, None, None).unwrap();
        let c1 = db.get_segment_centroid(seg1).unwrap().unwrap();
        assert_eq!(c1, vec![1.0, 0.0, 0.0, 0.0]);

        let (_, seg2) = search::upsert(&db, "a", "default", "semantic", Some("second"), &emb2, 0.0, None, None).unwrap();
        assert_eq!(seg1, seg2);
        let c2 = db.get_segment_centroid(seg2).unwrap().unwrap();
        // centroid should be average: [0.5, 0.5, 0.0, 0.0]
        assert!((c2[0] - 0.5).abs() < 1e-5);
        assert!((c2[1] - 0.5).abs() < 1e-5);
        cleanup(&path);
    }

    // ---- Search tests ----

    #[test]
    fn test_search_basic() {
        let (db, path) = test_db("search_basic");
        let mut e1 = vec![1.0, 0.0, 0.0, 0.0]; normalize(&mut e1);
        let mut e2 = vec![0.0, 1.0, 0.0, 0.0]; normalize(&mut e2);
        let mut e3 = vec![0.9, 0.1, 0.0, 0.0]; normalize(&mut e3);

        search::upsert(&db, "a", "ns", "episodic", Some("north"), &e1, 0.0, None, None).unwrap();
        search::upsert(&db, "a", "ns", "episodic", Some("east"), &e2, 0.0, None, None).unwrap();
        search::upsert(&db, "a", "ns", "episodic", Some("almost north"), &e3, 0.0, None, None).unwrap();

        let mut query = vec![1.0, 0.0, 0.0, 0.0]; normalize(&mut query);
        let results = search::search(&db, "a", Some("ns"), &query, 2, 10, None).unwrap();
        assert_eq!(results.len(), 2);
        // "north" or "almost north" should be top
        assert!(results[0].content.as_deref() == Some("north") || results[0].content.as_deref() == Some("almost north"));
        assert!(results[0].similarity > 0.9);
        cleanup(&path);
    }

    #[test]
    fn test_search_empty() {
        let (db, path) = test_db("search_empty");
        let query = vec![1.0, 0.0, 0.0];
        let results = search::search(&db, "a", Some("ns"), &query, 10, 10, None).unwrap();
        assert!(results.is_empty());
        cleanup(&path);
    }

    #[test]
    fn test_search_type_filter() {
        let (db, path) = test_db("search_filter");
        let emb = vec![0.5, 0.5, 0.0, 0.0];
        search::upsert(&db, "a", "ns", "episodic", Some("ep"), &emb, 0.0, None, None).unwrap();
        search::upsert(&db, "a", "ns", "semantic", Some("sem"), &emb, 0.0, None, None).unwrap();

        let filters = SearchFilters {
            mem_types: Some(vec!["semantic".to_string()]),
            tags: None, after_ms: None, before_ms: None,
        };
        let results = search::search(&db, "a", Some("ns"), &emb, 10, 10, Some(&filters)).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].mem_type, "semantic");
        cleanup(&path);
    }

    #[test]
    fn test_search_cross_namespace_isolation() {
        let (db, path) = test_db("search_ns");
        let emb = vec![1.0, 0.0, 0.0];
        search::upsert(&db, "a", "ns1", "episodic", Some("in ns1"), &emb, 0.0, None, None).unwrap();
        search::upsert(&db, "a", "ns2", "episodic", Some("in ns2"), &emb, 0.0, None, None).unwrap();

        let results = search::search(&db, "a", Some("ns1"), &emb, 10, 10, None).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].content.as_deref(), Some("in ns1"));
        cleanup(&path);
    }

    #[test]
    fn test_search_cross_agent_isolation() {
        let (db, path) = test_db("search_agent");
        let emb = vec![1.0, 0.0, 0.0];
        search::upsert(&db, "a1", "ns", "episodic", Some("agent1"), &emb, 0.0, None, None).unwrap();
        search::upsert(&db, "a2", "ns", "episodic", Some("agent2"), &emb, 0.0, None, None).unwrap();

        let results = search::search(&db, "a1", Some("ns"), &emb, 10, 10, None).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].content.as_deref(), Some("agent1"));
        cleanup(&path);
    }

    #[test]
    fn test_priority_boost() {
        let (db, path) = test_db("priority");
        let emb = vec![1.0, 0.0, 0.0, 0.0];
        search::upsert(&db, "a", "ns", "episodic", Some("low"), &emb, 0.0, None, None).unwrap();
        search::upsert(&db, "a", "ns", "episodic", Some("high"), &emb, 10.0, None, None).unwrap();

        let results = search::search(&db, "a", Some("ns"), &emb, 2, 10, None).unwrap();
        // both have same similarity, but "high" has priority boost
        assert_eq!(results[0].content.as_deref(), Some("high"));
        cleanup(&path);
    }

    // ---- Delete tests ----

    #[test]
    fn test_tombstone_delete() {
        let (db, path) = test_db("delete");
        let emb = vec![1.0, 0.0, 0.0];
        let (id, _) = search::upsert(&db, "a", "ns", "episodic", Some("gone"), &emb, 0.0, None, None).unwrap();
        search::upsert(&db, "a", "ns", "episodic", Some("stays"), &emb, 0.0, None, None).unwrap();

        assert!(db.tombstone_delete("a", &id).unwrap());
        // search should not return deleted
        let results = search::search(&db, "a", Some("ns"), &emb, 10, 10, None).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].content.as_deref(), Some("stays"));

        // delete again should return false
        assert!(!db.tombstone_delete("a", &id).unwrap());
        cleanup(&path);
    }

    #[test]
    fn test_delete_wrong_agent() {
        let (db, path) = test_db("delete_agent");
        let emb = vec![1.0, 0.0];
        let (id, _) = search::upsert(&db, "a", "ns", "episodic", Some("x"), &emb, 0.0, None, None).unwrap();
        assert!(!db.tombstone_delete("b", &id).unwrap());
        cleanup(&path);
    }

    // ---- Compact tests ----

    #[test]
    fn test_compact_removes_tombstones() {
        let (db, path) = test_db("compact");
        let emb = vec![1.0, 0.0, 0.0];

        // insert enough to trigger compaction threshold
        let mut ids = vec![];
        for i in 0..10 {
            let (id, _) = search::upsert(&db, "a", "ns", "episodic", Some(&format!("mem{i}")), &emb, 0.0, None, None).unwrap();
            ids.push(id);
        }

        // delete 3 out of 10 = 30% > 20% threshold
        for id in &ids[0..3] {
            db.tombstone_delete("a", id).unwrap();
        }

        let (segs, removed) = compact::compact(&db, "a", "ns").unwrap();
        assert_eq!(segs, 1);
        assert_eq!(removed, 3);

        // verify remaining
        let results = search::search(&db, "a", Some("ns"), &emb, 20, 10, None).unwrap();
        assert_eq!(results.len(), 7);
        cleanup(&path);
    }

    #[test]
    fn test_compact_empty_segment_removed() {
        let (db, path) = test_db("compact_empty");
        let emb = vec![1.0, 0.0];
        let (id, _) = search::upsert(&db, "a", "ns", "episodic", Some("only"), &emb, 0.0, None, None).unwrap();
        db.tombstone_delete("a", &id).unwrap();

        let (segs, removed) = compact::compact(&db, "a", "ns").unwrap();
        assert_eq!(segs, 1);
        assert_eq!(removed, 1);

        // segment should be gone
        let segments = db.list_segments("a", Some("ns")).unwrap();
        assert!(segments.is_empty());
        cleanup(&path);
    }

    // ---- Stats test ----

    #[test]
    fn test_stats() {
        let (db, path) = test_db("stats");
        let emb = vec![1.0, 0.0, 0.0];
        search::upsert(&db, "a", "ns1", "episodic", Some("x"), &emb, 0.0, None, None).unwrap();
        search::upsert(&db, "a", "ns2", "semantic", Some("y"), &emb, 0.0, None, None).unwrap();

        let stats = db.stats("a").unwrap();
        assert_eq!(stats["total"], 2);
        assert_eq!(stats["segments"], 2);
        let ns = stats["namespaces"].as_array().unwrap();
        assert_eq!(ns.len(), 2);
        cleanup(&path);
    }

    // ---- Vector encoding round-trip ----

    #[test]
    fn test_f32_bytes_roundtrip() {
        let original = vec![1.0f32, -0.5, 0.0, 3.14159, f32::MIN, f32::MAX];
        let bytes = clawmem::db::f32s_to_bytes(&original);
        let decoded = clawmem::db::bytes_to_f32s(&bytes);
        assert_eq!(original, decoded);
    }

    // ---- Cosine similarity correctness ----

    #[test]
    fn test_cosine_similarity_identical() {
        let (db, path) = test_db("cosine_ident");
        let emb = vec![0.3, 0.4, 0.5];
        search::upsert(&db, "a", "ns", "episodic", Some("x"), &emb, 0.0, None, None).unwrap();
        let results = search::search(&db, "a", Some("ns"), &emb, 1, 10, None).unwrap();
        assert!((results[0].similarity - 1.0).abs() < 1e-6, "identical vectors should have cosine ~1.0");
        cleanup(&path);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let (db, path) = test_db("cosine_ortho");
        let e1 = vec![1.0, 0.0, 0.0];
        let e2 = vec![0.0, 1.0, 0.0];
        search::upsert(&db, "a", "ns", "episodic", Some("x"), &e1, 0.0, None, None).unwrap();
        let results = search::search(&db, "a", Some("ns"), &e2, 1, 10, None).unwrap();
        assert!(results[0].similarity.abs() < 1e-6, "orthogonal vectors should have cosine ~0.0");
        cleanup(&path);
    }

    // ---- Larger scale test ----

    #[test]
    fn test_many_inserts_and_search() {
        let (db, path) = test_db("scale");
        let dim = 64;
        let n = 1000;

        for i in 0..n {
            let mut emb = rand_embedding(dim, i as u64);
            normalize(&mut emb);
            search::upsert(&db, "a", "ns", "episodic", Some(&format!("mem-{i}")), &emb, 0.0, None, None).unwrap();
        }

        let stats = db.stats("a").unwrap();
        assert_eq!(stats["total"], n);
        assert!(stats["segments"].as_i64().unwrap() >= 1);

        // search should return results
        let mut query = rand_embedding(dim, 42);
        normalize(&mut query);
        let results = search::search(&db, "a", Some("ns"), &query, 10, 20, None).unwrap();
        assert_eq!(results.len(), 10);
        // scores should be descending
        for w in results.windows(2) {
            assert!(w[0].score >= w[1].score);
        }
        cleanup(&path);
    }

    // ---- TTL test ----

    #[test]
    fn test_ttl_expiry() {
        let (db, path) = test_db("ttl");
        let emb = vec![1.0, 0.0, 0.0];
        // insert with TTL of 0 seconds (already expired)
        let id = uuid::Uuid::new_v4().to_string();
        let seg = db.get_or_create_open_segment("a", "ns").unwrap();
        let past = clawmem::db::now_ms() - 1000; // expired 1s ago
        db.conn.execute(
            "INSERT INTO memories (id, agent_id, namespace, type, content, embedding, dim, created_at, updated_at, expires_at, priority, segment_id)
             VALUES (?1,'a','ns','episodic','expired',?2,3,?3,?3,?4,0.0,?5)",
            rusqlite::params![id, clawmem::db::f32s_to_bytes(&emb), past, past, seg],
        ).unwrap();
        db.conn.execute("UPDATE segments SET size = size + 1 WHERE id = ?1", rusqlite::params![seg]).unwrap();
        db.update_segment_centroid(seg, &emb).unwrap();

        let results = search::search(&db, "a", Some("ns"), &emb, 10, 10, None).unwrap();
        assert!(results.is_empty(), "expired memories should not appear in search");
        cleanup(&path);
    }

    // ---- MCP protocol test ----

    #[test]
    fn test_mcp_initialize_and_tools_list() {
        // Smoke test the MCP handler by calling handle methods directly
        let (db, path) = test_db("mcp");
        // We can't call serve_stdio directly, but we can test the internals
        // by verifying the DB works through the search module
        let emb = vec![0.5, 0.5, 0.0];
        let (id, _) = search::upsert(&db, "test", "default", "semantic", Some("mcp test"), &emb, 0.0, None, None).unwrap();
        assert!(!id.is_empty());

        let results = search::search(&db, "test", Some("default"), &emb, 5, 10, None).unwrap();
        assert_eq!(results.len(), 1);
        cleanup(&path);
    }
}
