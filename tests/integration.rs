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

        // Disable auto-compact so we can test manual compact
        db.set_kv("segment_compact_threshold", "0.99").unwrap();

        let mut ids = vec![];
        for i in 0..10 {
            let (id, _) = search::upsert(&db, "a", "ns", "episodic", Some(&format!("mem{i}")), &emb, 0.0, None, None).unwrap();
            ids.push(id);
        }

        // delete 3 out of 10 = 30% > 20% threshold
        for id in &ids[0..3] {
            db.tombstone_delete("a", id).unwrap();
        }

        // Now set normal threshold and compact manually
        db.set_kv("segment_compact_threshold", "0.2").unwrap();
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

        // Disable auto-compact completely
        db.set_kv("segment_compact_threshold", "2.0").unwrap();

        let (id, _) = search::upsert(&db, "a", "ns", "episodic", Some("only"), &emb, 0.0, None, None).unwrap();
        db.tombstone_delete("a", &id).unwrap();

        db.set_kv("segment_compact_threshold", "0.2").unwrap();
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

    // ---- Auto-compact and segment merge tests ----

    #[test]
    fn test_auto_compact_trigger() {
        let (db, path) = test_db("auto_compact");
        let emb = vec![1.0, 0.0, 0.0];

        // Insert 5 memories
        let mut ids = vec![];
        for i in 0..5 {
            let (id, _) = search::upsert(&db, "a", "ns", "episodic", Some(&format!("mem{i}")), &emb, 0.0, None, None).unwrap();
            ids.push(id);
        }

        // Set low threshold to trigger auto-compact easily
        db.set_kv("segment_compact_threshold", "0.2").unwrap();

        // Delete 2 out of 5 = 40% > 20% threshold → should auto-compact
        db.tombstone_delete("a", &ids[0]).unwrap();
        // First delete: 1/5 = 20%, exactly at threshold → triggers
        // After compact, tombstones are physically removed

        // Delete another
        db.tombstone_delete("a", &ids[1]).unwrap();

        // Verify: search should return only 3
        let results = search::search(&db, "a", Some("ns"), &emb, 10, 10, None).unwrap();
        assert_eq!(results.len(), 3);

        cleanup(&path);
    }

    #[test]
    fn test_merge_underfilled_segments() {
        let (db, path) = test_db("merge_segments");

        // Set small capacity so we create multiple segments
        db.set_kv("segment_capacity:ns", "3").unwrap();
        // Disable auto-compact so we control when it happens
        db.set_kv("segment_compact_threshold", "0.99").unwrap();

        let emb = vec![1.0, 0.0, 0.0];
        // Insert 9 items → should create 3 segments of 3 each (all closed except last)
        let mut ids = vec![];
        for i in 0..9 {
            let (id, _) = search::upsert(&db, "a", "ns", "episodic", Some(&format!("m{i}")), &emb, 0.0, None, None).unwrap();
            ids.push(id);
        }

        let segments_before = db.list_segments("a", Some("ns")).unwrap();
        assert_eq!(segments_before.len(), 3);

        // Tombstone-delete 2 from first segment and 2 from second
        // (don't trigger auto-compact because threshold is 99%)
        db.tombstone_delete("a", &ids[0]).unwrap();
        db.tombstone_delete("a", &ids[1]).unwrap();
        db.tombstone_delete("a", &ids[3]).unwrap();
        db.tombstone_delete("a", &ids[4]).unwrap();

        // Now set a normal threshold and compact
        db.set_kv("segment_compact_threshold", "0.2").unwrap();
        compact::compact(&db, "a", "ns").unwrap();

        // After compact + merge: two segments had 1 item each (< 50% of 3)
        // They should merge together
        let segments_after = db.list_segments("a", Some("ns")).unwrap();
        assert!(segments_after.len() < segments_before.len(),
            "should have fewer segments after merge, got {} (was {})", segments_after.len(), segments_before.len());

        // All 5 remaining memories should still be searchable
        let results = search::search(&db, "a", Some("ns"), &emb, 10, 10, None).unwrap();
        assert_eq!(results.len(), 5);

        cleanup(&path);
    }

    #[test]
    fn test_configurable_segment_capacity() {
        let (db, path) = test_db("config_capacity");

        // Set capacity to 2 for namespace "small"
        db.set_kv("segment_capacity:small", "2").unwrap();

        let emb = vec![1.0, 0.0];
        // Insert 5 items → should create 3 segments (2, 2, 1)
        for i in 0..5 {
            search::upsert(&db, "a", "small", "episodic", Some(&format!("s{i}")), &emb, 0.0, None, None).unwrap();
        }

        let segments = db.list_segments("a", Some("small")).unwrap();
        assert_eq!(segments.len(), 3, "5 items with capacity 2 should create 3 segments");

        // Default namespace should still use 512
        assert_eq!(db.get_segment_capacity("default"), 512);
        assert_eq!(db.get_segment_capacity("small"), 2);

        cleanup(&path);
    }

    // ---- Export/Import tests ----

    #[test]
    fn test_export_import_roundtrip() {
        let (db1, path1) = test_db("export_src");
        let (db2, path2) = test_db("import_dst");

        // Populate source DB
        let emb1 = vec![1.0, 0.0, 0.0, 0.0];
        let emb2 = vec![0.0, 1.0, 0.0, 0.0];
        search::upsert(&db1, "a", "ns", "episodic", Some("memory one"), &emb1, 0.5,
            Some(r#"{"project":"test"}"#), None).unwrap();
        search::upsert(&db1, "a", "ns", "semantic", Some("memory two"), &emb2, 1.0,
            None, None).unwrap();

        // Export
        let exported = db1.export_memories("a", Some("ns")).unwrap();
        assert_eq!(exported.len(), 2);

        // Write to JSONL string, then parse as BatchItems and import
        let mut items: Vec<search::BatchItem> = Vec::new();
        for mem in &exported {
            let embedding: Vec<f32> = serde_json::from_value(mem["embedding"].clone()).unwrap();
            let tags = mem.get("tags").and_then(|t| if t.is_null() { None } else { Some(t.to_string()) });
            items.push(search::BatchItem {
                agent_id: "b".to_string(), // different agent
                namespace: mem["namespace"].as_str().unwrap().to_string(),
                mem_type: mem["type"].as_str().unwrap().to_string(),
                content: mem["content"].as_str().map(|s| s.to_string()),
                embedding,
                priority: mem["priority"].as_f64().unwrap_or(0.0),
                tags,
                ttl_seconds: None,
            });
        }
        search::batch_upsert(&db2, &items).unwrap();

        // Verify import
        let stats = db2.stats("b").unwrap();
        assert_eq!(stats["total"], 2);

        // Search should yield same results
        let results = search::search(&db2, "b", Some("ns"), &emb1, 2, 10, None).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].content.as_deref(), Some("memory one"));

        // Tag preserved
        let filters = SearchFilters {
            mem_types: None,
            tags: Some(serde_json::json!({"project": "test"})),
            after_ms: None, before_ms: None,
        };
        let tagged = search::search(&db2, "b", Some("ns"), &emb1, 10, 10, Some(&filters)).unwrap();
        assert_eq!(tagged.len(), 1);
        assert_eq!(tagged[0].content.as_deref(), Some("memory one"));

        cleanup(&path1);
        cleanup(&path2);
    }

    // ---- Batch upsert tests ----

    #[test]
    fn test_batch_upsert() {
        let (db, path) = test_db("batch_upsert");
        let dim = 8;
        let mut items: Vec<search::BatchItem> = Vec::new();
        for i in 0..100 {
            let mut emb = rand_embedding(dim, i as u64);
            normalize(&mut emb);
            items.push(search::BatchItem {
                agent_id: "a".to_string(),
                namespace: "ns".to_string(),
                mem_type: "episodic".to_string(),
                content: Some(format!("batch-mem-{i}")),
                embedding: emb,
                priority: 0.0,
                tags: None,
                ttl_seconds: None,
            });
        }

        let results = search::batch_upsert(&db, &items).unwrap();
        assert_eq!(results.len(), 100);

        // All IDs should be unique
        let ids: std::collections::HashSet<_> = results.iter().map(|(id, _)| id.clone()).collect();
        assert_eq!(ids.len(), 100);

        // Stats should show 100 memories
        let stats = db.stats("a").unwrap();
        assert_eq!(stats["total"], 100);

        // Search should work
        let query = &items[42].embedding;
        let search_results = search::search(&db, "a", Some("ns"), query, 5, 10, None).unwrap();
        assert_eq!(search_results.len(), 5);
        // Top result should be the exact match
        assert_eq!(search_results[0].content.as_deref(), Some("batch-mem-42"));

        // Verify centroids exist for all segments
        let segments = db.list_segments("a", Some("ns")).unwrap();
        for seg in &segments {
            assert!(seg.centroid.is_some(), "segment {} should have centroid", seg.id);
        }

        cleanup(&path);
    }

    // ---- Tag filtering tests ----

    #[test]
    fn test_search_tag_filter() {
        let (db, path) = test_db("tag_filter");
        let emb = vec![1.0, 0.0, 0.0, 0.0];

        // Insert with different tags
        search::upsert(&db, "a", "ns", "episodic", Some("tagged-project"),
            &emb, 0.0, Some(r#"{"project":"brain-arch","env":"prod"}"#), None).unwrap();
        search::upsert(&db, "a", "ns", "episodic", Some("tagged-other"),
            &emb, 0.0, Some(r#"{"project":"clawmem","env":"dev"}"#), None).unwrap();
        search::upsert(&db, "a", "ns", "episodic", Some("no-tags"),
            &emb, 0.0, None, None).unwrap();

        // Filter by project=brain-arch
        let filters = SearchFilters {
            mem_types: None,
            tags: Some(serde_json::json!({"project": "brain-arch"})),
            after_ms: None, before_ms: None,
        };
        let results = search::search(&db, "a", Some("ns"), &emb, 10, 10, Some(&filters)).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].content.as_deref(), Some("tagged-project"));

        // Filter by env=dev
        let filters2 = SearchFilters {
            mem_types: None,
            tags: Some(serde_json::json!({"env": "dev"})),
            after_ms: None, before_ms: None,
        };
        let results2 = search::search(&db, "a", Some("ns"), &emb, 10, 10, Some(&filters2)).unwrap();
        assert_eq!(results2.len(), 1);
        assert_eq!(results2[0].content.as_deref(), Some("tagged-other"));

        // Filter by both (AND): project=brain-arch AND env=prod
        let filters3 = SearchFilters {
            mem_types: None,
            tags: Some(serde_json::json!({"project": "brain-arch", "env": "prod"})),
            after_ms: None, before_ms: None,
        };
        let results3 = search::search(&db, "a", Some("ns"), &emb, 10, 10, Some(&filters3)).unwrap();
        assert_eq!(results3.len(), 1);
        assert_eq!(results3[0].content.as_deref(), Some("tagged-project"));

        // Filter by non-matching combo: project=brain-arch AND env=dev → 0 results
        let filters4 = SearchFilters {
            mem_types: None,
            tags: Some(serde_json::json!({"project": "brain-arch", "env": "dev"})),
            after_ms: None, before_ms: None,
        };
        let results4 = search::search(&db, "a", Some("ns"), &emb, 10, 10, Some(&filters4)).unwrap();
        assert_eq!(results4.len(), 0);

        // No tag filter → all 3 returned
        let results_all = search::search(&db, "a", Some("ns"), &emb, 10, 10, None).unwrap();
        assert_eq!(results_all.len(), 3);

        cleanup(&path);
    }

    // ---- Concurrent stress tests ----

    #[test]
    fn test_concurrent_stress() {
        use std::sync::{Arc, Barrier};
        use std::thread;

        let db_path = format!("/tmp/clawmem_stress_{}.db", std::process::id());
        // Initialize DB
        {
            let db = Db::open(&db_path).unwrap();
            // Disable auto-compact to avoid contention
            db.set_kv("segment_compact_threshold", "2.0").unwrap();
        }

        let n_writers = 4;
        let writes_per = 100;
        let n_readers = 4;
        let reads_per = 50;
        let dim = 8;
        let path = Arc::new(db_path.clone());
        let barrier = Arc::new(Barrier::new(n_writers + n_readers));

        let mut handles = vec![];

        // Writer threads
        for w in 0..n_writers {
            let path = Arc::clone(&path);
            let barrier = Arc::clone(&barrier);
            handles.push(thread::spawn(move || {
                let db = Db::open(&path).unwrap();
                barrier.wait();
                for i in 0..writes_per {
                    let mut emb = vec![0.0f32; dim];
                    emb[w % dim] = 1.0;
                    emb[(w + i) % dim] += 0.1 * i as f32;
                    let content = format!("w{w}-m{i}");
                    // Retry on busy
                    for attempt in 0..5 {
                        match search::upsert(&db, "stress", "ns", "episodic",
                                            Some(&content), &emb, 0.0, None, None) {
                            Ok(_) => break,
                            Err(e) if attempt < 4 => {
                                thread::sleep(std::time::Duration::from_millis(10 * (attempt + 1) as u64));
                                if attempt == 4 { panic!("writer {w} failed after retries: {e}"); }
                            }
                            Err(e) => panic!("writer {w} item {i}: {e}"),
                        }
                    }
                }
            }));
        }

        // Reader threads
        for r in 0..n_readers {
            let path = Arc::clone(&path);
            let barrier = Arc::clone(&barrier);
            handles.push(thread::spawn(move || {
                let db = Db::open(&path).unwrap();
                barrier.wait();
                let mut emb = vec![0.0f32; dim];
                emb[r % dim] = 1.0;
                for _ in 0..reads_per {
                    // Search should not panic or return corrupted data
                    let results = search::search(&db, "stress", Some("ns"), &emb, 10, 10, None).unwrap();
                    for result in &results {
                        assert!(result.score >= 0.0, "negative score!");
                        assert!(result.content.is_some(), "null content!");
                    }
                    thread::sleep(std::time::Duration::from_millis(1));
                }
            }));
        }

        for h in handles {
            h.join().expect("thread panicked");
        }

        // Verify all 400 memories present
        let db = Db::open(&db_path).unwrap();
        let stats = db.stats("stress").unwrap();
        assert_eq!(stats["total"], n_writers as i64 * writes_per as i64,
            "expected {} memories, got {}", n_writers * writes_per, stats["total"]);

        // Search should still work correctly
        let emb = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let results = search::search(&db, "stress", Some("ns"), &emb, 10, 20, None).unwrap();
        assert_eq!(results.len(), 10);

        let _ = std::fs::remove_file(&db_path);
    }

    // ---- MCP Protocol Tests ----

    fn mcp_roundtrip(db_path: &str, requests: &[serde_json::Value]) -> Vec<serde_json::Value> {
        use std::io::{BufRead, Write};
        use std::process::{Command, Stdio};

        let binary = env!("CARGO_BIN_EXE_clawmem");
        let mut child = Command::new(binary)
            .args(&["--db", db_path, "serve"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("failed to spawn clawmem serve");

        let mut stdin = child.stdin.take().unwrap();
        let stdout = child.stdout.take().unwrap();

        // Write all requests
        for req in requests {
            writeln!(stdin, "{}", serde_json::to_string(req).unwrap()).unwrap();
        }
        drop(stdin); // close stdin to signal EOF

        // Read all responses
        let reader = std::io::BufReader::new(stdout);
        let responses: Vec<serde_json::Value> = reader.lines()
            .filter_map(|l| l.ok())
            .filter(|l| !l.trim().is_empty())
            .filter_map(|l| serde_json::from_str(&l).ok())
            .collect();

        child.wait().unwrap();
        responses
    }

    #[test]
    fn test_mcp_protocol_initialize() {
        let db_path = format!("/tmp/clawmem_mcp_init_{}.db", std::process::id());
        let responses = mcp_roundtrip(&db_path, &[
            serde_json::json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}),
        ]);
        assert_eq!(responses.len(), 1);
        assert_eq!(responses[0]["id"], 1);
        assert!(responses[0]["result"]["serverInfo"]["name"].as_str().unwrap().contains("clawmem"));
        assert!(responses[0]["result"]["capabilities"]["tools"].is_object());
        let _ = std::fs::remove_file(&db_path);
    }

    #[test]
    fn test_mcp_protocol_tools_list() {
        let db_path = format!("/tmp/clawmem_mcp_tools_{}.db", std::process::id());
        let responses = mcp_roundtrip(&db_path, &[
            serde_json::json!({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}),
        ]);
        assert_eq!(responses.len(), 1);
        let tools = responses[0]["result"]["tools"].as_array().unwrap();
        let names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
        assert!(names.contains(&"memory.upsert"));
        assert!(names.contains(&"memory.search"));
        assert!(names.contains(&"memory.delete"));
        assert!(names.contains(&"memory.compact"));
        assert!(names.contains(&"memory.stats"));
        assert!(names.contains(&"memory.batch_upsert"));
        let _ = std::fs::remove_file(&db_path);
    }

    #[test]
    fn test_mcp_protocol_full_roundtrip() {
        let db_path = format!("/tmp/clawmem_mcp_full_{}.db", std::process::id());
        let emb = vec![1.0, 0.0, 0.0, 0.0];
        let responses = mcp_roundtrip(&db_path, &[
            // Initialize
            serde_json::json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}),
            // Upsert
            serde_json::json!({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
                "name": "memory.upsert",
                "arguments": {"agent_id": "test", "embedding": emb, "content": "hello world", "type": "episodic"}
            }}),
            // Search
            serde_json::json!({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {
                "name": "memory.search",
                "arguments": {"agent_id": "test", "query_embedding": emb, "k": 5}
            }}),
            // Stats
            serde_json::json!({"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {
                "name": "memory.stats",
                "arguments": {"agent_id": "test"}
            }}),
            // Delete (we'll get the ID from upsert result)
            serde_json::json!({"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {
                "name": "memory.delete",
                "arguments": {"agent_id": "test", "id": "nonexistent"}
            }}),
            // Compact
            serde_json::json!({"jsonrpc": "2.0", "id": 6, "method": "tools/call", "params": {
                "name": "memory.compact",
                "arguments": {"agent_id": "test", "namespace": "default"}
            }}),
        ]);

        assert_eq!(responses.len(), 6);

        // All should have valid jsonrpc and matching ids
        for (i, resp) in responses.iter().enumerate() {
            assert_eq!(resp["jsonrpc"], "2.0");
            assert_eq!(resp["id"], (i + 1) as i64);
            assert!(resp.get("error").is_none(), "unexpected error on request {}: {:?}", i+1, resp["error"]);
        }

        // Upsert result should have an id
        let upsert_text: serde_json::Value = serde_json::from_str(
            responses[1]["result"]["content"][0]["text"].as_str().unwrap()
        ).unwrap();
        assert!(upsert_text["id"].is_string());

        // Search should return 1 result
        let search_text: serde_json::Value = serde_json::from_str(
            responses[2]["result"]["content"][0]["text"].as_str().unwrap()
        ).unwrap();
        assert_eq!(search_text["results"].as_array().unwrap().len(), 1);

        // Stats should show total=1
        let stats_text: serde_json::Value = serde_json::from_str(
            responses[3]["result"]["content"][0]["text"].as_str().unwrap()
        ).unwrap();
        assert_eq!(stats_text["total"], 1);

        let _ = std::fs::remove_file(&db_path);
    }

    #[test]
    fn test_mcp_protocol_error_handling() {
        let db_path = format!("/tmp/clawmem_mcp_err_{}.db", std::process::id());
        let responses = mcp_roundtrip(&db_path, &[
            // Malformed JSON (not valid, but we send as string)
            serde_json::json!({"jsonrpc": "2.0", "id": 1, "method": "nonexistent_method", "params": {}}),
        ]);
        assert_eq!(responses.len(), 1);
        assert!(responses[0].get("error").is_some());
        let _ = std::fs::remove_file(&db_path);
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
