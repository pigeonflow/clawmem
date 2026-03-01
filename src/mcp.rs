use anyhow::Result;
use serde_json::{json, Value};
use std::io::{BufRead, Write};
use crate::db::Db;
use crate::types::*;

/// JSON-RPC 2.0 MCP server over stdio
pub fn serve_stdio(db_path: &str) -> Result<()> {
    let db = Db::open(db_path)?;
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();

    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() { continue; }

        let req: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                write_response(&mut stdout, json!({
                    "jsonrpc": "2.0", "id": null,
                    "error": {"code": -32700, "message": format!("Parse error: {e}")}
                }))?;
                continue;
            }
        };

        let id = req.get("id").cloned().unwrap_or(Value::Null);
        let method = req.get("method").and_then(|m| m.as_str()).unwrap_or("");
        let params = req.get("params").cloned().unwrap_or(json!({}));

        let result = handle_method(&db, method, &params);
        match result {
            Ok(val) => write_response(&mut stdout, json!({
                "jsonrpc": "2.0", "id": id, "result": val
            }))?,
            Err(e) => write_response(&mut stdout, json!({
                "jsonrpc": "2.0", "id": id,
                "error": {"code": -32000, "message": e.to_string()}
            }))?,
        }
    }
    Ok(())
}

fn handle_method(db: &Db, method: &str, params: &Value) -> Result<Value> {
    match method {
        "initialize" => Ok(json!({
            "protocolVersion": "2024-11-05",
            "capabilities": { "tools": {} },
            "serverInfo": { "name": "clawmem", "version": env!("CARGO_PKG_VERSION") }
        })),
        "tools/list" => Ok(json!({
            "tools": [
                tool_def("memory.upsert", "Store a memory with embedding", json!({
                    "type": "object",
                    "properties": {
                        "agent_id": {"type": "string"},
                        "namespace": {"type": "string", "default": "default"},
                        "type": {"type": "string", "enum": ["episodic","semantic","procedural","tool_log"]},
                        "content": {"type": "string"},
                        "embedding": {"type": "array", "items": {"type": "number"}},
                        "priority": {"type": "number", "default": 0.0},
                        "tags": {"type": "object"},
                        "ttl_seconds": {"type": "integer"}
                    },
                    "required": ["agent_id", "embedding"]
                })),
                tool_def("memory.search", "Search memories by embedding similarity", json!({
                    "type": "object",
                    "properties": {
                        "agent_id": {"type": "string"},
                        "namespace": {"type": "string"},
                        "query_embedding": {"type": "array", "items": {"type": "number"}},
                        "k": {"type": "integer", "default": 10},
                        "filters": {"type": "object"}
                    },
                    "required": ["agent_id", "query_embedding"]
                })),
                tool_def("memory.delete", "Delete a memory by ID", json!({
                    "type": "object",
                    "properties": {
                        "agent_id": {"type": "string"},
                        "id": {"type": "string"}
                    },
                    "required": ["agent_id", "id"]
                })),
                tool_def("memory.compact", "Compact a namespace", json!({
                    "type": "object",
                    "properties": {
                        "agent_id": {"type": "string"},
                        "namespace": {"type": "string", "default": "default"}
                    },
                    "required": ["agent_id"]
                })),
                tool_def("memory.stats", "Get memory statistics", json!({
                    "type": "object",
                    "properties": { "agent_id": {"type": "string"} },
                    "required": ["agent_id"]
                })),
            ]
        })),
        "tools/call" => {
            let tool_name = params.get("name").and_then(|n| n.as_str()).unwrap_or("");
            let args = params.get("arguments").cloned().unwrap_or(json!({}));
            let result = handle_tool(db, tool_name, &args)?;
            Ok(json!({ "content": [{"type": "text", "text": result.to_string()}] }))
        }
        "notifications/initialized" | "notifications/cancelled" => Ok(json!({})),
        _ => anyhow::bail!("Unknown method: {method}"),
    }
}

fn handle_tool(db: &Db, name: &str, args: &Value) -> Result<Value> {
    match name {
        "memory.upsert" => {
            let agent_id = args["agent_id"].as_str().ok_or_else(|| anyhow::anyhow!("missing agent_id"))?;
            let namespace = args["namespace"].as_str().unwrap_or("default");
            let mem_type = args["type"].as_str().unwrap_or("episodic");
            let content = args["content"].as_str();
            let embedding: Vec<f32> = serde_json::from_value(args["embedding"].clone())?;
            let priority = args["priority"].as_f64().unwrap_or(0.0);
            let tags = args.get("tags").map(|t| t.to_string());
            let ttl = args["ttl_seconds"].as_i64();

            let (id, seg_id) = crate::search::upsert(
                db, agent_id, namespace, mem_type, content,
                &embedding, priority, tags.as_deref(), ttl,
            )?;
            Ok(json!({"id": id, "segment_id": seg_id}))
        }
        "memory.search" => {
            let agent_id = args["agent_id"].as_str().ok_or_else(|| anyhow::anyhow!("missing agent_id"))?;
            let namespace = args["namespace"].as_str();
            let query_embedding: Vec<f32> = serde_json::from_value(args["query_embedding"].clone())?;
            let k = args["k"].as_u64().unwrap_or(DEFAULT_K as u64) as usize;
            let filters: Option<SearchFilters> = args.get("filters")
                .and_then(|f| serde_json::from_value(f.clone()).ok());

            let results = crate::search::search(
                db, agent_id, namespace, &query_embedding, k, DEFAULT_TOP_SEGMENTS, filters.as_ref(),
            )?;
            Ok(json!({"results": results}))
        }
        "memory.delete" => {
            let agent_id = args["agent_id"].as_str().ok_or_else(|| anyhow::anyhow!("missing agent_id"))?;
            let id = args["id"].as_str().ok_or_else(|| anyhow::anyhow!("missing id"))?;
            let deleted = db.tombstone_delete(agent_id, id)?;
            Ok(json!({"deleted": deleted}))
        }
        "memory.compact" => {
            let agent_id = args["agent_id"].as_str().ok_or_else(|| anyhow::anyhow!("missing agent_id"))?;
            let namespace = args["namespace"].as_str().unwrap_or("default");
            let (segs, removed) = crate::compact::compact(db, agent_id, namespace)?;
            Ok(json!({"segments_compacted": segs, "records_removed": removed}))
        }
        "memory.stats" => {
            let agent_id = args["agent_id"].as_str().ok_or_else(|| anyhow::anyhow!("missing agent_id"))?;
            db.stats(agent_id)
        }
        _ => anyhow::bail!("Unknown tool: {name}"),
    }
}

fn tool_def(name: &str, desc: &str, schema: Value) -> Value {
    json!({"name": name, "description": desc, "inputSchema": schema})
}

fn write_response(out: &mut impl Write, resp: Value) -> Result<()> {
    let s = serde_json::to_string(&resp)?;
    writeln!(out, "{s}")?;
    out.flush()?;
    Ok(())
}
