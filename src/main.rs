mod db;
mod types;
mod search;
mod compact;
mod mcp;
mod embed;

// Re-export for lib.rs (lib.rs has its own pub mod declarations)

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::io::Write;
use db::Db;
use types::*;

#[derive(Parser)]
#[command(name = "clawmem", version, about = "Portable local vector memory DB for OpenClaw agents")]
struct Cli {
    /// Path to database file
    #[arg(long, default_value = "clawmem.db")]
    db: String,

    /// OpenAI API key for semantic embeddings (optional, falls back to CIMBA moments)
    #[arg(long, env = "OPENAI_API_KEY")]
    openai_key: Option<String>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Initialize a new database
    Init,

    /// Insert a memory from JSON (stdin)
    Upsert {
        #[arg(long)]
        agent: String,
        #[arg(long, default_value = "default")]
        namespace: String,
    },

    /// Search memories
    Search {
        #[arg(long)]
        agent: String,
        #[arg(long)]
        namespace: Option<String>,
        #[arg(long, default_value_t = DEFAULT_K)]
        k: usize,
        #[arg(long, default_value_t = DEFAULT_TOP_SEGMENTS)]
        top_segments: usize,
    },

    /// Batch upsert from JSONL (stdin or file)
    BatchUpsert {
        #[arg(long)]
        agent: String,
        #[arg(long, default_value = "default")]
        namespace: String,
        /// Path to JSONL file (reads stdin if not provided)
        #[arg(long)]
        file: Option<String>,
    },

    /// Delete a memory by ID
    Delete {
        #[arg(long)]
        agent: String,
        #[arg(long)]
        id: String,
    },

    /// Compact a namespace
    Compact {
        #[arg(long)]
        agent: String,
        #[arg(long, default_value = "default")]
        namespace: String,
    },

    /// Show stats
    Stats {
        #[arg(long)]
        agent: String,
    },

    /// Export memories as JSONL
    Export {
        #[arg(long)]
        agent: String,
        #[arg(long)]
        namespace: Option<String>,
        /// Output file (writes to stdout if not provided)
        #[arg(long)]
        output: Option<String>,
    },

    /// Import memories from JSONL (uses batch upsert)
    Import {
        #[arg(long)]
        agent: String,
        /// Input JSONL file (reads stdin if not provided)
        #[arg(long)]
        file: Option<String>,
    },

    /// Check database integrity
    Doctor,

    /// Start MCP server (stdio)
    Serve,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let db = Db::open(&cli.db)?;

    match cli.command {
        Commands::Init => {
            println!("Database initialized at {}", cli.db);
        }
        Commands::Upsert { agent, namespace } => {
            let input: serde_json::Value = serde_json::from_reader(std::io::stdin())?;
            let mem_type = input["type"].as_str().unwrap_or("episodic");
            let content = input["content"].as_str();
            let priority = input["priority"].as_f64().unwrap_or(0.0);
            let tags = input.get("tags").map(|t| t.to_string());
            let ttl = input["ttl_seconds"].as_i64();

            // Embedding: use provided vector, or auto-generate from content
            let embedding = if let Some(arr) = input.get("embedding").and_then(|e| e.as_array()) {
                // Explicit embedding provided
                arr.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect()
            } else if let Some(text) = content {
                // Auto-embed from content text
                let (emb, mode) = embed::auto_embed(text, cli.openai_key.as_deref())?;
                eprintln!("auto-embedded via {:?} ({} dims)", mode, emb.len());
                emb
            } else {
                anyhow::bail!("Either 'embedding' array or 'content' text is required");
            };

            let (id, seg_id) = search::upsert(
                &db, &agent, &namespace, mem_type, content,
                &embedding, priority, tags.as_deref(), ttl,
            )?;
            println!("{}", serde_json::json!({"id": id, "segment_id": seg_id}));
        }
        Commands::Search { agent, namespace, k, top_segments } => {
            let input: serde_json::Value = serde_json::from_reader(std::io::stdin())?;
            let filters: Option<SearchFilters> = input.get("filters")
                .and_then(|f| serde_json::from_value(f.clone()).ok());

            // Query embedding: use provided vector, or auto-generate from query text
            let query_embedding = if let Some(arr) = input.get("query_embedding").and_then(|e| e.as_array()) {
                arr.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect()
            } else if let Some(text) = input["query"].as_str() {
                let (emb, _mode) = embed::auto_embed(text, cli.openai_key.as_deref())?;
                emb
            } else {
                anyhow::bail!("Either 'query_embedding' array or 'query' text is required");
            };

            let results = search::search(
                &db, &agent, namespace.as_deref(),
                &query_embedding, k, top_segments, filters.as_ref(),
            )?;
            println!("{}", serde_json::to_string_pretty(&results)?);
        }
        Commands::BatchUpsert { agent, namespace, file } => {
            use std::io::BufRead;
            let reader: Box<dyn std::io::Read> = match file {
                Some(ref path) => Box::new(std::fs::File::open(path)?),
                None => Box::new(std::io::stdin()),
            };
            let mut items: Vec<search::BatchItem> = Vec::new();
            for line in std::io::BufReader::new(reader).lines() {
                let line = line?;
                if line.trim().is_empty() { continue; }
                let mut item: search::BatchItem = serde_json::from_str(&line)?;
                if item.agent_id.is_empty() { item.agent_id = agent.clone(); }
                if item.namespace == "default" && namespace != "default" {
                    item.namespace = namespace.clone();
                }
                items.push(item);
            }
            let results = search::batch_upsert(&db, &items)?;
            println!("{}", serde_json::json!({"inserted": results.len()}));
        }
        Commands::Delete { agent, id } => {
            let deleted = db.tombstone_delete(&agent, &id)?;
            println!("{}", serde_json::json!({"deleted": deleted}));
        }
        Commands::Compact { agent, namespace } => {
            let (segs, removed) = compact::compact(&db, &agent, &namespace)?;
            println!("{}", serde_json::json!({"segments_compacted": segs, "records_removed": removed}));
        }
        Commands::Stats { agent } => {
            let stats = db.stats(&agent)?;
            println!("{}", serde_json::to_string_pretty(&stats)?);
        }
        Commands::Export { agent, namespace, output } => {
            let memories = db.export_memories(&agent, namespace.as_deref())?;
            let mut writer: Box<dyn std::io::Write> = match output {
                Some(ref path) => Box::new(std::fs::File::create(path)?),
                None => Box::new(std::io::stdout()),
            };
            for mem in &memories {
                writeln!(writer, "{}", serde_json::to_string(mem)?)?;
            }
            eprintln!("Exported {} memories", memories.len());
        }
        Commands::Import { agent, file } => {
            use std::io::BufRead;
            let reader: Box<dyn std::io::Read> = match file {
                Some(ref path) => Box::new(std::fs::File::open(path)?),
                None => Box::new(std::io::stdin()),
            };
            let mut items: Vec<search::BatchItem> = Vec::new();
            for line in std::io::BufReader::new(reader).lines() {
                let line = line?;
                if line.trim().is_empty() { continue; }
                let v: serde_json::Value = serde_json::from_str(&line)?;
                let embedding: Vec<f32> = serde_json::from_value(v["embedding"].clone())?;
                let tags = v.get("tags").and_then(|t| if t.is_null() { None } else { Some(t.to_string()) });
                items.push(search::BatchItem {
                    agent_id: v["agent_id"].as_str().unwrap_or(&agent).to_string(),
                    namespace: v["namespace"].as_str().unwrap_or("default").to_string(),
                    mem_type: v["type"].as_str().unwrap_or("episodic").to_string(),
                    content: v["content"].as_str().map(|s| s.to_string()),
                    embedding,
                    priority: v["priority"].as_f64().unwrap_or(0.0),
                    tags,
                    ttl_seconds: None,
                });
            }
            let results = search::batch_upsert(&db, &items)?;
            println!("{}", serde_json::json!({"imported": results.len()}));
        }
        Commands::Doctor => {
            let result: String = db.conn.query_row("PRAGMA integrity_check", [], |r| r.get(0))?;
            println!("SQLite integrity: {result}");
            let qc: String = db.conn.query_row("PRAGMA quick_check", [], |r| r.get(0))?;
            println!("Quick check: {qc}");
        }
        Commands::Serve => {
            drop(db); // close the CLI-opened connection
            mcp::serve_stdio(&cli.db)?;
        }
    }

    Ok(())
}
