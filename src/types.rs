use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    pub id: String,
    pub agent_id: String,
    pub namespace: String,
    #[serde(rename = "type")]
    pub mem_type: MemoryType,
    pub content: Option<String>,
    pub embedding: Vec<f32>,
    pub dim: usize,
    pub created_at: i64,
    pub updated_at: i64,
    pub expires_at: Option<i64>,
    pub priority: f64,
    pub tags: Option<serde_json::Value>,
    pub segment_id: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum MemoryType {
    Episodic,
    Semantic,
    Procedural,
    ToolLog,
}

impl MemoryType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Episodic => "episodic",
            Self::Semantic => "semantic",
            Self::Procedural => "procedural",
            Self::ToolLog => "tool_log",
        }
    }
    pub fn from_str(s: &str) -> anyhow::Result<Self> {
        match s {
            "episodic" => Ok(Self::Episodic),
            "semantic" => Ok(Self::Semantic),
            "procedural" => Ok(Self::Procedural),
            "tool_log" => Ok(Self::ToolLog),
            _ => anyhow::bail!("unknown memory type: {s}"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Segment {
    pub id: i64,
    pub agent_id: String,
    pub namespace: String,
    pub created_at: i64,
    pub closed_at: Option<i64>,
    pub size: i64,
    pub tombstones: i64,
    pub centroid: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchFilters {
    #[serde(rename = "type")]
    pub mem_types: Option<Vec<String>>,
    pub tags: Option<serde_json::Value>,
    pub after_ms: Option<i64>,
    pub before_ms: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: String,
    pub score: f64,
    pub similarity: f64,
    pub content: Option<String>,
    pub mem_type: String,
    pub created_at: i64,
    pub tags: Option<serde_json::Value>,
}

pub const DEFAULT_SEGMENT_CAPACITY: i64 = 512;
pub const DEFAULT_TOP_SEGMENTS: usize = 20;
pub const DEFAULT_K: usize = 10;
pub const DEFAULT_HALF_LIFE_MS: f64 = 7.0 * 24.0 * 3600.0 * 1000.0;
pub const DEFAULT_RECENCY_WEIGHT: f64 = 0.1;
pub const DEFAULT_PRIORITY_WEIGHT: f64 = 0.05;
