//! Embedding generation: CIMBA moments (zero-dep default) + OpenAI (optional upgrade).
//!
//! CIMBA moments: project text tokens onto a circle, compute low-order Fourier moments.
//! Captures distributional/structural similarity — good for topic and near-duplicate matching.
//!
//! OpenAI: text-embedding-3-small (1536 dims), real semantic search.

use anyhow::{Result, anyhow};
use std::f32::consts::PI;

/// Default dimension for CIMBA moment embeddings (real+imag parts of S harmonics)
pub const CIMBA_DIM: usize = 64; // S=32 harmonics → 64 floats (re,im pairs)
pub const CIMBA_HARMONICS: usize = 32;

/// OpenAI embedding dimension
pub const OPENAI_DIM: usize = 1536;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EmbedMode {
    Cimba,
    OpenAI,
}

/// Generate a CIMBA moment embedding from text.
///
/// Algorithm:
/// 1. Tokenize text into words (lowercased, stripped of punctuation)
/// 2. Hash each token to a phase θ ∈ [0, 2π) on the unit circle
/// 3. Compute complex Fourier moments: m_k = (1/N) Σ e^{ikθ_j} for k=1..S
/// 4. Output as interleaved [Re(m_1), Im(m_1), Re(m_2), Im(m_2), ..., Re(m_S), Im(m_S)]
/// 5. L2-normalize the result
///
/// This captures the "distributional shape" of token phases — texts with similar
/// vocabulary and structure produce similar moment signatures.
pub fn cimba_embed(text: &str) -> Vec<f32> {
    let tokens = tokenize(text);
    if tokens.is_empty() {
        return vec![0.0; CIMBA_DIM];
    }

    let n = tokens.len() as f32;
    let mut embedding = Vec::with_capacity(CIMBA_DIM);

    for k in 1..=CIMBA_HARMONICS {
        let mut re_sum: f32 = 0.0;
        let mut im_sum: f32 = 0.0;

        for token in &tokens {
            let phase = token_to_phase(token);
            let angle = k as f32 * phase;
            re_sum += angle.cos();
            im_sum += angle.sin();
        }

        // Normalize by token count (moment = average)
        embedding.push(re_sum / n);
        embedding.push(im_sum / n);
    }

    // L2-normalize
    l2_normalize(&mut embedding);
    embedding
}

/// Generate embeddings via OpenAI text-embedding-3-small API.
pub fn openai_embed(text: &str, api_key: &str) -> Result<Vec<f32>> {
    // Use ureq for synchronous HTTP (no tokio needed)
    let body = serde_json::json!({
        "model": "text-embedding-3-small",
        "input": text,
    });

    let resp: serde_json::Value = ureq::post("https://api.openai.com/v1/embeddings")
        .header("Authorization", &format!("Bearer {api_key}"))
        .header("Content-Type", "application/json")
        .send_json(&body)?
        .body_mut()
        .read_json()?;

    let embedding_arr = resp["data"][0]["embedding"]
        .as_array()
        .ok_or_else(|| anyhow!("No embedding in response: {}", resp))?;

    let mut embedding: Vec<f32> = embedding_arr
        .iter()
        .filter_map(|v| v.as_f64().map(|f| f as f32))
        .collect();

    if embedding.len() != OPENAI_DIM {
        return Err(anyhow!(
            "Expected {} dims, got {}",
            OPENAI_DIM,
            embedding.len()
        ));
    }

    l2_normalize(&mut embedding);
    Ok(embedding)
}

/// Auto-embed: use OpenAI if key is available, fall back to CIMBA.
pub fn auto_embed(text: &str, openai_key: Option<&str>) -> Result<(Vec<f32>, EmbedMode)> {
    if let Some(key) = openai_key {
        if !key.is_empty() {
            match openai_embed(text, key) {
                Ok(emb) => return Ok((emb, EmbedMode::OpenAI)),
                Err(e) => {
                    eprintln!("OpenAI embedding failed, falling back to CIMBA: {e}");
                }
            }
        }
    }
    Ok((cimba_embed(text), EmbedMode::Cimba))
}

// ---- Internal helpers ----

/// Simple whitespace tokenizer with lowercasing and punctuation stripping.
fn tokenize(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|w| {
            w.chars()
                .filter(|c| c.is_alphanumeric() || *c == '\'')
                .collect::<String>()
                .to_lowercase()
        })
        .filter(|w| !w.is_empty())
        .collect()
}

/// Hash a token to a phase θ ∈ [0, 2π) using FNV-1a.
fn token_to_phase(token: &str) -> f32 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in token.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    // Map to [0, 2π)
    (h as f32 / u64::MAX as f32) * 2.0 * PI
}

/// L2-normalize a vector in-place.
fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-8 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cimba_basic() {
        let emb = cimba_embed("hello world");
        assert_eq!(emb.len(), CIMBA_DIM);
        // Should be L2-normalized
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cimba_empty() {
        let emb = cimba_embed("");
        assert_eq!(emb.len(), CIMBA_DIM);
        assert!(emb.iter().all(|x| *x == 0.0));
    }

    #[test]
    fn test_cimba_similar_texts() {
        let e1 = cimba_embed("PostgreSQL is our primary database");
        let e2 = cimba_embed("PostgreSQL is the primary database");
        let e3 = cimba_embed("I like pizza and pasta");

        let sim_close = cosine(&e1, &e2);
        let sim_far = cosine(&e1, &e3);

        assert!(sim_close > sim_far, "Similar texts should score higher: {sim_close} vs {sim_far}");
        assert!(sim_close > 0.7, "Near-identical texts should be very similar: {sim_close}");
    }

    #[test]
    fn test_cimba_deterministic() {
        let e1 = cimba_embed("test phrase");
        let e2 = cimba_embed("test phrase");
        assert_eq!(e1, e2);
    }

    #[test]
    fn test_tokenize() {
        let tokens = tokenize("Hello, World! It's a test.");
        assert_eq!(tokens, vec!["hello", "world", "it's", "a", "test"]);
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na < 1e-8 || nb < 1e-8 { return 0.0; }
        dot / (na * nb)
    }
}
