//! QLANG Provider Registry — discovers and manages LLM providers.
//!
//! This module provides a unified interface to multiple LLM backends
//! (local Ollama, OpenAI, Anthropic, etc.) with automatic discovery,
//! cost tracking, and data-class-aware routing.

use std::collections::HashMap;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Core enums
// ---------------------------------------------------------------------------

/// Security classification for data being sent to a model.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DataClass {
    /// Public data — can be sent anywhere
    Public,
    /// Internal data — prefer local, allow trusted cloud
    Internal,
    /// Confidential — local only
    Confidential,
}

/// What a model can do.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Capability {
    TextGeneration,
    CodeGeneration,
    Summarization,
    Embedding,
    ImageGeneration,
    Classification,
    Custom(String),
}

/// How much we trust a provider with sensitive data.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum TrustLevel {
    /// Runs on user's machine
    Local,
    /// Cloud provider with data-processing agreement
    Trusted,
    /// Public cloud, no special agreement
    Untrusted,
}

/// Preference for model selection.
#[derive(Debug, Clone)]
pub enum SelectionPreference {
    /// Prefer a local model
    Local,
    /// Pick the cheapest that can do the job
    Cheapest,
    /// Pick the fastest
    Fastest,
    /// Pick the highest quality
    BestQuality,
}

// ---------------------------------------------------------------------------
// Model / Provider config
// ---------------------------------------------------------------------------

/// Configuration for a single model exposed by a provider.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub model_id: String,
    pub display_name: String,
    pub capabilities: Vec<Capability>,
    pub cost_per_million_tokens: f64,
    pub context_window: usize,
    pub quality_score: f32, // 0.0-1.0 subjective quality ranking
}

/// A provider that can serve LLM requests.
#[derive(Debug, Clone)]
pub struct Provider {
    pub id: String,
    pub name: String,
    pub trust_level: TrustLevel,
    pub models: Vec<ModelConfig>,
    pub base_url: String,
    pub api_key: Option<String>,
    pub available: bool,
}

// ---------------------------------------------------------------------------
// Response
// ---------------------------------------------------------------------------

/// Response from an LLM call.
#[derive(Debug, Clone)]
pub struct LlmResponse {
    pub provider_id: String,
    pub model_id: String,
    pub text: String,
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub cost: f64,
    pub latency_ms: u64,
}

// ---------------------------------------------------------------------------
// Cost tracking
// ---------------------------------------------------------------------------

/// Tracks cumulative spending across providers.
#[derive(Debug, Clone)]
pub struct CostTracker {
    pub total_spent: f64,
    pub budget_limit: f64,
    pub by_provider: HashMap<String, f64>,
    pub request_count: usize,
}

impl CostTracker {
    pub fn new(budget: f64) -> Self {
        Self {
            total_spent: 0.0,
            budget_limit: budget,
            by_provider: HashMap::new(),
            request_count: 0,
        }
    }

    pub fn record(&mut self, provider_id: &str, cost: f64) {
        self.total_spent += cost;
        *self
            .by_provider
            .entry(provider_id.to_string())
            .or_insert(0.0) += cost;
        self.request_count += 1;
    }

    pub fn remaining(&self) -> f64 {
        (self.budget_limit - self.total_spent).max(0.0)
    }
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

/// Central registry that discovers providers and routes requests.
pub struct ProviderRegistry {
    pub providers: Vec<Provider>,
    pub cost_tracker: CostTracker,
}

impl ProviderRegistry {
    pub fn new(budget: f64) -> Self {
        Self {
            providers: Vec::new(),
            cost_tracker: CostTracker::new(budget),
        }
    }

    /// Discover all available providers (Ollama, env-configured cloud, etc.)
    pub fn discover_all(&mut self) {
        // Always try local Ollama
        self.discover_ollama();

        // Cloud providers from environment
        self.discover_openai();
        self.discover_anthropic();
    }

    fn discover_ollama(&mut self) {
        let client = crate::ollama::OllamaClient::from_env();
        match client.list_models() {
            Ok(models) => {
                let model_configs: Vec<ModelConfig> = models
                    .iter()
                    .map(|m| ModelConfig {
                        model_id: m.clone(),
                        display_name: m.clone(),
                        capabilities: vec![
                            Capability::TextGeneration,
                            Capability::CodeGeneration,
                            Capability::Summarization,
                        ],
                        cost_per_million_tokens: 0.0, // local = free
                        context_window: 8192,
                        quality_score: 0.6,
                    })
                    .collect();

                if !model_configs.is_empty() {
                    self.providers.push(Provider {
                        id: "ollama".into(),
                        name: "Ollama (local)".into(),
                        trust_level: TrustLevel::Local,
                        models: model_configs,
                        base_url: format!(
                            "http://{}:{}",
                            std::env::var("QLANG_OLLAMA_HOST")
                                .unwrap_or_else(|_| "127.0.0.1".into()),
                            std::env::var("QLANG_OLLAMA_PORT").unwrap_or_else(|_| "11434".into()),
                        ),
                        api_key: None,
                        available: true,
                    });
                }
            }
            Err(_) => {
                eprintln!("[providers] Ollama not available");
            }
        }
    }

    fn discover_openai(&mut self) {
        if let Ok(key) = std::env::var("OPENAI_API_KEY") {
            self.providers.push(Provider {
                id: "openai".into(),
                name: "OpenAI".into(),
                trust_level: TrustLevel::Trusted,
                models: vec![
                    ModelConfig {
                        model_id: "gpt-4o".into(),
                        display_name: "GPT-4o".into(),
                        capabilities: vec![
                            Capability::TextGeneration,
                            Capability::CodeGeneration,
                            Capability::Summarization,
                        ],
                        cost_per_million_tokens: 5.0,
                        context_window: 128_000,
                        quality_score: 0.95,
                    },
                    ModelConfig {
                        model_id: "gpt-4o-mini".into(),
                        display_name: "GPT-4o Mini".into(),
                        capabilities: vec![
                            Capability::TextGeneration,
                            Capability::CodeGeneration,
                            Capability::Summarization,
                        ],
                        cost_per_million_tokens: 0.15,
                        context_window: 128_000,
                        quality_score: 0.80,
                    },
                ],
                base_url: "https://api.openai.com".into(),
                api_key: Some(key),
                available: true,
            });
        }
    }

    fn discover_anthropic(&mut self) {
        if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
            self.providers.push(Provider {
                id: "anthropic".into(),
                name: "Anthropic".into(),
                trust_level: TrustLevel::Trusted,
                models: vec![ModelConfig {
                    model_id: "claude-sonnet-4-20250514".into(),
                    display_name: "Claude Sonnet 4".into(),
                    capabilities: vec![
                        Capability::TextGeneration,
                        Capability::CodeGeneration,
                        Capability::Summarization,
                    ],
                    cost_per_million_tokens: 3.0,
                    context_window: 200_000,
                    quality_score: 0.95,
                }],
                base_url: "https://api.anthropic.com".into(),
                api_key: Some(key),
                available: true,
            });
        }
    }

    /// List all known models: (provider_id, model_id, display_name, cost)
    pub fn list_all_models(&self) -> Vec<(String, String, String, f64)> {
        let mut out = Vec::new();
        for p in &self.providers {
            for m in &p.models {
                out.push((
                    p.id.clone(),
                    m.model_id.clone(),
                    m.display_name.clone(),
                    m.cost_per_million_tokens,
                ));
            }
        }
        out
    }

    /// Select the best model matching requirements.
    pub fn select_model(
        &self,
        capability: &Capability,
        data_class: DataClass,
        pref: SelectionPreference,
    ) -> Option<(&Provider, &ModelConfig)> {
        let mut candidates: Vec<(&Provider, &ModelConfig)> = Vec::new();

        for provider in &self.providers {
            if !provider.available {
                continue;
            }
            // Data-class filter
            match data_class {
                DataClass::Confidential => {
                    if provider.trust_level != TrustLevel::Local {
                        continue;
                    }
                }
                DataClass::Internal => {
                    if provider.trust_level == TrustLevel::Untrusted {
                        continue;
                    }
                }
                DataClass::Public => {}
            }
            for model in &provider.models {
                if model.capabilities.contains(capability) {
                    candidates.push((provider, model));
                }
            }
        }

        if candidates.is_empty() {
            return None;
        }

        match pref {
            SelectionPreference::Local => candidates
                .iter()
                .find(|(p, _)| p.trust_level == TrustLevel::Local)
                .or(candidates.first())
                .copied(),
            SelectionPreference::Cheapest => {
                candidates.sort_by(|a, b| {
                    a.1.cost_per_million_tokens
                        .partial_cmp(&b.1.cost_per_million_tokens)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                candidates.first().copied()
            }
            SelectionPreference::Fastest => {
                // Prefer local (no network latency), then cheapest (smaller = faster)
                candidates.sort_by(|a, b| {
                    let a_local = if a.0.trust_level == TrustLevel::Local {
                        0
                    } else {
                        1
                    };
                    let b_local = if b.0.trust_level == TrustLevel::Local {
                        0
                    } else {
                        1
                    };
                    a_local.cmp(&b_local).then(
                        a.1.cost_per_million_tokens
                            .partial_cmp(&b.1.cost_per_million_tokens)
                            .unwrap_or(std::cmp::Ordering::Equal),
                    )
                });
                candidates.first().copied()
            }
            SelectionPreference::BestQuality => {
                candidates.sort_by(|a, b| {
                    b.1.quality_score
                        .partial_cmp(&a.1.quality_score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                candidates.first().copied()
            }
        }
    }

    /// Generate text using the best matching model.
    pub fn generate(
        &mut self,
        prompt: &str,
        capability: &Capability,
        data_class: DataClass,
        pref: SelectionPreference,
        _system: Option<&str>,
    ) -> Result<LlmResponse, String> {
        let (provider, model) = self
            .select_model(capability, data_class, pref)
            .ok_or_else(|| "No suitable model found".to_string())?;

        let provider_id = provider.id.clone();
        let model_id = model.model_id.clone();
        let cost_per_m = model.cost_per_million_tokens;

        let start = Instant::now();

        // Route to the right backend
        let text = match provider_id.as_str() {
            "ollama" => {
                let client = crate::ollama::OllamaClient::from_env();
                client
                    .generate(&model_id, prompt, None)
                    .map_err(|e| format!("Ollama error: {}", e))?
            }
            _ => {
                // For cloud providers, fall back to Ollama if available,
                // otherwise return a placeholder.
                // Full cloud HTTP clients will be added later.
                let client = crate::ollama::OllamaClient::from_env();
                match client.generate(&model_id, prompt, None) {
                    Ok(t) => t,
                    Err(_) => {
                        return Err(format!(
                            "Cloud provider {} not yet implemented for direct calls",
                            provider_id
                        ));
                    }
                }
            }
        };

        let latency_ms = start.elapsed().as_millis() as u64;

        // Rough token estimate (4 chars per token)
        let input_tokens = prompt.len() / 4;
        let output_tokens = text.len() / 4;
        let total_tokens = input_tokens + output_tokens;
        let cost = (total_tokens as f64 / 1_000_000.0) * cost_per_m;

        self.cost_tracker.record(&provider_id, cost);

        Ok(LlmResponse {
            provider_id,
            model_id,
            text,
            input_tokens,
            output_tokens,
            cost,
            latency_ms,
        })
    }

    /// Human-readable cost summary.
    pub fn cost_summary(&self) -> String {
        let ct = &self.cost_tracker;
        let mut s = format!(
            "Total: ${:.4} / ${:.2} ({} requests)",
            ct.total_spent, ct.budget_limit, ct.request_count
        );
        for (pid, cost) in &ct.by_provider {
            s.push_str(&format!("\n  {}: ${:.4}", pid, cost));
        }
        s
    }
}
