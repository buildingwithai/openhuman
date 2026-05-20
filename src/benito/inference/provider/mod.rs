//! Unified provider abstraction — cloud + local chat, embedding, and streaming.
//!
//! This module was previously `src/Benito/providers/`. It now lives under
//! `inference/provider/` so all inference concerns (local runtime, cloud
//! providers, HTTP endpoint) share a single domain root.

pub mod billing_error;
pub mod compatible;
pub mod compatible_dump;
pub mod compatible_parse;
pub mod compatible_stream;
pub mod compatible_types;
pub mod config_rejection;
pub mod factory;
pub mod benito_backend;
pub mod ops;
pub mod reliable;
pub mod router;
pub mod schemas;
pub mod temperature;
pub mod thread_context;
pub mod traits;

// Re-export benito_backend as BENITO_backend for backward compatibility
pub use benito_backend as BENITO_backend;

#[allow(unused_imports)]
pub use traits::{
    ChatMessage, ChatRequest, ChatResponse, ConversationMessage, Provider, ProviderCapabilities,
    ProviderCapabilityError, ProviderDelta, ToolCall, ToolResultMessage, UsageInfo,
};

pub use billing_error::is_budget_exhausted_message;
pub use config_rejection::is_provider_config_rejection_message;
pub use factory::{create_chat_provider, provider_for_role};
pub use ops::*;
