//! OpenAI-compatible HTTP endpoint at `/v1/chat/completions` and `/v1/models`.
//!
//! ## Mounting
//!
//! The router is mounted by `src/core/jsonrpc.rs`:
//! ```ignore
//! .nest("/v1", crate::benito::inference::http::router())
//! ```
//! It inherits the same bearer-token auth middleware that guards `/rpc`.

pub mod server;
pub mod types;

pub use server::router;
