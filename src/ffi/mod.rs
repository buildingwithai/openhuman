//! C-FFI surface for embedding Benito's agent loop into a Swift host (Benit0).
//!
//! # Architecture
//!
//! ```text
//!  Benit0 (Swift) ──calls──► extern "C" functions ──► Benito Rust core
//!       ▲                                                │
//!       │                                                │
//!       └────── callbacks (text_delta, tool_use, etc.) ──┘
//! ```
//!
//! # Usage from Swift
//!
//! 1. Build the Rust core as a `staticlib` (`cargo build --release --lib`).
//! 2. Link `libbenito_core.a` into the Xcode project.
//! 3. Import the generated bridging header (`BENITO_ffi.h`).
//! 4. Call `benito_agent_init()`, `benito_agent_set_provider()`,
//!    `benito_agent_register_tool()`, then `benito_agent_send_prompt()`.
//!
//! # Thread Safety
//!
//! All FFI functions are `Send + Sync` safe. The agent loop runs on a
//! Tokio runtime spawned inside `benito_agent_init()`. Callbacks are
//! invoked on the Tokio worker threads — Swift callers must dispatch
//! to the main actor if they touch UI.

use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::sync::Arc;
use tokio::runtime::Runtime;
use tokio::sync::mpsc;

use crate::benito::agent::harness::run_tool_call_loop;
use crate::benito::inference::provider::{
    ChatMessage, Provider, ProviderCapabilities,
};
use crate::benito::tools::traits::{Tool, ToolResult};

// ─── Opaque handle types ─────────────────────────────────────────────

/// Opaque handle to the agent runtime. Swift sees this as `UnsafeMutableRawPointer`.
pub struct BenitoAgentHandle {
    runtime: Runtime,
    provider: Arc<dyn Provider>,
    model: String,
    temperature: f64,
    tools: Vec<Box<dyn Tool>>,
    history: Vec<ChatMessage>,
    system_prompt: String,
    callbacks: BenitoCallbacks,
}

/// Callbacks from Rust → Swift. Each field is an `extern "C"` function pointer.
/// Note: Callbacks use `*mut c_void` for C compatibility. The `user_data` field
/// is wrapped in `UserDataPtr` internally for `Send + Sync`.
#[derive(Clone)]
pub struct BenitoCallbacks {
    pub on_text_delta: Option<extern "C" fn(*mut std::ffi::c_void, *const c_char)>,
    pub on_tool_use: Option<extern "C" fn(*mut std::ffi::c_void, *const c_char, *const c_char)>,
    pub on_tool_result: Option<extern "C" fn(*mut std::ffi::c_void, *const c_char)>,
    pub on_done: Option<extern "C" fn(*mut std::ffi::c_void, *const c_char)>,
    pub on_error: Option<extern "C" fn(*mut std::ffi::c_void, *const c_char)>,
    pub user_data: UserDataPtr,
}

unsafe impl Send for BenitoCallbacks {}
unsafe impl Sync for BenitoCallbacks {}

// ─── Agent lifecycle ─────────────────────────────────────────────────

/// Initialize the agent runtime. Returns an opaque handle.
///
/// # Safety
/// Caller must eventually call `benito_agent_free()` with the returned handle.
#[no_mangle]
pub unsafe extern "C" fn benito_agent_init() -> *mut BenitoAgentHandle {
    let runtime = match Runtime::new() {
        Ok(rt) => rt,
        Err(_) => return std::ptr::null_mut(),
    };

    let handle = Box::new(BenitoAgentHandle {
        runtime,
        provider: Arc::new(NullProvider),
        model: String::new(),
        temperature: 0.7,
        tools: Vec::new(),
        history: Vec::new(),
        system_prompt: String::new(),
        callbacks: BenitoCallbacks {
            on_text_delta: None,
            on_tool_use: None,
            on_tool_result: None,
            on_done: None,
            on_error: None,
            user_data: UserDataPtr(std::ptr::null_mut()),
        },
    });

    Box::into_raw(handle)
}

/// Free the agent handle.
///
/// # Safety
/// `handle` must be a valid pointer from `benito_agent_init()` and must not
/// be used after this call.
#[no_mangle]
pub unsafe extern "C" fn benito_agent_free(handle: *mut BenitoAgentHandle) {
    if !handle.is_null() {
        let _ = Box::from_raw(handle);
    }
}

/// Set the LLM provider and model.
///
/// `provider_name` must be one of: `"xai"`, `"anthropic"`, `"openai"`, `"ollama"`.
/// `api_key` is the API key for the provider.
/// `model` is the model identifier (e.g. `"grok-4.3"`).
///
/// # Safety
/// All string pointers must be valid, null-terminated UTF-8.
#[no_mangle]
pub unsafe extern "C" fn benito_agent_set_provider(
    handle: *mut BenitoAgentHandle,
    provider_name: *const c_char,
    api_key: *const c_char,
    model: *const c_char,
) -> i32 {
    if handle.is_null() || provider_name.is_null() || api_key.is_null() || model.is_null() {
        return -1;
    }

    let handle = &mut *handle;
    let provider_name = match CStr::from_ptr(provider_name).to_str() {
        Ok(s) => s.to_string(),
        Err(_) => return -1,
    };
    let api_key = match CStr::from_ptr(api_key).to_str() {
        Ok(s) => s.to_string(),
        Err(_) => return -1,
    };
    let model = match CStr::from_ptr(model).to_str() {
        Ok(s) => s.to_string(),
        Err(_) => return -1,
    };

    let provider = match create_provider(&provider_name, &api_key) {
        Ok(p) => p,
        Err(_) => return -1,
    };

    handle.provider = provider;
    handle.model = model;
    0
}

/// Set the system prompt.
///
/// # Safety
/// `prompt` must be a valid, null-terminated UTF-8 string.
#[no_mangle]
pub unsafe extern "C" fn benito_agent_set_system_prompt(
    handle: *mut BenitoAgentHandle,
    prompt: *const c_char,
) -> i32 {
    if handle.is_null() || prompt.is_null() {
        return -1;
    }

    let handle = &mut *handle;
    match CStr::from_ptr(prompt).to_str() {
        Ok(s) => {
            handle.system_prompt = s.to_string();
            0
        }
        Err(_) => -1,
    }
}

/// Set the temperature (0.0 - 2.0).
#[no_mangle]
pub unsafe extern "C" fn benito_agent_set_temperature(
    handle: *mut BenitoAgentHandle,
    temperature: f64,
) -> i32 {
    if handle.is_null() {
        return -1;
    }
    let handle = &mut *handle;
    handle.temperature = temperature.clamp(0.0, 2.0);
    0
}

// ─── Tool registration ───────────────────────────────────────────────

/// Register a tool with the agent.
///
/// `tool_name` — unique tool name (e.g. `"screenshot"`).
/// `description` — human-readable description for the LLM.
/// `parameters_schema_json` — JSON Schema string for the tool's parameters.
/// `execute_fn` — C function pointer called when the LLM wants to use this tool.
///   Signature: `fn(user_data, tool_name, args_json) -> *mut c_char` (result JSON).
///
/// # Safety
/// All string pointers must be valid, null-terminated UTF-8.
/// `execute_fn` must be thread-safe.
#[no_mangle]
pub unsafe extern "C" fn benito_agent_register_tool(
    handle: *mut BenitoAgentHandle,
    tool_name: *const c_char,
    description: *const c_char,
    parameters_schema_json: *const c_char,
    execute_fn: extern "C" fn(*mut std::ffi::c_void, *const c_char, *const c_char) -> *mut c_char,
) -> i32 {
    if handle.is_null() || tool_name.is_null() || description.is_null() || parameters_schema_json.is_null() {
        return -1;
    }

    let handle = &mut *handle;
    let name = match CStr::from_ptr(tool_name).to_str() {
        Ok(s) => s.to_string(),
        Err(_) => return -1,
    };
    let desc = match CStr::from_ptr(description).to_str() {
        Ok(s) => s.to_string(),
        Err(_) => return -1,
    };
    let schema_json = match CStr::from_ptr(parameters_schema_json).to_str() {
        Ok(s) => s.to_string(),
        Err(_) => return -1,
    };

    let schema: serde_json::Value = match serde_json::from_str(&schema_json) {
        Ok(v) => v,
        Err(_) => return -1,
    };

    let tool = FFITool {
        name,
        description: desc,
        parameters_schema: schema,
        execute_fn,
        user_data: UserDataPtr(handle.callbacks.user_data.0),
        on_tool_use: ToolUseCb(handle.callbacks.on_tool_use),
    };

    handle.tools.push(Box::new(tool));
    0
}

/// Set the user data pointer passed to tool execute callbacks.
#[no_mangle]
pub unsafe extern "C" fn benito_agent_set_user_data(
    handle: *mut BenitoAgentHandle,
    user_data: *mut std::ffi::c_void,
) {
    if !handle.is_null() {
        let handle = &mut *handle;
        handle.callbacks.user_data = UserDataPtr(user_data);
    }
}

// ─── Callbacks ───────────────────────────────────────────────────────

/// Set the text delta callback. Called for each chunk of assistant text.
///
/// `callback(user_data, text_chunk)` — `text_chunk` is null-terminated UTF-8.
#[no_mangle]
pub unsafe extern "C" fn benito_agent_set_text_delta_callback(
    handle: *mut BenitoAgentHandle,
    callback: Option<extern "C" fn(*mut std::ffi::c_void, *const c_char)>,
) {
    if !handle.is_null() {
        (*handle).callbacks.on_text_delta = callback;
    }
}

/// Set the tool use callback. Called when the LLM requests a tool.
///
/// `callback(user_data, tool_name, args_json)` — both strings are null-terminated UTF-8.
#[no_mangle]
pub unsafe extern "C" fn benito_agent_set_tool_use_callback(
    handle: *mut BenitoAgentHandle,
    callback: Option<extern "C" fn(*mut std::ffi::c_void, *const c_char, *const c_char)>,
) {
    if !handle.is_null() {
        (*handle).callbacks.on_tool_use = callback;
    }
}

/// Set the done callback. Called when the agent turn completes.
///
/// `callback(user_data, final_text)` — null-terminated UTF-8.
#[no_mangle]
pub unsafe extern "C" fn benito_agent_set_done_callback(
    handle: *mut BenitoAgentHandle,
    callback: Option<extern "C" fn(*mut std::ffi::c_void, *const c_char)>,
) {
    if !handle.is_null() {
        (*handle).callbacks.on_done = callback;
    }
}

/// Set the error callback. Called when an error occurs.
///
/// `callback(user_data, error_message)` — null-terminated UTF-8.
#[no_mangle]
pub unsafe extern "C" fn benito_agent_set_error_callback(
    handle: *mut BenitoAgentHandle,
    callback: Option<extern "C" fn(*mut std::ffi::c_void, *const c_char)>,
) {
    if !handle.is_null() {
        (*handle).callbacks.on_error = callback;
    }
}

// ─── Agent loop entry point ──────────────────────────────────────────

/// Send a prompt to the agent and start the agent loop.
///
/// This function returns immediately. Results are delivered via callbacks.
/// Returns 0 on success, -1 on failure.
///
/// # Safety
/// `prompt` must be a valid, null-terminated UTF-8 string.
#[no_mangle]
pub unsafe extern "C" fn benito_agent_send_prompt(
    handle: *mut BenitoAgentHandle,
    prompt: *const c_char,
) -> i32 {
    if handle.is_null() || prompt.is_null() {
        return -1;
    }

    let handle = &mut *handle;
    let prompt_str = match CStr::from_ptr(prompt).to_str() {
        Ok(s) => s.to_string(),
        Err(_) => return -1,
    };

    if handle.model.is_empty() {
        if let Some(cb) = handle.callbacks.on_error {
            let msg = CString::new("No provider configured. Call benito_agent_set_provider() first.").unwrap();
            cb(handle.callbacks.user_data.0, msg.as_ptr());
        }
        return -1;
    }

    let provider = Arc::clone(&handle.provider);
    let model = handle.model.clone();
    let temperature = handle.temperature;
    let system_prompt = handle.system_prompt.clone();
    // Move tools out of the handle into the async task
    let tools: Vec<Box<dyn Tool>> = handle.tools.drain(..).collect();
    let callbacks = handle.callbacks.clone();

    // Clone history for the async task
    let mut history = handle.history.clone();
    // Add user message to history
    history.push(ChatMessage::user(&prompt_str));

    // Spawn the agent loop on the Tokio runtime
    // Extract callbacks before moving into async
    let on_text_delta = TextDeltaCb(callbacks.on_text_delta);
    let on_tool_use = ToolUseCb(callbacks.on_tool_use);
    let on_done = callbacks.on_done;
    let on_error = callbacks.on_error;
    let user_data = UserDataPtr(callbacks.user_data.0);
    
    handle.runtime.spawn(async move {
        let result = run_agent_turn(
            provider.as_ref(),
            &mut history,
            &tools,
            &model,
            temperature,
            &system_prompt,
            on_text_delta,
            on_tool_use,
            user_data,
        ).await;

        match result {
            Ok(final_text) => {
                if let Some(cb) = on_done {
                    if let Ok(msg) = CString::new(final_text) {
                        cb(user_data.0, msg.as_ptr());
                    }
                }
            }
            Err(e) => {
                if let Some(cb) = on_error {
                    if let Ok(msg) = CString::new(format!("{}", e)) {
                        cb(user_data.0, msg.as_ptr());
                    }
                }
            }
        }
    });

    0
}

/// Block until the agent loop finishes the current turn and return the final text.
///
/// Returns a null-terminated UTF-8 string. Caller must free with `benito_free_string()`.
/// Returns NULL on error.
///
/// # Safety
/// `prompt` must be a valid, null-terminated UTF-8 string.
#[no_mangle]
pub unsafe extern "C" fn benito_agent_send_prompt_blocking(
    handle: *mut BenitoAgentHandle,
    prompt: *const c_char,
) -> *mut c_char {
    if handle.is_null() || prompt.is_null() {
        return std::ptr::null_mut();
    }

    let handle = &mut *handle;
    let prompt_str = match CStr::from_ptr(prompt).to_str() {
        Ok(s) => s.to_string(),
        Err(_) => return std::ptr::null_mut(),
    };

    if handle.model.is_empty() {
        return std::ptr::null_mut();
    }

    handle.history.push(ChatMessage::user(&prompt_str));

    let _tools: Vec<&dyn Tool> = handle.tools.iter().map(|t| t.as_ref()).collect();
    let on_text_delta = TextDeltaCb(handle.callbacks.on_text_delta);
    let on_tool_use = ToolUseCb(handle.callbacks.on_tool_use);
    let user_data = UserDataPtr(handle.callbacks.user_data.0);

    let result = handle.runtime.block_on(run_agent_turn(
        handle.provider.as_ref(),
        &mut handle.history,
        &handle.tools,
        &handle.model,
        handle.temperature,
        &handle.system_prompt,
        on_text_delta,
        on_tool_use,
        user_data,
    ));

    match result {
        Ok(text) => match CString::new(text) {
            Ok(s) => s.into_raw(),
            Err(_) => std::ptr::null_mut(),
        },
        Err(_) => std::ptr::null_mut(),
    }
}

/// Free a string returned by `benito_agent_send_prompt_blocking()`.
///
/// # Safety
/// `ptr` must be a valid pointer from `benito_agent_send_prompt_blocking()`.
#[no_mangle]
pub unsafe extern "C" fn benito_free_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        let _ = CString::from_raw(ptr);
    }
}

/// Clear the conversation history.
#[no_mangle]
pub unsafe extern "C" fn benito_agent_clear_history(handle: *mut BenitoAgentHandle) -> i32 {
    if handle.is_null() {
        return -1;
    }
    let handle = &mut *handle;
    // Keep system prompt, clear messages
    handle.history.clear();
    if !handle.system_prompt.is_empty() {
        handle.history.insert(0, ChatMessage::system(&handle.system_prompt));
    }
    0
}

// ─── Internal: agent turn execution ──────────────────────────────────

async fn run_agent_turn(
    provider: &dyn Provider,
    history: &mut Vec<ChatMessage>,
    tools: &[Box<dyn Tool>],
    model: &str,
    temperature: f64,
    system_prompt: &str,
    on_text_delta: TextDeltaCb,
    _on_tool_use: ToolUseCb,
    user_data: UserDataPtr,
) -> anyhow::Result<String> {
    // Ensure system prompt is first
    if !history.iter().any(|m| m.role == "system") && !system_prompt.is_empty() {
        history.insert(0, ChatMessage::system(system_prompt));
    }

    // Create a channel for streaming text deltas
    let (delta_tx, mut delta_rx) = mpsc::channel::<String>(128);

    // Run the tool call loop with our delta channel
    let result = run_tool_call_loop(
        provider,
        history,
        tools,
        "custom",
        model,
        temperature,
        false,  // not silent
        None,   // no approval manager
        "benito-ffi",
        &crate::benito::config::MultimodalConfig::default(),
        10,     // max iterations
        Some(delta_tx),  // our delta channel
        None,   // no tool name filter
        &[],    // no extra tools
        None,   // no progress sink
        None,   // no payload summarizer
    ).await;

    // Process any remaining deltas after the loop completes
    while let Ok(text) = delta_rx.try_recv() {
        if let Some(cb) = on_text_delta.0 {
            if let Ok(cstr) = CString::new(text) {
                cb(user_data.0, cstr.as_ptr());
                std::mem::forget(cstr);
            }
        }
    }

    result
}

// ─── Provider creation ───────────────────────────────────────────────

fn create_provider(name: &str, api_key: &str) -> anyhow::Result<Arc<dyn Provider>> {
    use crate::benito::inference::provider::compatible::{AuthStyle, OpenAiCompatibleProvider};

    match name {
        "xai" | "grok" => {
            // xAI Grok provider via OpenAI-compatible API
            let provider = OpenAiCompatibleProvider::new(
                "xAI",
                "https://api.x.ai/v1",
                Some(api_key),
                AuthStyle::Bearer,
            );
            Ok(Arc::new(provider))
        }
        "anthropic" | "claude" => {
            anyhow::bail!("Anthropic provider not yet implemented in FFI")
        }
        "openai" => {
            let provider = OpenAiCompatibleProvider::new(
                "OpenAI",
                "https://api.openai.com/v1",
                Some(api_key),
                AuthStyle::Bearer,
            );
            Ok(Arc::new(provider))
        }
        "ollama" => {
            let provider = OpenAiCompatibleProvider::new(
                "Ollama",
                "http://localhost:11434/v1",
                None,
                AuthStyle::Bearer,
            );
            Ok(Arc::new(provider))
        }
        _ => anyhow::bail!("Unknown provider: {}", name),
    }
}

// ─── FFI Tool wrapper ────────────────────────────────────────────────

/// A tool whose execution is delegated to a Swift function via C-FFI.
struct FFITool {
    name: String,
    description: String,
    parameters_schema: serde_json::Value,
    execute_fn: extern "C" fn(*mut std::ffi::c_void, *const c_char, *const c_char) -> *mut c_char,
    user_data: UserDataPtr,
    /// Optional callback fired BEFORE execute_fn for UI notification.
    on_tool_use: ToolUseCb,
}

// Explicitly mark FFITool as Send + Sync (all fields are Send + Sync)
unsafe impl Send for FFITool {}
unsafe impl Sync for FFITool {}

/// Wrapper for `*mut c_void` that is `Send + Sync` (unsafe, but required for FFI callbacks).
#[repr(transparent)]
struct UserDataPtr(*mut std::ffi::c_void);

unsafe impl Send for UserDataPtr {}
unsafe impl Sync for UserDataPtr {}

impl Clone for UserDataPtr {
    fn clone(&self) -> Self {
        Self(self.0)
    }
}

impl Copy for UserDataPtr {}

/// Wrapper for callback function pointers that makes them `Send + Sync`.
/// Function pointers containing `*mut c_void` in their signature are not `Send` by default.
#[repr(transparent)]
struct TextDeltaCb(Option<extern "C" fn(*mut std::ffi::c_void, *const c_char)>);

unsafe impl Send for TextDeltaCb {}
unsafe impl Sync for TextDeltaCb {}

impl Clone for TextDeltaCb {
    fn clone(&self) -> Self {
        Self(self.0)
    }
}

impl Copy for TextDeltaCb {}

#[repr(transparent)]
struct ToolUseCb(Option<extern "C" fn(*mut std::ffi::c_void, *const c_char, *const c_char)>);

unsafe impl Send for ToolUseCb {}
unsafe impl Sync for ToolUseCb {}

impl Clone for ToolUseCb {
    fn clone(&self) -> Self {
        Self(self.0)
    }
}

impl Copy for ToolUseCb {}

#[async_trait::async_trait]
impl Tool for FFITool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn parameters_schema(&self) -> serde_json::Value {
        self.parameters_schema.clone()
    }

    async fn execute(&self, args: serde_json::Value) -> anyhow::Result<ToolResult> {
        // Fire on_tool_use callback for UI notification before executing
        if let Some(cb) = self.on_tool_use.0 {
            let name_cstr = CString::new(self.name.as_str()).ok();
            let args_cstr = CString::new(serde_json::to_string(&args).unwrap_or_default()).ok();
            if let (Some(name), Some(args)) = (name_cstr, args_cstr) {
                cb(self.user_data.0, name.as_ptr(), args.as_ptr());
            }
        }

        let args_json = serde_json::to_string(&args)?;
        let name_cstr = CString::new(self.name.as_str())?;
        let args_cstr = CString::new(args_json.as_str())?;

        let result_ptr = (self.execute_fn)(self.user_data.0, name_cstr.as_ptr(), args_cstr.as_ptr());

        if result_ptr.is_null() {
            return Ok(ToolResult::error("Tool execution returned null"));
        }

        let result_cstr = unsafe { CStr::from_ptr(result_ptr) };
        let result_str = result_cstr.to_str().unwrap_or("Tool execution failed");

        // Parse the result — Swift should return JSON like {"content": "...", "is_error": false}
        let result: serde_json::Value = serde_json::from_str(result_str).unwrap_or_else(|_| {
            serde_json::json!({"content": result_str, "is_error": false})
        });

        // Free the string that Swift allocated
        unsafe {
            let _ = CString::from_raw(result_ptr);
        }

        let content = result.get("content")
            .and_then(|v| v.as_str())
            .unwrap_or(result_str)
            .to_string();

        let is_error = result.get("is_error")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        if is_error {
            Ok(ToolResult::error(&content))
        } else {
            Ok(ToolResult::success(&content))
        }
    }
}

// ─── Null provider (fallback) ────────────────────────────────────────

struct NullProvider;

#[async_trait::async_trait]
impl Provider for NullProvider {
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::default()
    }

    async fn chat_with_system(
        &self,
        _system_prompt: Option<&str>,
        _message: &str,
        _model: &str,
        _temperature: f64,
    ) -> anyhow::Result<String> {
        anyhow::bail!("No provider configured. Call benito_agent_set_provider() first.")
    }
}
