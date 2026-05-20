//! Stub whisper_engine module - whisper-rs disabled (requires libclang)
//! 
//! This stub allowed compilation without whisper-rs. Full whisper support
//! requires libclang to be installed for the whisper-rs-sys build.

use parking_lot::Mutex;
use std::path::PathBuf;
use std::sync::Arc;

/// Result of a transcription call.
#[derive(Debug, Clone)]
pub struct TranscriptionResult {
    pub text: String,
    pub avg_logprob: Option<f32>,
    pub segments_accepted: usize,
    pub segments_total: usize,
}

/// Thread-safe handle to an optionally-loaded whisper engine.
pub type WhisperEngineHandle = Arc<Mutex<Option<StubEngine>>>;

pub struct StubEngine;

/// Create a new empty engine handle.
pub fn new_handle() -> WhisperEngineHandle {
    Arc::new(Mutex::new(None))
}

/// Check if the engine is loaded (always false for stub).
pub fn is_loaded(_handle: &WhisperEngineHandle) -> bool {
    false
}

/// Load the engine (stub - does nothing).
pub fn load_engine(
    _handle: &WhisperEngineHandle,
    _model_path: &PathBuf,
    _use_gpu: bool,
    _gpu_desc: Option<&str>,
) -> anyhow::Result<()> {
    anyhow::bail!("whisper-rs disabled: requires libclang for whisper-rs-sys build")
}

/// Transcribe PCM audio (stub - returns empty).
pub fn transcribe_pcm_i16(
    _handle: &WhisperEngineHandle,
    _samples: &[i16],
    _language: Option<&str>,
    _prompt: Option<&str>,
) -> anyhow::Result<TranscriptionResult> {
    anyhow::bail!("whisper-rs disabled: requires libclang for whisper-rs-sys build")
}

/// Load engine from model path (stub).
pub fn load_engine_from_path(_model_path: &std::path::Path) -> anyhow::Result<StubEngine> {
    anyhow::bail!("whisper-rs disabled: requires libclang for whisper-rs-sys build")
}

/// Transcribe a WAV file (stub).
pub fn transcribe_wav_file(
    _handle: &WhisperEngineHandle,
    _wav_path: &std::path::Path,
    _language: Option<&str>,
    _prompt: Option<&str>,
) -> Result<TranscriptionResult, String> {
    Err("whisper-rs disabled: requires libclang for whisper-rs-sys build".to_string())
}
