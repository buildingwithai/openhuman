//! Resolve OpenClaw / AI prompt directories for bundled and dev layouts.

use std::path::{Path, PathBuf};

/// OpenClaw markdown directory inside a bundled resource dir.
pub fn bundled_openclaw_prompts_dir(resource_dir: &Path) -> Option<PathBuf> {
    let candidates = [
        resource_dir.join("Benito").join("agent").join("prompts"),
        resource_dir.join("prompts"),
        resource_dir.join("ai"),
        resource_dir
            .join("src")
            .join("Benito")
            .join("agent")
            .join("prompts"),
    ];
    candidates.into_iter().find(|p| p.is_dir())
}

/// Locate `src/Benito/agent/prompts` by walking up from `cwd`.
pub fn repo_ai_prompts_dir(cwd: &Path) -> Option<PathBuf> {
    for up in 0..=8 {
        let mut base = cwd.to_path_buf();
        let mut ok = true;
        for _ in 0..up {
            if !base.pop() {
                ok = false;
                break;
            }
        }
        if !ok {
            continue;
        }
        let candidate = base
            .join("src")
            .join("Benito")
            .join("agent")
            .join("prompts");
        if candidate.is_dir() {
            return Some(candidate);
        }
    }
    None
}
