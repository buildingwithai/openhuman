//! Accessibility focus, clipboard/paste insertion, and key state probes.
//!
//! Delegates to the shared `accessibility` middleware module.

pub(super) use crate::benito::accessibility::any_modifier_down;
pub(super) use crate::benito::accessibility::apply_text_to_focused_field;
pub(super) use crate::benito::accessibility::focused_text_context_verbose;
pub(super) use crate::benito::accessibility::is_escape_key_down;
pub(super) use crate::benito::accessibility::is_tab_key_down;
pub(super) use crate::benito::accessibility::send_backspace;
#[cfg(target_os = "macos")]
pub(super) use crate::benito::accessibility::validate_focused_target;
