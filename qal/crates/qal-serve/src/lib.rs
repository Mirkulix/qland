//! QAL services (spec Phase 3 §2): a stored handler `fn(HttpRequest) -> HttpResponse` served over
//! HTTP/1.1. [`service`] is the request pipeline (also used by `qal_request` without TCP),
//! [`http`] reads and writes HTTP on a connection, [`server`] accepts connections and runs the
//! workers.

pub mod http;
pub mod server;
pub mod service;

pub use http::HttpLimits;
pub use server::{bind, serve, Log, ServerConfig};
pub use service::{Handled, LogEntry, Request, Response, Service, ServiceConfig, Worker};

use std::sync::atomic::AtomicBool;
use std::sync::Arc;

/// Sets `flag` on SIGINT and SIGTERM (the server then shuts down gracefully).
pub fn on_shutdown_signals(flag: &Arc<AtomicBool>) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        signal_hook::flag::register(signal_hook::consts::SIGINT, flag.clone())?;
        signal_hook::flag::register(signal_hook::consts::SIGTERM, flag.clone())?;
    }
    #[cfg(not(unix))]
    let _ = flag;
    Ok(())
}
