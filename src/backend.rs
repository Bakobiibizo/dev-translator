//! Backend subprocess management.
//!
//! Spawns and manages the Python backend as a child process.

use std::process::Stdio;
use tokio::process::{Child, Command};
use tracing::{error, info, warn};

/// Configuration for the backend subprocess.
#[derive(Clone, Debug)]
pub struct BackendConfig {
    /// Command to run (e.g., "python3").
    pub command: String,
    /// Arguments (e.g., ["-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7100"]).
    pub args: Vec<String>,
    /// Working directory for the backend.
    pub workdir: String,
    /// Port the backend listens on.
    pub port: u16,
    /// Health endpoint path.
    pub health_path: String,
}

impl BackendConfig {
    pub fn from_env(default_port: u16) -> Self {
        use std::env;
        let command = env::var("BACKEND_CMD").unwrap_or_else(|_| "python3".to_string());
        let args: Vec<String> = env::var("BACKEND_ARGS")
            .unwrap_or_else(|_| {
                format!(
                    "-m uvicorn api:app --host 0.0.0.0 --port {}",
                    default_port
                )
            })
            .split_whitespace()
            .map(String::from)
            .collect();
        let workdir = env::var("BACKEND_WORKDIR").unwrap_or_else(|_| "/app/backend".to_string());
        let port: u16 = env::var("BACKEND_PORT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default_port);
        let health_path = env::var("BACKEND_HEALTH_PATH").unwrap_or_else(|_| "/health".to_string());
        Self {
            command,
            args,
            workdir,
            port,
            health_path,
        }
    }
}

/// Spawn the backend subprocess.
pub async fn spawn_backend(cfg: &BackendConfig) -> Result<Child, String> {
    info!(
        command = %cfg.command,
        args = ?cfg.args,
        workdir = %cfg.workdir,
        "spawning backend subprocess"
    );

    let child = Command::new(&cfg.command)
        .args(&cfg.args)
        .current_dir(&cfg.workdir)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|e| format!("failed to spawn backend: {}", e))?;

    info!("backend subprocess started with pid {}", child.id().unwrap_or(0));
    Ok(child)
}

/// Health-check loop: periodically pings the backend health endpoint.
/// If the backend dies, attempts to restart it.
pub async fn health_check_loop(cfg: BackendConfig, client: reqwest::Client) {
    let health_url = format!("http://localhost:{}{}", cfg.port, cfg.health_path);
    let mut consecutive_failures = 0u32;
    let mut child: Option<Child> = None;

    loop {
        tokio::time::sleep(std::time::Duration::from_secs(10)).await;

        let healthy = match client.get(&health_url).send().await {
            Ok(resp) => resp.status().is_success(),
            Err(_) => false,
        };

        if healthy {
            consecutive_failures = 0;
        } else {
            consecutive_failures += 1;
            warn!(failures = consecutive_failures, "backend health check failed");

            if consecutive_failures >= 3 {
                warn!("restarting backend subprocess after repeated failures");
                // Kill existing child if any
                if let Some(ref mut c) = child {
                    let _ = c.kill().await;
                }
                match spawn_backend(&cfg).await {
                    Ok(c) => {
                        child = Some(c);
                        consecutive_failures = 0;
                    }
                    Err(e) => {
                        error!(error = %e, "failed to restart backend");
                    }
                }
            }
        }
    }
}

/// Ensure backend is running, spawn if needed.
pub async fn ensure_backend_running(cfg: &BackendConfig, client: &reqwest::Client) -> Result<Option<Child>, String> {
    let health_url = format!("http://localhost:{}{}", cfg.port, cfg.health_path);
    
    // Check if already running
    if let Ok(resp) = client.get(&health_url).send().await {
        if resp.status().is_success() {
            info!("backend already running on port {}", cfg.port);
            return Ok(None);
        }
    }

    // Spawn backend
    let child = spawn_backend(cfg).await?;
    
    // Wait a bit for it to start
    tokio::time::sleep(std::time::Duration::from_secs(2)).await;
    
    Ok(Some(child))
}
