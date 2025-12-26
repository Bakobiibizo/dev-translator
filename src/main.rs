use std::net::SocketAddr;
use std::sync::Arc;

use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use reqwest::Client;
use tokio::net::TcpListener;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tower_http::cors::CorsLayer;
use tracing::{error, info};

mod backend;
mod config;

#[derive(Clone)]
struct AppState {
    config: Arc<config::Config>,
    client: Client,
    ready: Arc<RwLock<bool>>,
}

#[derive(Deserialize)]
struct TranslateRequest {
    text: String,
    #[serde(default)]
    source_lang: Option<String>,
    #[serde(default)]
    target_lang: Option<String>,
}

#[derive(Serialize)]
struct ReadyStatus {
    ready: bool,
}

async fn health() -> &'static str {
    "ok"
}

async fn ready(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let ready = *state.ready.read().await;
    (StatusCode::OK, Json(ReadyStatus { ready }))
}

async fn warm(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    match call_backend_health(&state).await {
        Ok(_) => {
            *state.ready.write().await = true;
            (StatusCode::OK, "warmed")
        }
        Err(err) => {
            error!("warm failed: {}", err);
            (StatusCode::BAD_GATEWAY, "warm failed")
        }
    }
}

async fn translate(
    State(state): State<Arc<AppState>>,
    Json(req): Json<TranslateRequest>,
) -> impl IntoResponse {
    let url = format!("{}/translate", state.config.backend_url);
    let payload = serde_json::json!({
        "text": req.text,
        "source_lang": req.source_lang,
        "target_lang": req.target_lang,
    });
    match state.client.post(&url).json(&payload).send().await {
        Ok(resp) => {
            let status = resp.status();
            match resp.text().await {
                Ok(body) => (status, body),
                Err(err) => {
                    error!("backend read error: {}", err);
                    (StatusCode::BAD_GATEWAY, "backend read error".into())
                }
            }
        }
        Err(err) => {
            error!("backend error: {}", err);
            (StatusCode::BAD_GATEWAY, "backend error".into())
        }
    }
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();
    dotenv::dotenv().ok();

    let cfg = Arc::new(config::Config::load());
    let client = Client::new();

    // Ensure backend subprocess is running
    if let Err(e) = backend::ensure_backend_running(&cfg.backend, &client).await {
        error!("failed to start backend: {}", e);
    }

    let state = Arc::new(AppState {
        config: cfg.clone(),
        client: client.clone(),
        ready: Arc::new(RwLock::new(false)),
    });

    // Start health-check loop for backend subprocess
    let health_cfg = cfg.backend.clone();
    let health_client = client.clone();
    tokio::spawn(async move {
        backend::health_check_loop(health_cfg, health_client).await;
    });

    let preload_state = state.clone();
    tokio::spawn(async move {
        if preload_state.config.preload {
            // Wait a bit for container to be ready
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
            if call_backend_health(&preload_state).await.is_ok() {
                *preload_state.ready.write().await = true;
                info!("backend warmed");
            }
        } else {
            *preload_state.ready.write().await = true;
        }
    });

    let app = Router::new()
        .route("/health", get(health))
        .route("/ready", get(ready))
        .route("/warm", post(warm))
        .route("/translate", post(translate))
        .layer(CorsLayer::permissive())
        .with_state(state.clone());

    let addr: SocketAddr = format!("{}:{}", state.config.api_host, state.config.api_port)
        .parse()
        .unwrap_or_else(|e| {
            error!("Invalid bind address: {}", e);
            SocketAddr::from(([0, 0, 0, 0], 7104))
        });
    info!("listening on {}, proxying to {}", addr, state.config.backend_url);
    let listener = TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn call_backend_health(state: &AppState) -> Result<(), String> {
    let url = format!("{}/health", state.config.backend_url);
    match state.client.get(url).send().await {
        Ok(resp) if resp.status().is_success() => Ok(()),
        Ok(_) => Err("backend unhealthy".to_string()),
        Err(e) => Err(e.to_string()),
    }
}
