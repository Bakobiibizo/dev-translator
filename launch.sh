#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[launch] starting dev-translator container..."
cd "$ROOT_DIR"
docker compose up -d dev-translator
