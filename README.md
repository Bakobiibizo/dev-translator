# dev-translator

HuggingFace SeamlessM4T v2 translator (text↔text, speech↔text, text↔speech, speech↔speech). The Rust proxy exposes `POST /translate`.

## Requirements
- Docker + NVIDIA GPU (CUDA) for best performance (CPU works but slower)
- HuggingFace cache mounted to preserve the ~9GB model (`-v ~/.cache/huggingface:/models/huggingface`)
- Built and tested on **aarch64** with NGC PyTorch 25.09 (Blackwell-capable). Cross-arch: ensure base image matches host arch (e.g., x86_64 tag) and rebuild.

## Build
```bash
docker build -f Dockerfile.hf -t inference/translator:local .
```

## Run (standalone)
```bash
docker run --gpus all -d -p 7104:8104 \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v ~/.cache/huggingface:/models/huggingface \
  inference/translator:local
```

## Run with docker-compose (root of repo)
```bash
docker compose up translator
```

## Test
```bash
curl -X POST http://localhost:7104/translate \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello world","source_lang":"eng","target_lang":"spa"}'
# => {"text":"Hola mundo"}
```

Service port: external 7104 → internal 8104.***
