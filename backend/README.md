# Seamless API

FastAPI translation service powered by Meta's SeamlessM4T model. Supports speech-to-text, text-to-speech, speech-to-speech, and text-to-text translation across 100+ languages.

**[Hydra Dynamix](https://www.hydradynamix.com/)** | [GitHub](https://github.com/hydra-dynamix)

## Features

- **Multi-modal translation**: Speech ↔ Text in any combination
- **100+ languages**: Full SeamlessM4T language support
- **GPU accelerated**: CUDA support for fast inference
- **Docker ready**: Easy deployment with docker-compose
- **Legacy compatible**: Matches original `/modules/translation/process` API

## Quick Start

### Docker (Recommended)

```bash
# GPU version (requires NVIDIA GPU + nvidia-docker)
docker compose up seamless-api

# CPU version (slower, no GPU required)
docker compose --profile cpu up seamless-api-cpu
```

### Local Development

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Run the server
uv run seamless-api
```

## Configuration

Copy `.env.example` to `.env` and configure:

| Variable | Description | Default |
|----------|-------------|---------|
| `HOST` | Server host | `0.0.0.0` |
| `PORT` | Server port | `8000` |
| `MODEL_NAME` | HuggingFace model | `facebook/seamless-m4t-v2-large` |
| `DEVICE` | Compute device | `cuda` |
| `TORCH_DTYPE` | Model precision | `float16` |
| `HF_TOKEN` | HuggingFace token (optional) | - |

## API Endpoints

### Health Check

```bash
GET /health
```

### Translation (Legacy Format)

```bash
POST /modules/translation/process
Content-Type: application/json

{
  "data": {
    "input": "Hello, world!",
    "task_string": "text2text",
    "source_language": "English",
    "target_language": "Spanish"
  }
}
```

### Translation (Modern Format)

```bash
POST /translate
Content-Type: application/json

{
  "data": {
    "input": "Hello, world!",
    "task_string": "text2text",
    "source_language": "eng",
    "target_language": "spa"
  }
}
```

### Supported Tasks

| Task | Input | Output |
|------|-------|--------|
| `text2text` | Text | Text |
| `speech2text` | Base64 audio | Text |
| `text2speech` | Text | Base64 audio |
| `speech2speech` | Base64 audio | Base64 audio |
| `auto_speech_recognition` | Base64 audio | Text (same language) |

### List Languages

```bash
GET /languages
```

## Hardware Requirements

### GPU (Recommended)
- NVIDIA GPU with 8GB+ VRAM
- CUDA 12.1+
- nvidia-docker for containerized deployment

### CPU
- 16GB+ RAM
- Significantly slower inference (~10-30x)

## Model Information

This service uses [SeamlessM4T](https://github.com/facebookresearch/seamless_communication) from Meta AI.

Available models:
- `seamlessM4T_medium` - Smaller, faster
- `seamlessM4T_large` - Better quality
- `seamlessM4T_v2_large` - Latest version (default)

- **Languages**: 100+ for text, 35+ for speech
- **License**: CC-BY-NC 4.0

## License

MIT License - See LICENSE file
