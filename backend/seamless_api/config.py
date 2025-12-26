"""Configuration for SeamlessM4T API."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Application settings loaded from environment."""

    # Server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8020"))

    # Model (seamless_communication model names)
    # Options: seamlessM4T_v2_large (~9GB VRAM), seamlessM4T_medium (~5GB VRAM)
    MODEL_NAME: str = os.getenv("MODEL_NAME", "seamlessM4T_v2_large")
    # vocoder_v2 for v2_large, vocoder_36langs for medium
    VOCODER_NAME: str = os.getenv("VOCODER_NAME", "vocoder_v2")
    DEVICE: str = os.getenv("DEVICE", "cuda")  # cuda, cpu, mps
    TORCH_DTYPE: str = os.getenv("TORCH_DTYPE", "float16")  # float16, float32, bfloat16

    # Cache
    MODEL_CACHE_DIR: Path = Path(os.getenv("MODEL_CACHE_DIR", "/root/repos/sylliba-backend/models")).resolve()
    HF_HOME: str = os.getenv("HF_HOME", str(MODEL_CACHE_DIR))

    # Limits
    MAX_INPUT_LENGTH: int = int(os.getenv("MAX_INPUT_LENGTH", "1024"))
    MAX_AUDIO_DURATION_SEC: int = int(os.getenv("MAX_AUDIO_DURATION_SEC", "60"))


settings = Settings()
