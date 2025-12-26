"""FastAPI application for SeamlessM4T translation service."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from loguru import logger

from .config import settings
from .models import HealthResponse, TaskType, TranslationRequest
from .translator_hf import translator


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    logger.info("Starting SeamlessM4T API...")
    translator.load()
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Seamless API",
    description="Translation service powered by SeamlessM4T",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check service health and model status."""
    return HealthResponse(
        status="healthy" if translator._loaded else "loading",
        model=settings.MODEL_NAME,
        device=str(translator.device) if translator.device else settings.DEVICE,
    )


@app.post("/modules/translation/process")
async def process_translation(request: TranslationRequest):
    """
    Process a translation request.

    This endpoint matches the legacy API format for backward compatibility.
    Returns plain text for text outputs, base64 for audio outputs.
    """
    try:
        data = request.data

        logger.info(
            f"Translation request: {data.task_string} "
            f"({data.source_language} -> {data.target_language})"
        )

        result = translator.process(
            input_data=data.input,
            task=data.task_string,
            source_lang=data.source_language,
            target_lang=data.target_language,
        )

        # Return format depends on task
        if data.task_string in (TaskType.TEXT2TEXT, TaskType.SPEECH2TEXT, TaskType.ASR):
            # Legacy format wraps text in CString
            return PlainTextResponse(f"[CString('{result}')]")
        else:
            # Audio tasks return raw base64
            return PlainTextResponse(result)

    except Exception as e:
        logger.error(f"Translation error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/translate")
async def translate(request: Request):
    """
    Simple translation endpoint for Rust proxy.
    Accepts: {"text": str, "source_lang": str, "target_lang": str}
    Returns: {"text": str}
    """
    try:
        body = await request.json()
        text = body.get("text", "")
        source_lang = body.get("source_lang", "eng")
        target_lang = body.get("target_lang", "spa")

        result = translator.process(
            input_data=text,
            task=TaskType.TEXT2TEXT,
            source_lang=source_lang,
            target_lang=target_lang,
        )

        return {"text": result}

    except Exception as e:
        logger.error(f"Translation error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/translate/advanced")
async def translate_advanced(request: TranslationRequest):
    """
    Advanced translation endpoint with full task support.
    """
    try:
        data = request.data

        result = translator.process(
            input_data=data.input,
            task=data.task_string,
            source_lang=data.source_language,
            target_lang=data.target_language,
        )

        return {
            "output": result,
            "source_language": data.source_language,
            "target_language": data.target_language,
            "task": data.task_string,
        }

    except Exception as e:
        logger.error(f"Translation error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/languages")
async def list_languages(
    task: TaskType | None = Query(default=None),
    direction: str | None = Query(default=None, pattern="^(source|target)$"),
):
    """List supported languages."""
    from .models import (
        LANGUAGE_MATRIX,
        LANGUAGE_NAME_TO_CODE,
        SUPPORTED_LANGUAGES,
        supported_languages_for_task,
    )

    codes = SUPPORTED_LANGUAGES
    if task is not None and direction is not None:
        codes = supported_languages_for_task(task, direction)

    def _json_safe_meta(meta: dict) -> dict:
        safe = dict(meta)
        if isinstance(safe.get("source_caps"), set):
            safe["source_caps"] = sorted(safe["source_caps"])
        if isinstance(safe.get("target_caps"), set):
            safe["target_caps"] = sorted(safe["target_caps"])
        return safe

    return {
        "codes": codes,
        "names": LANGUAGE_NAME_TO_CODE,
        "matrix": {code: _json_safe_meta(LANGUAGE_MATRIX[code]) for code in codes},
    }


def main():
    """Entry point for CLI."""
    import uvicorn

    uvicorn.run(
        "seamless_api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False,
    )


if __name__ == "__main__":
    main()
