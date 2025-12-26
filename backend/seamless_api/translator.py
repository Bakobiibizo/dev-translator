"""SeamlessM4T translation engine using seamless_communication."""

import base64
import io
import tempfile
from pathlib import Path

import torch
import torchaudio
from loguru import logger
from pydub import AudioSegment
from seamless_communication.inference import Translator

from .config import settings
from .models import TaskType, normalize_language

# Sample rate for SeamlessM4T
SAMPLE_RATE = 16000


class SeamlessTranslator:
    """Wrapper around SeamlessM4T model for translation tasks."""

    def __init__(self):
        self.translator: Translator | None = None
        self.device: torch.device | None = None
        self._loaded = False

    def load(self) -> None:
        """Load the SeamlessM4T model."""
        if self._loaded:
            return

        logger.info(f"Loading model: {settings.MODEL_NAME}")
        logger.info(f"Device: {settings.DEVICE}, dtype: {settings.TORCH_DTYPE}")

        # Set device
        if settings.DEVICE == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif settings.DEVICE == "mps" and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
            if settings.DEVICE != "cpu":
                logger.warning(f"Requested device {settings.DEVICE} not available, using CPU")

        # Set dtype
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(settings.TORCH_DTYPE, torch.float16)

        # Don't use float16 on CPU
        if self.device.type == "cpu":
            torch_dtype = torch.float32
            logger.info("Using float32 on CPU")

        # Load the Translator from seamless_communication
        self.translator = Translator(
            model_name_or_card=settings.MODEL_NAME,
            vocoder_name_or_card=settings.VOCODER_NAME,
            device=self.device,
            dtype=torch_dtype,
        )

        self._loaded = True
        logger.info(f"Model loaded on {self.device}")

    def translate_text(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> str:
        """Translate text to text (T2TT)."""
        src_code = normalize_language(source_lang)
        tgt_code = normalize_language(target_lang)

        logger.debug(f"T2TT: {src_code} -> {tgt_code}: {text[:50]}...")

        translated_text, _ = self.translator.predict(
            input=text,
            task_str="T2TT",
            src_lang=src_code,
            tgt_lang=tgt_code,
        )

        result = str(translated_text)
        logger.debug(f"Result: {result[:50]}...")
        return result

    def translate_speech_to_text(
        self,
        audio_base64: str,
        source_lang: str,
        target_lang: str,
    ) -> str:
        """Translate speech to text (S2TT)."""
        src_code = normalize_language(source_lang)
        tgt_code = normalize_language(target_lang)

        logger.debug(f"S2TT: {src_code} -> {tgt_code}")

        # Decode and save audio to temp file
        audio_path = self._decode_audio_to_file(audio_base64)

        try:
            translated_text, _ = self.translator.predict(
                input=str(audio_path),
                task_str="S2TT",
                src_lang=src_code,
                tgt_lang=tgt_code,
            )
            return str(translated_text)
        finally:
            audio_path.unlink(missing_ok=True)

    def translate_text_to_speech(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> str:
        """Translate text to speech (T2ST), return base64 audio."""
        src_code = normalize_language(source_lang)
        tgt_code = normalize_language(target_lang)

        logger.debug(f"T2ST: {src_code} -> {tgt_code}: {text[:50]}...")

        _, audio_output = self.translator.predict(
            input=text,
            task_str="T2ST",
            src_lang=src_code,
            tgt_lang=tgt_code,
        )

        if audio_output is None:
            raise RuntimeError("No audio output generated")

        # audio_output is a tuple of (waveform, sample_rate)
        waveform = audio_output.audio_wavs[0][0]
        sample_rate = audio_output.sample_rate

        return self._encode_audio(waveform, sample_rate)

    def translate_speech_to_speech(
        self,
        audio_base64: str,
        source_lang: str,
        target_lang: str,
    ) -> str:
        """Translate speech to speech (S2ST), return base64 audio."""
        src_code = normalize_language(source_lang)
        tgt_code = normalize_language(target_lang)

        logger.debug(f"S2ST: {src_code} -> {tgt_code}")

        # Decode and save audio to temp file
        audio_path = self._decode_audio_to_file(audio_base64)

        try:
            _, audio_output = self.translator.predict(
                input=str(audio_path),
                task_str="S2ST",
                src_lang=src_code,
                tgt_lang=tgt_code,
            )

            if audio_output is None:
                raise RuntimeError("No audio output generated")

            waveform = audio_output.audio_wavs[0][0]
            sample_rate = audio_output.sample_rate

            return self._encode_audio(waveform, sample_rate)
        finally:
            audio_path.unlink(missing_ok=True)

    def transcribe(
        self,
        audio_base64: str,
        source_lang: str,
    ) -> str:
        """Automatic speech recognition (ASR) - transcribe audio to text."""
        src_code = normalize_language(source_lang)

        logger.debug(f"ASR: {src_code}")

        # Decode and save audio to temp file
        audio_path = self._decode_audio_to_file(audio_base64)

        try:
            transcribed_text, _ = self.translator.predict(
                input=str(audio_path),
                task_str="ASR",
                src_lang=src_code,
                tgt_lang=src_code,
            )
            return str(transcribed_text)
        finally:
            audio_path.unlink(missing_ok=True)

    def _decode_audio_to_file(self, audio_base64: str) -> Path:
        """Decode base64 audio and save to temp file."""
        # Handle padding
        audio_bytes = base64.b64decode(audio_base64 + "==")

        # Create temp file
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_path = Path(temp_file.name)
        temp_file.close()

        # Use pydub to handle various formats and convert to WAV
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        audio = audio.set_channels(1)  # Mono
        audio = audio.set_frame_rate(SAMPLE_RATE)  # 16kHz for SeamlessM4T
        audio.export(temp_path, format="wav")

        return temp_path

    def _encode_audio(self, waveform: torch.Tensor, sample_rate: int) -> str:
        """Encode audio tensor to base64 WAV."""
        # Ensure CPU tensor
        if waveform.is_cuda:
            waveform = waveform.cpu()

        # Ensure 2D (channels, samples)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Ensure float32 for torchaudio
        waveform = waveform.float()

        # Save to buffer
        buffer = io.BytesIO()
        torchaudio.save(buffer, waveform, sample_rate, format="wav")
        buffer.seek(0)

        return base64.b64encode(buffer.read()).decode("utf-8")

    def process(
        self,
        input_data: str,
        task: TaskType,
        source_lang: str,
        target_lang: str,
    ) -> str:
        """Process a translation request based on task type."""
        if task == TaskType.TEXT2TEXT:
            return self.translate_text(input_data, source_lang, target_lang)
        elif task == TaskType.SPEECH2TEXT:
            return self.translate_speech_to_text(input_data, source_lang, target_lang)
        elif task == TaskType.TEXT2SPEECH:
            return self.translate_text_to_speech(input_data, source_lang, target_lang)
        elif task == TaskType.SPEECH2SPEECH:
            return self.translate_speech_to_speech(input_data, source_lang, target_lang)
        elif task == TaskType.ASR:
            return self.transcribe(input_data, source_lang)
        else:
            raise ValueError(f"Unknown task type: {task}")


# Global instance
translator = SeamlessTranslator()
