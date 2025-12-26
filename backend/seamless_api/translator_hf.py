"""SeamlessM4T translation engine using HuggingFace Transformers."""

import base64
import io
import tempfile
from pathlib import Path
from typing import Optional

import torch
import torchaudio
from loguru import logger
from pydub import AudioSegment
from transformers import (
    SeamlessM4Tv2ForTextToText,
    SeamlessM4Tv2ForSpeechToText, 
    SeamlessM4Tv2ForTextToSpeech,
    SeamlessM4Tv2ForSpeechToSpeech,
    AutoProcessor,
)

from .config import settings
from .models import TaskType, normalize_language

# Sample rate for SeamlessM4T
SAMPLE_RATE = 16000


class SeamlessTranslator:
    """Wrapper around HuggingFace SeamlessM4Tv2 models for translation tasks."""

    def __init__(self):
        self.t2t_model = None  # Text to Text
        self.s2t_model = None  # Speech to Text  
        self.t2s_model = None  # Text to Speech
        self.s2s_model = None  # Speech to Speech
        self.processor: Optional[AutoProcessor] = None
        self.device: Optional[torch.device] = None
        self.torch_dtype = None
        self._loaded = False

    def load(self) -> None:
        """Load the SeamlessM4Tv2 model from HuggingFace."""
        if self._loaded:
            return

        model_name = "facebook/seamless-m4t-v2-large"
        logger.info(f"Loading HuggingFace model: {model_name}")
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

        self.torch_dtype = torch_dtype
        self.model_name = model_name

        # Load processor (handles tokenization and audio processing)
        logger.info("Loading processor...")
        self.processor = AutoProcessor.from_pretrained(model_name)

        # Load text-to-text model (most common use case, load eagerly)
        logger.info("Loading text-to-text model...")
        self.t2t_model = SeamlessM4Tv2ForTextToText.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
        ).to(self.device)
        self.t2t_model.eval()

        self._loaded = True
        logger.info(f"Model loaded on {self.device}")

    def _get_s2t_model(self):
        """Lazy load speech-to-text model."""
        if self.s2t_model is None:
            logger.info("Loading speech-to-text model...")
            self.s2t_model = SeamlessM4Tv2ForSpeechToText.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
            ).to(self.device)
            self.s2t_model.eval()
        return self.s2t_model

    def _get_t2s_model(self):
        """Lazy load text-to-speech model."""
        if self.t2s_model is None:
            logger.info("Loading text-to-speech model...")
            self.t2s_model = SeamlessM4Tv2ForTextToSpeech.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
            ).to(self.device)
            self.t2s_model.eval()
        return self.t2s_model

    def _get_s2s_model(self):
        """Lazy load speech-to-speech model."""
        if self.s2s_model is None:
            logger.info("Loading speech-to-speech model...")
            self.s2s_model = SeamlessM4Tv2ForSpeechToSpeech.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
            ).to(self.device)
            self.s2s_model.eval()
        return self.s2s_model

    def _hf_lang_code(self, lang: str) -> str:
        """Convert our language codes to HuggingFace format."""
        # HuggingFace uses 3-letter codes like 'eng', 'spa', 'fra'
        # Our normalize_language should already return these
        code = normalize_language(lang)
        return code

    def translate_text(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> str:
        """Translate text to text (T2TT)."""
        src_code = self._hf_lang_code(source_lang)
        tgt_code = self._hf_lang_code(target_lang)

        logger.debug(f"T2TT: {src_code} -> {tgt_code}: {text[:50]}...")

        # Process text input
        text_inputs = self.processor(
            text=text,
            src_lang=src_code,
            return_tensors="pt",
        ).to(self.device)

        # Generate translation
        with torch.no_grad():
            output_tokens = self.t2t_model.generate(
                **text_inputs,
                tgt_lang=tgt_code,
            )

        # Decode output
        result = self.processor.decode(
            output_tokens[0],
            skip_special_tokens=True
        )

        logger.debug(f"Result: {result[:50]}...")
        return result

    def translate_speech_to_text(
        self,
        audio_base64: str,
        source_lang: str,
        target_lang: str,
    ) -> str:
        """Translate speech to text (S2TT)."""
        src_code = self._hf_lang_code(source_lang)
        tgt_code = self._hf_lang_code(target_lang)

        logger.debug(f"S2TT: {src_code} -> {tgt_code}")

        # Decode audio
        waveform, sample_rate = self._decode_audio(audio_base64)

        # Process audio input
        audio_inputs = self.processor(
            audios=waveform,
            sampling_rate=sample_rate,
            return_tensors="pt"
        ).to(self.device)

        # Generate translation
        model = self._get_s2t_model()
        with torch.no_grad():
            output_tokens = model.generate(
                **audio_inputs,
                tgt_lang=tgt_code,
            )

        # Decode output
        result = self.processor.decode(
            output_tokens[0],
            skip_special_tokens=True
        )

        return result

    def translate_text_to_speech(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> str:
        """Translate text to speech (T2ST), return base64 audio."""
        src_code = self._hf_lang_code(source_lang)
        tgt_code = self._hf_lang_code(target_lang)

        logger.debug(f"T2ST: {src_code} -> {tgt_code}: {text[:50]}...")

        # Process text input
        text_inputs = self.processor(
            text=text,
            src_lang=src_code,
            return_tensors="pt"
        ).to(self.device)

        # Generate translation with speech
        model = self._get_t2s_model()
        with torch.no_grad():
            output = model.generate(
                **text_inputs,
                tgt_lang=tgt_code,
            )

        # Extract audio waveform - output is (text_tokens, audio_waveform)
        if isinstance(output, tuple) and len(output) >= 2:
            waveform = output[1].squeeze().cpu()
        else:
            waveform = output.cpu()

        return self._encode_audio(waveform, SAMPLE_RATE)

    def translate_speech_to_speech(
        self,
        audio_base64: str,
        source_lang: str,
        target_lang: str,
    ) -> str:
        """Translate speech to speech (S2ST), return base64 audio."""
        src_code = self._hf_lang_code(source_lang)
        tgt_code = self._hf_lang_code(target_lang)

        logger.debug(f"S2ST: {src_code} -> {tgt_code}")

        # Decode audio
        waveform, sample_rate = self._decode_audio(audio_base64)

        # Process audio input
        audio_inputs = self.processor(
            audios=waveform,
            sampling_rate=sample_rate,
            return_tensors="pt"
        ).to(self.device)

        # Generate translation with speech
        model = self._get_s2s_model()
        with torch.no_grad():
            output = model.generate(
                **audio_inputs,
                tgt_lang=tgt_code,
            )

        # Extract audio waveform - output is (text_tokens, audio_waveform)
        if isinstance(output, tuple) and len(output) >= 2:
            waveform = output[1].squeeze().cpu()
        else:
            waveform = output.cpu()

        return self._encode_audio(waveform, SAMPLE_RATE)

    def transcribe(
        self,
        audio_base64: str,
        source_lang: str,
    ) -> str:
        """Automatic speech recognition (ASR) - transcribe audio to text."""
        src_code = self._hf_lang_code(source_lang)

        logger.debug(f"ASR: {src_code}")

        # Decode audio
        waveform, sample_rate = self._decode_audio(audio_base64)

        # Process audio input
        audio_inputs = self.processor(
            audios=waveform,
            sampling_rate=sample_rate,
            return_tensors="pt"
        ).to(self.device)

        # Generate transcription (same language)
        model = self._get_s2t_model()
        with torch.no_grad():
            output_tokens = model.generate(
                **audio_inputs,
                tgt_lang=src_code,
            )

        # Decode output
        result = self.processor.decode(
            output_tokens[0],
            skip_special_tokens=True
        )

        return result

    def _decode_audio(self, audio_base64: str) -> tuple:
        """Decode base64 audio to waveform tensor."""
        # Handle padding
        audio_bytes = base64.b64decode(audio_base64 + "==")

        # Use pydub to handle various formats and convert to WAV
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        audio = audio.set_channels(1)  # Mono
        audio = audio.set_frame_rate(SAMPLE_RATE)  # 16kHz for SeamlessM4T

        # Export to WAV bytes
        buffer = io.BytesIO()
        audio.export(buffer, format="wav")
        buffer.seek(0)

        # Load with torchaudio
        waveform, sample_rate = torchaudio.load(buffer)

        # Squeeze to 1D if needed
        if waveform.dim() > 1:
            waveform = waveform.squeeze(0)

        return waveform.numpy(), sample_rate

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
