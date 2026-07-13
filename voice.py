"""Voice I/O: Speech-to-Text via faster-whisper, Text-to-Speech via piper-tts.

Both models are loaded lazily and cached at the Streamlit resource level so
they survive reruns and are shared across user interactions.
"""
from __future__ import annotations

import io
import os
import wave
import logging
from typing import Optional
import numpy as np
import streamlit as st

from i18n import piper_voice_for, whisper_lang_for

log = logging.getLogger(__name__)

# Where piper stores its downloaded voice files. Created on first use.
PIPER_VOICE_DIR = os.path.join("models", "piper")
os.makedirs(PIPER_VOICE_DIR, exist_ok=True)

# Whisper model size for STT. "small" is a good accuracy/speed tradeoff for RTX 4060.
WHISPER_MODEL_SIZE = "small"


@st.cache_resource(show_spinner="Loading Whisper (STT)...")
def _load_whisper():
    """Lazy-load faster-whisper once and cache it."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise RuntimeError(
            "faster-whisper is not installed. Run `pip install faster-whisper`."
        ) from e

    import torch

    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else None

    # NOTE: faster-whisper's actual inference backend is ctranslate2, which
    # does its own CUDA/cuDNN detection independent of torch — so even when
    # torch reports CUDA is available, ctranslate2 can still fail to init on
    # GPU if the matching cuBLAS/cuDNN DLLs aren't on PATH (common on Windows
    # if you installed torch/faster-whisper but not the NVIDIA runtime libs).
    # We try GPU first and fall back to CPU instead of trusting torch alone,
    # and — importantly — we print the outcome with `print()` rather than
    # `log.info()`, since Python's logging defaults to WARNING level and an
    # INFO call here would silently produce no console output at all.
    if cuda_available:
        try:
            print(f"🎮 CUDA detected ({gpu_name}). Loading Whisper {WHISPER_MODEL_SIZE} on GPU (float16)...")
            model = WhisperModel(WHISPER_MODEL_SIZE, device="cuda", compute_type="float16")
            print("✅ Whisper loaded on GPU.")
            return model
        except Exception as e:
            print(f"⚠️ GPU load failed ({e}). Falling back to CPU.")
            print(
                "   This usually means the NVIDIA cuBLAS/cuDNN runtime libraries "
                "aren't installed/discoverable, even though torch sees the GPU. "
                "See https://github.com/SYSTRAN/faster-whisper#gpu for the required libraries."
            )

    print(f"🖥️ Loading Whisper {WHISPER_MODEL_SIZE} on CPU (int8)...")
    model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    print("✅ Whisper loaded on CPU.")
    return model


def _voice_id_to_hf_path(voice_id: str) -> str:
    """
    Convert a piper voice id (e.g. 'en_US-amy-low') to the relative path on
    the rhasspy/piper-voices HuggingFace repo (e.g. 'en/en_US/amy/low').

    Format on HF:  <lang>/<lang_REGION>/<name>/<quality>
    Piper id:      <lang>_<REGION>-<name>-<quality>   (quality is 'low'/'medium'/'high')
    """
    # Split off the quality suffix
    if "-" in voice_id:
        head, quality = voice_id.rsplit("-", 1)
    else:
        head, quality = voice_id, "medium"

    # head is '<lang>_<REGION>-<name>' — split off the name
    if "-" in head:
        lang_region, name = head.split("-", 1)
    else:
        lang_region, name = head, "default"

    # lang_region is '<lang>_<REGION>' — split off the region
    if "_" in lang_region:
        lang, region = lang_region.split("_", 1)
    else:
        lang, region = lang_region, ""

    parts = [lang]
    if region:
        parts.append(f"{lang}_{region}")
    parts.append(name)
    parts.append(quality)
    return "/".join(parts)


def _ensure_piper_voice(voice_id: str) -> tuple:
    """
    Make sure the .onnx + .onnx.json for `voice_id` exist locally; download
    them from the rhasspy/piper-voices HuggingFace repo if not. Returns
    (onnx_path, config_path). piper-tts 1.4 removed the bundled
    `piper.download.ensure_voice` helper, so we do it ourselves.
    """
    import urllib.request

    onnx_path = os.path.join(PIPER_VOICE_DIR, f"{voice_id}.onnx")
    config_path = os.path.join(PIPER_VOICE_DIR, f"{voice_id}.onnx.json")

    if os.path.exists(onnx_path) and os.path.exists(config_path):
        return onnx_path, config_path

    hf_path = _voice_id_to_hf_path(voice_id)
    base = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/{hf_path}/{voice_id}"
    for filename, path in (("onnx", onnx_path), ("onnx.json", config_path)):
        if os.path.exists(path):
            continue
        url = f"{base}.{filename}"
        print(f"⬇️ Downloading {url} ...")
        urllib.request.urlretrieve(url, path)

    return onnx_path, config_path


@st.cache_resource(show_spinner="Loading Piper (TTS)...")
def _load_piper(voice_id: str):
    """Lazy-load piper for a given voice. Cached per voice_id."""
    try:
        from piper import PiperVoice
    except ImportError as e:
        raise RuntimeError(
            "piper-tts is not installed. Run `pip install piper-tts`."
        ) from e

    onnx_path, config_path = _ensure_piper_voice(voice_id)
    return PiperVoice.load(onnx_path, config_path=config_path)


class VoiceIO:
    """High-level façade for STT and TTS. Falls back gracefully if a backend is missing."""

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000, language_name: Optional[str] = None) -> str:
        """
        Transcribe a mono float32 numpy array (range [-1, 1]) to text.
        `sample_rate` is the rate the array is recorded at (webrtc gives 48000).
        `language_name` is the display language selected in the UI (e.g.
        "Hindi", "Spanish") — it's translated to a Whisper language code so
        speech in that language is actually transcribed correctly instead of
        being forced through English. Pass None to let Whisper auto-detect.
        """
        if audio is None or len(audio) == 0:
            return ""

        # faster-whisper accepts numpy float32 directly
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        duration = len(audio) / float(sample_rate)
        whisper_lang = whisper_lang_for(language_name) if language_name else None
        print(f"🎤 Transcribing {duration:.2f}s of audio (language={whisper_lang or 'auto'})...")

        model = _load_whisper()
        segments, _info = model.transcribe(
            audio,
            language=whisper_lang,  # None lets Whisper auto-detect
            beam_size=5,
            vad_filter=True,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()

        if not text:
            # vad_filter can misjudge a short/quiet clip as pure silence and
            # strip the whole thing before whisper ever sees it, which
            # silently yields an empty string — indistinguishable from a
            # hang from the UI's perspective. Retry once without VAD before
            # giving up, since whisper itself is decent at ignoring silence.
            print("⚠️ Empty transcript with VAD filtering — retrying without VAD...")
            segments, _info = model.transcribe(
                audio,
                language=whisper_lang,
                beam_size=5,
                vad_filter=False,
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()

        print(f"📝 Transcript: {text!r}" if text else "📝 Transcript empty (no speech detected).")
        return text

    def speak(self, text: str, language_name: str = "English") -> bytes:
        """Synthesize `text` to a WAV byte string in the given language."""
        if not text.strip():
            return b""

        voice_id = piper_voice_for(language_name)
        voice = _load_piper(voice_id)

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            # NOTE: piper-tts >=1.2 renamed this method. The old `synthesize()`
            # used to set the WAV header (channels/sample width/frame rate) on
            # the file object itself; the new `synthesize()` just yields raw
            # audio chunks and leaves the wave file untouched, which is why
            # `wave` raised "# channels not specified" when writeframes() ran
            # against a header-less file. `synthesize_wav` is the modern
            # equivalent that still configures the wave file for you.
            voice.synthesize_wav(text, wf)
        return buf.getvalue()


def wav_bytes_to_numpy(wav_bytes: bytes) -> np.ndarray:
    """
    Helper: parse a complete WAV byte string (what streamlit-mic-recorder hands
    us — it records via the browser's native MediaRecorder API, so unlike the
    old webrtc path this is already a well-formed WAV file, not raw PCM) into
    a mono float32 numpy array normalized to [-1, 1] and resampled to 16kHz
    for faster-whisper.
    """
    if not wav_bytes:
        return np.array([], dtype=np.float32)

    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())

    if sampwidth == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sampwidth == 1:
        # WAV 8-bit PCM is unsigned, centered at 128
        audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")

    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1).astype(np.float32)

    if framerate != 16000 and len(audio) > 0:
        duration = len(audio) / framerate
        target_len = max(1, int(duration * 16000))
        audio = np.interp(
            np.linspace(0, len(audio), target_len, endpoint=False),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)

    return audio


def pcm_to_numpy(pcm_bytes: bytes, sample_rate: int = 48000) -> np.ndarray:
    """
    Helper: convert raw int16 little-endian PCM (what streamlit-webrtc hands us)
    into a mono float32 numpy array normalized to [-1, 1] and resampled to 16kHz.
    A no-op resampler is used when sample rates match (the common case is fine
    since faster-whisper resamples internally).
    """
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    if sample_rate != 16000:
        # Lightweight linear resample. Whisper will resample again internally
        # so this just keeps the array length sensible.
        duration = len(audio) / sample_rate
        target_len = int(duration * 16000)
        audio = np.interp(
            np.linspace(0, len(audio), target_len, endpoint=False),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)
    return audio
