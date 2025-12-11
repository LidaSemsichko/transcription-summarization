import io
import os
import re
from typing import Optional, List

import numpy as np
import librosa
from transformers import pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "transcribation")


CHUNK_DURATION_SEC = 10.0     
MIN_ENERGY_THRESHOLD = 0.01   


def load_stt():
    if not os.path.isdir(MODEL_PATH):
        raise RuntimeError(f"Не знайдено папку з моделлю Whisper: {MODEL_PATH}")

    asr = pipeline(
        task="automatic-speech-recognition",
        model=MODEL_PATH,
    )
    return asr


def _load_audio_from_uploaded_file(file, target_sr: int = 16000):
    data = file.read()
    buffer = io.BytesIO(data)

    audio, sr = librosa.load(buffer, sr=None, mono=True)

    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return audio, sr


def _split_into_chunks(audio: np.ndarray, sr: int, chunk_duration_sec: float) -> List[np.ndarray]:
    samples_per_chunk = int(sr * chunk_duration_sec)
    total = len(audio)

    chunks: List[np.ndarray] = []
    for start in range(0, total, samples_per_chunk):
        end = min(start + samples_per_chunk, total)
        chunk = audio[start:end]

        if len(chunk) == 0:
            continue

        if np.mean(np.abs(chunk)) < MIN_ENERGY_THRESHOLD:
            continue

        chunks.append(chunk)

    return chunks


def _clean_repetitions(text: str) -> str:
    if not text:
        return text

    text = re.sub(
        r"\b([\w'-]+)(\s+\1\b){2,}",
        r"\1 \1",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(game-start\s+){2,}",
        "game-start ",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"(the night of\s+){3,}",
        "the night of ",
        text,
        flags=re.IGNORECASE,
    )

    return text


def transcribe_audio(file, asr_pipeline, language: Optional[str] = None) -> str:
    audio, sr = _load_audio_from_uploaded_file(file, target_sr=16000)

    chunks = _split_into_chunks(audio, sr, CHUNK_DURATION_SEC)
    if not chunks:
        return ""

    parts: List[str] = []
    generate_kwargs = {}
    is_multilingual = False
    try:
        cfg = getattr(asr_pipeline.model, "config", None)
        if cfg is not None and getattr(cfg, "is_multilingual", False):
            is_multilingual = True
    except Exception:
        pass

    if is_multilingual and language is not None:
        generate_kwargs["task"] = "transcribe"
        generate_kwargs["language"] = language

    for chunk in chunks:
        input_data = {
            "array": chunk,
            "sampling_rate": sr,
        }

        if generate_kwargs:
            result = asr_pipeline(input_data, generate_kwargs=generate_kwargs)
        else:
            result = asr_pipeline(input_data)

        text_piece = ""

        if isinstance(result, dict):
            if result.get("text"):
                text_piece = result["text"]
            elif "chunks" in result:
                text_piece = " ".join(
                    ch.get("text", "") for ch in result["chunks"]
                )

        text_piece = (text_piece or "").strip()
        if text_piece:
            parts.append(text_piece)
    full_text = " ".join(parts).strip()
    full_text = _clean_repetitions(full_text)

    return full_text
