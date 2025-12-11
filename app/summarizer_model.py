import os
import re
from typing import Tuple, List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from llm_punctuation import punctuate_with_llm


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "summarizer")

MAX_CHUNK_WORDS = 260       
CHUNK_OVERLAP_WORDS = 40

DEFAULT_MIN_NEW_TOKENS = 120
DEFAULT_MAX_NEW_TOKENS = 260
DEFAULT_NUM_BEAMS = 5
DEFAULT_NO_REPEAT_NGRAM_SIZE = 4
DEFAULT_REPETITION_PENALTY = 1.2

DEFAULT_USE_LLM_PUNCTUATION = True


def load_summarizer() -> Tuple[AutoTokenizer, AutoModelForSeq2SeqLM]:
    if not os.path.isdir(MODEL_PATH):
        raise RuntimeError(
            f"Не знайдено папку з моделлю summarizer: {MODEL_PATH}\n"
            f"Переконайся, що чекпойнт лежить саме там."
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model


def _normalize_spaces(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _basic_sentence_ends(text: str) -> str:
    text = re.sub(r"\s+(but)\s+", r". \1 ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+(however)\s+", r". \1 ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+(then)\s+", r". \1 ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+(eventually)\s+", r". \1 ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+(finally)\s+", r". \1 ", text, flags=re.IGNORECASE)

    text = re.sub(r"\.\s*\.", ".", text)
    return text


def _normalize_text_for_summary(text: str) -> str:
    text = _normalize_spaces(text)
    if not text:
        return text
    if text[0].isalpha():
        text = text[0].upper() + text[1:]
    if not re.search(r"[\.!?]", text):
        text = _basic_sentence_ends(text)
    if text and text[-1] not in ".!?":
        text += "."

    return text


def _chunk_text_by_words_with_overlap(
    text: str,
    max_words: int,
    overlap_words: int,
) -> List[str]:
    words = text.split()
    n = len(words)
    if n <= max_words:
        return [text]

    chunks: List[str] = []
    start = 0
    while start < n:
        end = min(start + max_words, n)
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end == n:
            break
        start = max(0, end - overlap_words)

    return chunks


def _summarize_single_chunk(
    text: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForSeq2SeqLM,
    max_new_tokens: int,
    min_new_tokens: int,
    num_beams: int,
) -> str:
    text = text.strip()
    if not text:
        return ""

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            num_beams=num_beams,
            no_repeat_ngram_size=DEFAULT_NO_REPEAT_NGRAM_SIZE,
            repetition_penalty=DEFAULT_REPETITION_PENALTY,
            length_penalty=1.0,
            early_stopping=True,
        )

    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return summary.strip()


def summarize_text(
    text: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForSeq2SeqLM,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    min_new_tokens: int = DEFAULT_MIN_NEW_TOKENS,
    num_beams: int = DEFAULT_NUM_BEAMS,
    use_llm_punctuation: bool = DEFAULT_USE_LLM_PUNCTUATION,
) -> str:
    if not text or not text.strip():
        return ""

    raw_text = text
    if use_llm_punctuation:
        try:
            raw_text = punctuate_with_llm(raw_text, language_hint="en")
        except Exception:
            raw_text = text

    prepped_text = _normalize_text_for_summary(raw_text)
    if not prepped_text:
        return ""

    words = prepped_text.split()
    if len(words) <= MAX_CHUNK_WORDS:
        return _summarize_single_chunk(
            prepped_text,
            tokenizer,
            model,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            num_beams=num_beams,
        )

    chunks = _chunk_text_by_words_with_overlap(
        prepped_text,
        max_words=MAX_CHUNK_WORDS,
        overlap_words=CHUNK_OVERLAP_WORDS,
    )

    chunk_summaries: List[str] = []
    for chunk in chunks:
        s = _summarize_single_chunk(
            chunk,
            tokenizer,
            model,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            num_beams=num_beams,
        )
        if s:
            chunk_summaries.append(s)

    if not chunk_summaries:
        return ""

    combined = " ".join(chunk_summaries)
    combined = _normalize_text_for_summary(combined)

    final_summary = _summarize_single_chunk(
        combined,
        tokenizer,
        model,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        num_beams=num_beams,
    )

    return final_summary.strip()
