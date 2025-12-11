import re


def _normalize_spaces(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _insert_basic_punctuation(text: str) -> str:
    patterns = [
        r"\s+(but)\s+",
        r"\s+(however)\s+",
        r"\s+(and then)\s+",
        r"\s+(eventually)\s+",
        r"\s+(finally)\s+",
    ]

    for pat in patterns:
        text = re.sub(
            pat,
            lambda m: ". " + m.group(1) + " ",
            text,
            flags=re.IGNORECASE,
        )

    text = re.sub(r"\.\s*\.", ".", text)

    text = text.strip()
    if text and text[-1] not in ".?!":
        text += "."

    return text


def _split_into_sentences(text: str) -> list[str]:
    text = re.sub(r"([.!?])\s+", r"\1<SENT_SPLIT>", text)
    parts = [s.strip() for s in text.split("<SENT_SPLIT>") if s.strip()]
    return parts


def _capitalize_sentences(sentences: list[str]) -> list[str]:
    res = []
    for s in sentences:
        if not s:
            continue
        s = s.strip()
        if s and s[0].isalpha():
            s = s[0].upper() + s[1:]
        res.append(s)
    return res


def postprocess_transcript(text: str) -> str:
    if not text or not text.strip():
        return text
    text = _normalize_spaces(text)
    text = _insert_basic_punctuation(text)
    sentences = _split_into_sentences(text)
    sentences = _capitalize_sentences(sentences)

    result = "\n".join(sentences)
    return result
