import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def punctuate_with_llm(text: str, language_hint: str = "en") -> str:
    if not text or not text.strip():
        return text
    if len(text.split()) < 5:
        return text.strip()

    lang_phrase = {
        "en": "English",
        "uk": "Ukrainian",
        "auto": "the same language as the input",
    }.get(language_hint, "the same language as the input")

    prompt = f"""
You are a transcription post-processing assistant.

Task:
- Take the raw automatic speech recognition transcript below.
- Only add punctuation (.,?!), capitalization, and paragraph breaks.
- Do NOT translate the text.
- Do NOT summarize or shorten it.
- Do NOT add new information.
- Just rewrite it in {lang_phrase} with proper punctuation and natural line breaks.

Raw transcript:
{text}
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0,
    )

    return response.choices[0].message.content.strip()
