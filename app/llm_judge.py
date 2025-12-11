import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def evaluate_summary_with_llm(transcript: str, summary: str) -> str:
    prompt = f"""
You are evaluating a summary of a noisy ASR (automatic speech recognition) transcript.
The transcript may contain errors, repetitions and imperfect punctuation.
The summary is created by a small local model, not by a powerful LLM.
The project is in a prototype stage, and your goal is to give **supportive, realistic, but gently optimistic** feedback.

Given:
1) ORIGINAL TRANSCRIPT (noisy):
{transcript}

2) GENERATED SUMMARY:
{summary}

Evaluate using this rubric:

1. Faithfulness (0–4):
   - 4: No important hallucinations; summary mostly stays true to the transcript.
   - 3: Minor inaccuracies, but overall faithful.
   - 2: Some distortions, but the main idea is still roughly correct.
   - 1: Many inaccuracies; hard to trust.
   - 0: Completely misleading or invented.

2. Coverage (0–3):
   - 3: Covers most main events / ideas of the story.
   - 2: Covers several key points, but misses important aspects.
   - 1: Only a small fragment of the story is reflected.
   - 0: Almost nothing relevant to the original.

3. Clarity & coherence (0–3):
   - 3: Easy enough to read; logical enough for a prototype.
   - 2: Understandable but somewhat messy.
   - 1: Hard to follow.
   - 0: Almost unreadable.

Total score = Faithfulness + Coverage + Clarity  (0–10).

VERY IMPORTANT CALIBRATION:
- Remember: this is a prototype pipeline (noisy STT + small summarizer).
- Use a **soft, generous scale**:
  - 9–10: excellent for this prototype.
  - 8: good, minor issues.
  - 7: usable summary with noticeable imperfections (this is still a positive result).
  - 5–6: significant problems, but some value remains.
  - 0–4: only if there are serious failures (strong hallucinations, almost no coverage, or summary not about this text).
- In borderline cases, round **up**, not down.
- Do NOT be perfectionistic about literary style or small omissions.

Return your answer in exactly this format (English):

Score: X/10
Faithfulness: F/4
Coverage: C/3
Clarity: L/3
Explanation: <2–5 sentences: why this score, in a neutral and kind way>
Missing points: <key aspects of the story that the summary did NOT cover, or 'none'>
Hallucinations: <problematic invented facts, or 'none'>
Suggestions: <1–3 concrete, practical tips how to improve this kind of summary in the future>
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini", 
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a fair but gently optimistic evaluator of summaries "
                    "generated from noisy ASR transcripts in a prototype project. "
                    "You focus on encouragement and practical suggestions."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=380,
        temperature=0.0,
    )

    return resp.choices[0].message.content.strip()
