import io

import streamlit as st

from stt_model import load_stt, transcribe_audio
from llm_punctuation import punctuate_with_llm
from summarizer_model import load_summarizer, summarize_text
from llm_judge import evaluate_summary_with_llm
from build_docs import build_docx

try:
    from docx import Document  # лишається на випадок, якщо будеш ще щось юзати
except ImportError:
    Document = None


st.set_page_config(
    page_title="🎄 Transcription & Summarization 🎄",
    page_icon="🎅",
    layout="wide",
)

# ================== НОВИЙ НОВОРІЧНО-ПРЕЗЕНТАЦІЙНИЙ СТИЛЬ ==================

NEW_YEAR_CSS = """
<style>
/* Темний фон + легкий червоний відтінок, як у слайдах */
body {
    background: radial-gradient(circle at top left, #7f1d1d 0%, #020617 42%, #020617 100%);
    color: #e5e7eb;
}

/* Основний контейнер */
.main .block-container {
    padding-top: 2.2rem;
    padding-bottom: 2.6rem;
    border-radius: 24px;
}

/* HERO-блок у стилі презентації */
.hero-band {
    background: linear-gradient(135deg, #ef4444 0%, #b91c1c 55%, #111827 100%);
    border-radius: 24px;
    padding: 1.8rem 2.2rem;
    display: flex;
    gap: 2.2rem;
    align-items: center;
    box-shadow: 0 18px 40px rgba(0, 0, 0, 0.65);
    margin-bottom: 1.8rem;
    border: 1px solid rgba(254, 226, 226, 0.35);
}

.hero-left {
    flex: 2;
}

.hero-kicker {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #fee2e2;
    font-weight: 700;
    margin-bottom: 0.35rem;
}

.hero-title {
    font-size: 2.3rem;
    font-weight: 800;
    color: #fef2f2;
    text-shadow: 0 0 12px rgba(248, 250, 252, 0.4);
    margin-bottom: 0.4rem;
}

.hero-subtitle {
    font-size: 0.96rem;
    color: #fee2e2;
    max-width: 32rem;
}

.hero-right {
    flex: 1.4;
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
    align-items: flex-end;
}

.hero-badge-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    justify-content: flex-end;
}

.hero-stat-card {
    background: rgba(15, 23, 42, 0.92);
    border-radius: 18px;
    padding: 0.7rem 0.9rem;
    border: 1px solid rgba(254, 226, 226, 0.35);
    display: flex;
    flex-direction: column;
    gap: 0.1rem;
    min-width: 8.5rem;
}

.hero-stat-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    color: #fca5a5;
    font-weight: 700;
}

.hero-stat-value {
    font-size: 1.15rem;
    font-weight: 700;
    color: #fef2f2;
}

/* Маленькі бейджі (паралель до слайдів) */
.ny-badge {
    font-size: 0.8rem;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    background: rgba(15, 23, 42, 0.92);
    border: 1px solid rgba(254, 202, 202, 0.8);
    color: #fee2e2;
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
}

/* Текст під hero */
.helper-text {
    font-size: 0.9rem;
    color: #e5e7eb;
    margin-bottom: 0.8rem;
}

/* Радіо + аплоадер */
.stRadio > label, .stFileUploader label {
    font-weight: 600 !important;
}

/* Картки для колонок — dark card + червона смужка зверху */
.ny-card {
    background: rgba(15, 23, 42, 0.96);
    border-radius: 20px;
    padding: 1.1rem 1.15rem 1.35rem 1.15rem;
    box-shadow: 0 20px 45px rgba(0, 0, 0, 0.85);
    border: 1px solid rgba(248, 113, 113, 0.32);
    position: relative;
    overflow: hidden;
}

.ny-card::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    height: 6px;
    width: 100%;
    background: linear-gradient(90deg, #fecaca, #f97373, #fecaca);
}

/* Заголовки карток */
.ny-card h3 {
    margin-top: 0.5rem;
}

/* Текстові поля */
.stTextArea textarea {
    border-radius: 14px !important;
    background: rgba(15, 23, 42, 0.98) !important;
    border: 1px solid rgba(248, 113, 113, 0.55) !important;
    color: #e5e7eb !important;
    font-size: 0.9rem !important;
}

/* Кнопка */
.stButton > button {
    border-radius: 999px;
    background: linear-gradient(135deg, #f97316, #ea580c);
    color: white;
    font-weight: 700;
    border: none;
    box-shadow: 0 12px 30px rgba(234, 88, 12, 0.8);
    transition: transform 0.08s ease-out, box-shadow 0.08s ease-out;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 18px 46px rgba(248, 113, 22, 0.95);
}

/* Кнопка download */
.stDownloadButton > button {
    border-radius: 999px !important;
    border: 1px solid rgba(248, 113, 113, 0.7) !important;
    background: rgba(15, 23, 42, 0.96) !important;
    color: #fee2e2 !important;
    font-weight: 600 !important;
}

/* Сніжинки — лишаємо як було */
.snowflakes {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  pointer-events: none;
  z-index: 9999;
}

.snowflake {
  position: fixed;
  top: -10px;
  color: #e5e7eb;
  font-size: 1.1rem;
  text-shadow: 0 0 8px rgba(148, 163, 184, 0.9);
  animation-name: snowflakes-fall, snowflakes-shake;
  animation-duration: 12s, 4s;
  animation-timing-function: linear, ease-in-out;
  animation-iteration-count: infinite, infinite;
}

@keyframes snowflakes-fall {
  0% { top: -10%; }
  100% { top: 110%; }
}
@keyframes snowflakes-shake {
  0%, 100% { transform: translateX(0px); }
  50% { transform: translateX(25px); }
}

/* Позиції для кількох снежинок */
.snowflake:nth-child(1) { left: 5%;  animation-delay: 0s, 0s; }
.snowflake:nth-child(2) { left: 20%; animation-delay: 2s, 1s; }
.snowflake:nth-child(3) { left: 35%; animation-delay: 4s, 0.5s; }
.snowflake:nth-child(4) { left: 50%; animation-delay: 1s, 1.5s; }
.snowflake:nth-child(5) { left: 65%; animation-delay: 3s, 0s; }
.snowflake:nth-child(6) { left: 80%; animation-delay: 5s, 1s; }
.snowflake:nth-child(7) { left: 90%; animation-delay: 6s, 0.5s; }

</style>

<div class="snowflakes" aria-hidden="true">
  <div class="snowflake">❄</div>
  <div class="snowflake">✻</div>
  <div class="snowflake">❅</div>
  <div class="snowflake">✼</div>
  <div class="snowflake">❄</div>
  <div class="snowflake">✻</div>
  <div class="snowflake">❅</div>
</div>
"""

st.markdown(NEW_YEAR_CSS, unsafe_allow_html=True)


@st.cache_resource
def get_stt_model():
    return load_stt()


@st.cache_resource
def get_summarizer_model():
    return load_summarizer()


def main():
    # session_state для збереження результатів (щоб не зникали)
    if "transcript" not in st.session_state:
        st.session_state.transcript = None
    if "summary" not in st.session_state:
        st.session_state.summary = None
    if "evaluation" not in st.session_state:
        st.session_state.evaluation = None

    # ================= HERO-БЛОК =================

    st.markdown(
        """
        <div class="hero-band">
          <div class="hero-left">
            <div class="hero-kicker">Speech-to-Text · Summarization · Evaluation</div>
            <div class="hero-title">🎄 Meeting Whisperer 🎧</div>
            <div class="hero-subtitle">
              Prototype pipeline that turns noisy English audio into cleaned transcript,
              hierarchical summary and a gentle LLM evaluation — all in one festive UI.
            </div>
          </div>
          <div class="hero-right">
            <div class="hero-badge-row">
              <span class="ny-badge">🧠 Whisper-based STT</span>
              <span class="ny-badge">📚 Hierarchical summarization</span>
              <span class="ny-badge">🤖 LLM-as-a-Judge</span>
            </div>
            <div class="hero-stat-card">
              <div class="hero-stat-label">Mode</div>
              <div class="hero-stat-value">CPU · EN only</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="helper-text">'
        '1. Завантаж аудіофайл <b>англійською</b>.<br>'
        '2. Обери тип постпроцесингу: Simple або LLM punctuation.<br>'
        '3. Отримаєш транскрипт, summary та оцінку від LLM + DOCX-звіт.'
        '</div>',
        unsafe_allow_html=True,
    )

    mode = st.radio(
        "Тип постпроцесингу:",
        options=["Simple (local only)", "LLM punctuation (via API)"],
        horizontal=True,
        key="postprocess_mode",
    )

    audio_file = st.file_uploader(
        "Завантаж аудіо (mp3, wav, m4a, ogg):",
        type=["mp3", "wav", "m4a", "ogg"],
        key="stt_uploader",
    )

    # ================= ОБРОБКА АУДІО =================

    if audio_file is not None:
        if st.button("✨ Запустити транскрипцію + summary + оцінку", key="run_all"):
            st.info("Сорі, у мене тіко CPU, це може зайняти певний час...")

            asr_pipeline = get_stt_model()
            raw_text = transcribe_audio(
                audio_file,
                asr_pipeline,
                language="en",
            )

            if not raw_text or not raw_text.strip():
                st.error("Не вдалося отримати текст із аудіо 😥")
                return

            if "LLM punctuation" in mode:
                with st.spinner("Додаємо пунктуацію та структуру через LLM..."):
                    processed_text = punctuate_with_llm(raw_text, language_hint="en")
            else:
                processed_text = raw_text

            sum_tokenizer, sum_model = get_summarizer_model()
            with st.spinner("Генеруємо summary..."):
                summary = summarize_text(
                    processed_text,
                    tokenizer=sum_tokenizer,
                    model=sum_model,
                    max_new_tokens=260,
                    min_new_tokens=120,
                    use_llm_punctuation=True,
                )

            with st.spinner("LLM аналізує якість summary..."):
                evaluation = evaluate_summary_with_llm(
                    transcript=processed_text,
                    summary=summary,
                )

            # кладемо в session_state, щоб не зникало
            st.session_state.transcript = processed_text
            st.session_state.summary = summary
            st.session_state.evaluation = evaluation

            st.success("Успіх! Навіть нічо не впало! Результат нижче 🎁")

    # ================= ВІДМАЛЬОВУЄМО РЕЗУЛЬТАТ, ЯКЩО ВІН Є =================

    if st.session_state.transcript and st.session_state.summary:
        processed_text = st.session_state.transcript
        summary = st.session_state.summary
        evaluation = st.session_state.evaluation or ""

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<div class="ny-card">', unsafe_allow_html=True)
            st.subheader("📝 Транскрипт")
            if "LLM punctuation" in mode:
                st.caption("Після STT + LLM-пунктуації.")
            else:
                st.caption("Після STT (нарізка + прибрані дикі повтори).")
            st.text_area(
                "Транскрипт:",
                value=processed_text,
                height=420,
                key="processed_text_area",
                label_visibility="collapsed",
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="ny-card">', unsafe_allow_html=True)
            st.subheader("📚 Summary")
            st.caption("Згенеровано з обробленого транскрипту.")
            st.text_area(
                "Summary:",
                value=summary,
                height=420,
                key="summary_from_stt_area",
                label_visibility="collapsed",
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="ny-card">', unsafe_allow_html=True)
            st.subheader("🧠 LLM Evaluation")
            st.caption("Мʼяка оцінка summary від LLM (0–10) + короткий фідбек.")
            st.text_area(
                "Evaluation:",
                value=evaluation,
                height=420,
                key="evaluation_area",
                label_visibility="collapsed",
            )
            st.markdown("</div>", unsafe_allow_html=True)

        # DOCX download
        try:
            docx_bytes = build_docx(processed_text, summary, evaluation)
            st.download_button(
                label="⬇️ Завантажити все як DOCX",
                data=docx_bytes,
                file_name="transcription_summary_evaluation_new_year.docx",
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "wordprocessingml.document"
                ),
                key="download_docx_button",
            )
        except RuntimeError as e:
            st.warning(str(e))


if __name__ == "__main__":
    main()
