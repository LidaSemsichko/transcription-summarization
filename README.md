# transcription-summarization

# 🧠 Speech-to-Text + Summarization  
**UCU NLP Course Project — “NLP Fairies” (Lida, Khrystya, Yulia)**  
📅 *November 2025*

---

## 📖 Overview

This project implements an end-to-end system that:
1. Converts **spoken audio** into text using **OpenAI’s Whisper** model.  
2. Summarizes the resulting transcripts using an **abstractive transformer-based summarizer (BART-Large-CNN)**.  

🎯 **Goal:** Automate the process of transcribing and condensing multi-speaker conversations (e.g., meetings or interviews) into short, coherent summaries.

---

## 💡 Motivation

Every day, hours of speech are recorded in meetings, podcasts, and lectures — but most of it remains **unstructured**.  
Manual transcription and summarization are **slow**, **error-prone**, and **expensive**.  

An integrated **STT + Summarization** system makes spoken content:
- 🧭 Searchable  
- 💬 Easy to digest  
- 📊 Ready for analysis or downstream NLP tasks  

---

## 🗂️ Data Overview

### 🎧 Speech-to-Text dataset
- **Source:** FLEURS (clean subset)  
- **Languages:** English / Ukrainian  
- **Size:** 1,817 audio–text pairs (~4–5 hours total)  
- **Splits:** 80 % train | 10 % validation | 10 % test  
- **Average clip duration:** 6–12 seconds  
- **Average transcript length:** 10–20 words  

### 📰 Summarization dataset
- **Source:** [SAMSum dataset](https://huggingface.co/datasets/knkarthick/samsum)  
- **Domain:** multi-speaker chat dialogues  
- **Purpose:** fine-tuning the BART-Large-CNN summarizer  

---

## 🧹 Data Cleaning & Preprocessing

### Audio
- Converted all files → WAV (16 kHz mono)  
- Trimmed silence / normalized volume  
- Removed clips < 1 s or > 30 s  

### Text
- Lowercased + removed fillers and punctuation  
- Ensured alignment between audio and transcript  
- Split by speaker to avoid leakage between splits  

---

## 🧩 Model Architecture

### 1️⃣ Speech-to-Text: Whisper
- Pretrained multilingual encoder–decoder model from **OpenAI**  
- **Input:** log-mel spectrograms  
- **Output:** tokenized text sequences  
- Fine-tuned with frozen encoder to prevent overfitting on small data  
- Variable-length padding + masking for stable training  

### 2️⃣ Summarization: BART-Large-CNN
- Transformer encoder–decoder for **abstractive summarization**  
- Fine-tuned on **SAMSum** dialogues  
- Outputs short, human-like summaries of multi-speaker text  

---

## 🧮 Training Pipeline
COLLECT & SPLIT DATA
↓
PREPROCESS AUDIO + TEXT
↓
LOAD MODEL CONFIGURATIONS
↓
FREEZE ENCODER (Whisper)
↓
BATCH, PAD, MASK
↓
TRAIN + VALIDATE



---

## ⚙️ Model Inference Example

**Expected:**  
> “UN peacekeepers who arrived in Haiti after the 2010 earthquake are being blamed for the spread of the disease which started near the troops’ encampment.”

**Predicted:**  
> “You and peacekeepers who arrived in Hady after the 2010 earthquake are being blamed for the spread of the disease which started near the troops encampment.”

🗣️ *Close semantic match but minor pronunciation errors (Haiti → Hady).*

---

## 📰 Summarization Example

**Input dialogue:**  
> A: “Hey, did you finish the meeting notes?”  
> B: “Not yet, I’ll summarize them later.”  

**Generated summary:**  
> “They discussed finishing the meeting notes later.”  

---

## ⚠️ Challenges

| Type | Description | Mitigation |
|------|--------------|-------------|
| 🎧 Audio quality | Background noise, variable loudness | Normalization + silence trimming |
| 🧾 Text mismatch | Misalignment between audio & transcripts | Regex cleaning + manual spot-check |
| ⚖️ Imbalance | Variable clip lengths (1–30 s) | Quartile grouping + batching |
| 💻 Runtime limits | GPU memory & Colab timeouts | Checkpointing + smaller batch sizes |
| 🧩 Multilingual noise | Mixed EN/UA samples | Language-specific filtering |
| ✍ Summarization quality | Context loss in dialogues | Fine-tuning + ROUGE evaluation |

---

## 📈 Results (Preliminary)

| Model | Metric | Score |
|-------|---------|--------|
| Whisper (STT) | Word Error Rate (WER) | ~0.18–0.22 |
| BART-Large-CNN (Summary) | ROUGE-1 / ROUGE-L | ~0.42 / 0.39 |

*(Approximate scores based on validation subset.)*

---

## 🚀 Next Steps

1️⃣ **Integrate everything into one pipeline**  
   → 🎙️ Audio → 🧠 STT → 🧹 Preprocessing → 📰 Summarization → ✅ Output  

2️⃣ **Optimize model and data**  
   → Add more data via **augmentation** (noise, speed, pitch)  
   → Clean noisy/long clips 🧽  
   → Try smaller & faster models ⚡  

3️⃣ **Evaluate the final solution**  
   → Compute **WER / CER** for STT  
   → Compute **ROUGE / BERTScore** for summarization  
   → Compare baseline vs improved results 📊   

---

## 👩‍💻 Contributors

| Name | Role |
|------|------|
| **Lida** | Data preprocessing & EDA |
| **Khrystya** | Speech-to-Text (Whisper) |
| **Yulia** | Summarization (BART-Large-CNN) |

---

## 🧾 References
- OpenAI Whisper (2022) — [GitHub](https://github.com/openai/whisper)  
- Hugging Face Transformers (BART-Large-CNN)  
- SAMSum Dataset (2019) — Dialogue Summarization Benchmark  
- FLEURS Dataset (Google Research, 2022)  
