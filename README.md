# transcription-summarization

## Speech-to-Text and Summarization  
**UCU NLP Course Project — “NLP Fairies”**  
**Authors:** Lida, Khrystya, Yulia  
**Date:** November 2025

**Link for endpoints**: https://drive.google.com/drive/folders/1bu3kFe05f2nhmoxSZHienf1_DUMAw0KM?usp=drive_link
(you need to download those 2 folders in app folder)
---

## Overview

This project implements an end-to-end Natural Language Processing pipeline for processing spoken audio.  
The system:

1. Converts raw audio recordings into text using the Whisper automatic speech recognition model.
2. Applies post-processing to reduce transcription artifacts.
3. Generates abstractive summaries of long transcripts using a transformer-based summarization model.
4. Optionally evaluates the quality of generated summaries using a large language model acting as a judge.

The pipeline is designed to work with long-form conversational or narrative speech such as meetings, interviews, or recorded stories.

---

## Motivation

Large volumes of spoken content are produced daily in meetings, lectures, interviews, and podcasts.  
However, spoken data is difficult to search, analyze, and reuse without structured textual representations.

Manual transcription and summarization are time-consuming and error-prone.  
An automated speech-to-text and summarization pipeline enables:

- Faster access to spoken content  
- Efficient information retrieval  
- Downstream NLP tasks such as classification, retrieval, or analytics  

This project explores how modern transformer-based models can be combined into a practical and modular pipeline.

---

## Data Overview

### Speech-to-Text Data

- Source: FLEURS dataset (clean subset)
- Languages: English, Ukrainian
- Total duration: approximately 4–5 hours
- Number of samples: 1,817 audio–text pairs
- Data splits:
  - 80% training
  - 10% validation
  - 10% test
- Average audio length: 6–12 seconds
- Average transcript length: 10–20 words

### Summarization Data

- Source: SAMSum dataset
- Domain: multi-speaker chat dialogues
- Purpose: fine-tuning and evaluation of abstractive summarization models
- Input: conversational dialogue text
- Output: short abstractive summaries

---

## Data Preprocessing

### Audio Preprocessing

- All audio files converted to WAV format
- Sample rate standardized to 16 kHz mono
- Silence trimming applied
- Loudness normalized
- Audio clips shorter than 1 second or longer than 30 seconds removed

### Text Preprocessing

- Lowercasing
- Removal of filler words and excessive punctuation
- Alignment checks between audio and transcript
- Speaker-based segmentation to prevent data leakage across splits

---

## Model Architecture

### Speech-to-Text: Whisper

- Pretrained encoder–decoder transformer model
- Input: log-mel spectrograms extracted from audio
- Output: tokenized text sequences
- Multilingual support (English and Ukrainian)
- Encoder frozen during fine-tuning to reduce overfitting
- Variable-length padding and attention masking used during training

### Summarization: Transformer-based Abstractive Model

- Encoder–decoder transformer architecture
- Fine-tuned on dialogue summarization data
- Generates concise abstractive summaries rather than extractive copies
- Designed to handle conversational and narrative text

---

## Inference Pipeline

The full inference pipeline consists of the following stages:

1. Audio loading and normalization  
2. Chunking long audio into short segments (approximately 20–30 seconds)  
3. Speech-to-text transcription for each segment  
4. Text cleaning and removal of repeated or hallucinated phrases  
5. Optional punctuation and formatting refinement using a language model  
6. Hierarchical summarization of long transcripts  
7. Optional evaluation of summary quality using an LLM-based judge  
8. Export of results into structured document format  

---

## Hierarchical Summarization

Directly summarizing long transcripts often leads to poor results due to context limits and model bias toward early text.

To address this, hierarchical summarization is used:

1. The transcript is split into chunks of approximately 700–900 words  
2. Each chunk is summarized independently  
3. Intermediate summaries are concatenated  
4. A final summary is generated from the intermediate summaries  

This approach improves coherence, reduces hallucinations, and preserves information from the entire transcript.

---

## Model Inference Example

**Speech-to-text example**

Expected transcript:

> “UN peacekeepers who arrived in Haiti after the 2010 earthquake are being blamed for the spread of the disease which started near the troops’ encampment.”

Predicted transcript:

> “UN peacekeepers who arrived in Hady after the 2010 earthquake are being blamed for the spread of the disease which started near the troops encampment.”

The transcription preserves semantic meaning but contains minor pronunciation-based errors.

---

## Summarization Example

**Input dialogue:**

> A: “Hey, did you finish the meeting notes?”  
> B: “Not yet, I’ll summarize them later.”

**Generated summary:**

> “They discussed finishing the meeting notes later.”

---

## Evaluation

### Speech-to-Text Evaluation

- Metric: Word Error Rate (WER)
- Observed WER on validation set: approximately 0.18–0.22
- Common errors:
  - Named entities
  - Proper nouns
  - Accents and pronunciation variations

### Summarization Evaluation

- Metrics: ROUGE-1 and ROUGE-L
- Approximate scores on validation subset:
  - ROUGE-1: 0.42
  - ROUGE-L: 0.39

### LLM-Based Evaluation

In addition to automatic metrics, an optional evaluation step uses a large language model to:

- Compare the transcript and generated summary
- Assign a quality score (0–10)
- Provide qualitative feedback on coverage, coherence, and hallucinations

This approach allows semantic-level assessment beyond n-gram overlap metrics.

---

## Challenges and Limitations

| Category | Description | Mitigation |
|--------|------------|------------|
| Audio quality | Background noise and inconsistent loudness | Normalization and silence trimming |
| Transcription artifacts | Repetition and hallucinations | Audio chunking and text cleaning |
| Data imbalance | Variable clip lengths | Length-based batching |
| Computational limits | Limited GPU/CPU resources | Smaller batch sizes and checkpointing |
| Multilingual noise | Mixed-language samples | Language filtering |
| Long-context summarization | Loss of late-context information | Hierarchical summarization |

---

## Implementation Details

- Frameworks: PyTorch, Hugging Face Transformers
- Audio processing: ffmpeg, librosa
- Models loaded locally or from Hugging Face repositories
- Optional integration with external LLM APIs for punctuation and evaluation
- User interface implemented with Streamlit
- Output export supported in DOCX format

---

## Future Work

- Extend evaluation with BERTScore and human evaluation
- Improve punctuation and formatting robustness
- Experiment with faster and smaller speech models
- Add speaker diarization
- Support additional languages
- Deploy as a web service

---

## Contributors

| Name | Responsibility |
|------|----------------|
| Lida | Data preprocessing and exploratory analysis |
| Khrystya | Speech-to-text pipeline |
| Yulia | Summarization model |

---

## References

- OpenAI Whisper (2022)
- Hugging Face Transformers
- SAMSum Dataset (2019)
- FLEURS Dataset (Google Research, 2022)
