# Technical Design

## Pipeline Architecture

PDF document
→ convert to images
→ preprocess image
→ detect text regions
→ OCR model
→ LLM correction
→ evaluation

## Modules

src/

preprocessing.py  
text_detection.py  
ocr_model.py  
llm_correction.py  
evaluation.py  

## OCR Model

Recommended architecture:

Transformer OCR model (TrOCR)

Alternative architectures:

- CRNN
- self-supervised OCR models

## LLM Integration

LLM is used as a post-processing stage.

Purpose:

- correct OCR errors
- improve transcription quality
- normalize spelling

## Evaluation Metrics

Character Error Rate (CER)

CER = (Substitutions + Insertions + Deletions) / Total Characters

Word Error Rate (WER)

WER = (Substitutions + Insertions + Deletions) / Total Words