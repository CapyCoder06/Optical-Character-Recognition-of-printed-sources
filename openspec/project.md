# Renaissance OCR Project

## Overview

This project builds an OCR system for early modern historical documents.

Goal:

Recognize printed historical text from scanned documents and improve transcription quality using an LLM.

Core pipeline:

PDF scans  
→ image preprocessing  
→ text region detection  
→ OCR model  
→ LLM correction  
→ evaluation

Dataset contains scanned early modern printed documents with partial ground-truth transcriptions.

Target accuracy:

≥ 90% recognition accuracy.

---

## Problem Context

Historical documents present several challenges for OCR:

- non-standard typography
- degraded scans
- irregular layouts
- spelling variation

Traditional OCR models struggle with these characteristics.

This project combines:

deep learning OCR architectures + LLM correction.

---

## Tech Stack

Python

Libraries:

- PyTorch
- HuggingFace Transformers
- OpenCV
- pdf2image
- jiwer