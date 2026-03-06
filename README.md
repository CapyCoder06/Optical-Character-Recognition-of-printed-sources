# Optical Character Recognition for Historical Printed Sources

## Overview

This project focuses on building an **Optical Character Recognition (OCR) pipeline** for historical printed documents, particularly early modern Spanish sources.
The system aims to accurately detect and recognize text from scanned pages while ignoring decorative or non-text elements.

The project integrates **modern deep learning OCR architectures** together with **Large Language Models (LLMs) or Vision-Language Models (VLMs)** to improve text recognition quality.

---

## Objectives

The main objectives of this project are:

* Build a machine learning pipeline for recognizing printed historical text
* Detect the **main textual regions** while ignoring page embellishments
* Apply OCR models for text recognition
* Integrate **LLM/VLM-based post-processing** to improve OCR results
* Evaluate system performance using standard OCR metrics

The final system aims to achieve **at least 90% text recognition accuracy** on the dataset.

---

## Setup

### Python dependencies

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Windows note: `pdf2image` requires Poppler

`pdf2image` requires a Poppler install to convert PDFs to images on Windows.

- Install Poppler for Windows and ensure `pdftoppm.exe` is available.
- Either add Poppler to your `PATH`, or pass a `poppler_path` in the pipeline config (recommended).

---

## Running the OCR pipeline (CLI)

The pipeline will be runnable as:

```bash
python -m src.pipeline_runner --config configs/example.yaml --stages all
```

You can also run individual stages (e.g., `pdf_to_images`, `preprocess`, `detect`, `ocr`, `llm`, `eval`) once implemented.

## Dataset

The project uses historical Spanish printed documents including:

* *Instruccion de mercaderes* – Buendía
* *Tesoro de la lengua castellana* – Covarrubias
* *Tratado de nobleza* – Guardiola

These documents contain **non-standard typography and historical spelling**, making OCR recognition more challenging than modern printed text.

---

## System Pipeline

The OCR pipeline follows the stages below:

```
PDF Documents
      │
      ▼
PDF to Image Conversion
      │
      ▼
Image Preprocessing
  - Noise removal
  - Contrast enhancement
      │
      ▼
Text Region Detection
  - Identify main text areas
  - Remove decorative elements
      │
      ▼
OCR Recognition
  - Deep learning OCR models
  - CNN-RNN or Transformer-based models
      │
      ▼
LLM / VLM Post-processing
  - Correct OCR errors
  - Normalize historical text
      │
      ▼
Evaluation
```

---

## Model Architecture

The OCR system may use one of the following architectures:

### Convolutional-Recurrent Models

Examples:

* CRNN
* CNN + BiLSTM

### Transformer-based Models

Examples:

* TrOCR
* Vision Transformers

### Self-supervised Learning

Self-supervised approaches allow training models on unlabeled historical documents to improve generalization.

---

## LLM / VLM Integration

Large Language Models or Vision-Language Models are used as a **post-processing stage** to improve OCR output.

Tasks include:

* Correcting OCR mistakes
* Restoring missing characters
* Improving word segmentation
* Normalizing historical spellings

Possible models include:

* Gemini
* GPT-based models
* Other Vision-Language models

---

## Evaluation Metrics

Model performance will be evaluated using common OCR metrics:

**Character Error Rate (CER)**
Measures character-level recognition errors.

**Word Error Rate (WER)**
Measures word-level recognition errors.

**Accuracy**
Percentage of correctly recognized text.

These metrics allow quantitative comparison between different OCR models.

---

## Project Structure

```
project/
│
├── data/
│   └── Print/
│       ├── Buendia_Instruccion.pdf
│       ├── Covarrubias_Tesoro.pdf
│       └── Guardiola_Tratado.pdf
│
├── openspec/
│   ├── config.yaml
│   ├── project.md
│   └── changes/
│       └── historical-printed-ocr-pipeline/
│
└── README.md
```

---

## Expected Results

* OCR system capable of recognizing historical printed text
* Text extraction accuracy of **≥ 90%**
* Integration of LLM/VLM models to enhance OCR quality
* A structured pipeline for historical document digitization

---

## Applications

This project contributes to:

* **Digital humanities research**
* **Historical document digitization**
* **AI-assisted archival preservation**
* **Improved accessibility of historical texts**

---

## Future Improvements

Possible future work includes:

* Training OCR models on larger historical datasets
* Improving layout analysis for complex manuscripts
* Extending the pipeline to handwritten text recognition
* Applying multilingual OCR models

---

## Author

Project developed as part of an AI research project on **Optical Character Recognition for historical documents**.
