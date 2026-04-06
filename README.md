# Resume Analyzer Web Application

A FastAPI-based web application that allows users to upload a resume in PDF format, compare it against a job description, and receive structured feedback.

## Features

- Upload resume PDFs
- Extract text from normal PDFs
- OCR fallback for scanned PDFs
- NLP-based keyword extraction using YAKE
- Synonym-aware skill matching
- Weighted skill scoring for required vs preferred job skills
- Semantic similarity scoring using Sentence Transformers
- Formatting and structure checks
- Human-readable feedback summary
- Downloadable PDF analysis report

## Tech Stack

- **Backend:** FastAPI
- **Frontend:** HTML, CSS, JavaScript, Jinja2 templates
- **PDF Text Extraction:** pypdf
- **OCR:** pytesseract + Tesseract OCR + PyMuPDF
- **Keyword Extraction:** YAKE
- **Semantic Matching:** sentence-transformers
- **PDF Report Generation:** ReportLab

## Project Structure

```text
resume_analyzer_app/
│
├── app/
│   ├── main.py
│   ├── analyzer.py
│   ├── skills.py
│   ├── report_generator.py
│   ├── templates/
│   │   └── index.html
│   └── static/
│       ├── styles.css
│       └── app.js
│
├── requirements.txt
├── README.md
└── .gitignore