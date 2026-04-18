# Resume Analyzer Web App

A multi-user resume analysis web application built with **FastAPI**, **SQLite**, **OCR**, and **NLP**.  
The system allows users to upload resume PDFs, compare them against job descriptions, receive structured feedback, save analysis history, compare resume versions, and export reports as PDF files.

---

## Overview

This project was built as a personal product-style application to simulate a realistic resume screening and feedback workflow.

The app analyzes resumes against job descriptions and provides:
- matched and missing skills
- required vs preferred skill breakdown
- section-aware scoring
- semantic similarity scoring
- formatting feedback
- downloadable reports
- saved analysis history
- side-by-side comparison of resume versions
- user authentication

---

## Features

### Resume Analysis
- Upload resumes in **PDF** format
- Extract text from standard text-based PDFs
- **OCR fallback** for scanned/image-based PDFs
- Clean and preprocess extracted text
- NLP-based **keyword extraction**
- Synonym-aware skill detection
- Section-aware resume parsing
- Skill evidence extraction from resume sections

### Job Matching
- Parse job descriptions from pasted text
- Detect **required** and **preferred** skills
- Compare resume skills against job requirements
- Show:
  - matched skills
  - missing skills
  - missing must-have skills
  - missing preferred skills
- Generate:
  - weighted skill match score
  - semantic similarity score
  - combined overall match score
  - match label (Strong / Moderate / Weak)

### Feedback
- Human-readable feedback summary
- Top strengths
- Top risks
- Recommendations for improvement
- Formatting and structure analysis, including:
  - missing email or phone number
  - missing core sections
  - dense text detection
  - weak bullet usage
  - weak action verbs
  - low measurable-achievement signals
  - technical profile link suggestions

### Reports
- Export current analysis as a **PDF report**
- Re-download reports later from saved history

### History and Comparison
- Save completed analyses in **SQLite**
- View saved analysis history
- Open detailed history entries
- Search and filter history by:
  - filename
  - match label
  - minimum score
- Compare two saved analyses side by side
- Delete saved analyses

### Authentication
- User registration
- User login / logout
- Session-based authentication
- Per-user saved analysis history

### UI
- Styled analyzer page
- Styled login and register pages
- Styled history page
- Styled comparison page
- Loading state during analysis
- Score breakdown bars
- Tag-based skill display

---

## Tech Stack

### Backend
- FastAPI
- SQLite
- Jinja2 Templates
- Starlette Session Middleware

### NLP / Analysis
- YAKE
- Sentence Transformers
- Synonym-aware rule-based skill matching

### PDF / OCR
- pypdf
- PyMuPDF
- pytesseract
- Tesseract OCR
- Pillow

### Report Generation
- ReportLab

### Frontend
- HTML
- CSS
- JavaScript

---

Demo Video: https://youtu.be/mzO-KwtWfDA

## Project Structure

```text
resume_analyzer_app/
│
├── app/
│   ├── main.py
│   ├── analyzer.py
│   ├── auth.py
│   ├── database.py
│   ├── report_generator.py
│   ├── skills.py
│   ├── templates/
│   │   ├── index.html
│   │   ├── login.html
│   │   ├── register.html
│   │   ├── history.html
│   │   ├── history_detail.html
│   │   └── compare.html
│   └── static/
│       ├── styles.css
│       ├── app.js
│       └── theme.js
│
├── requirements.txt
├── README.md
├── .gitignore
├── resume_analyzer.db
└── sample_data/
