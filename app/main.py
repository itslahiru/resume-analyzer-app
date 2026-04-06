import json

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.analyzer import (
    clean_text,
    extract_keywords,
    compare_resume_to_job_section_aware,
    calculate_section_aware_skill_match_percentage,
    calculate_semantic_similarity,
    calculate_combined_match_score,
    analyze_formatting_issues,
    get_match_label,
    generate_recommendations,
    generate_feedback_summary,
    generate_top_strengths,
    generate_top_risks,
    extract_resume_text,
)
from app.report_generator import generate_pdf_report
from app.skills import SKILL_SYNONYMS

app = FastAPI()

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")


def get_match_css_class(match_label: str) -> str:
    if match_label == "Strong Match":
        return "match-strong"
    if match_label == "Moderate Match":
        return "match-moderate"
    return "match-weak"


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "results": None,
            "error": None,
            "job_description": "",
            "uploaded_filename": "",
        },
    )


@app.post("/upload", response_class=HTMLResponse)
async def upload_resume(
    request: Request,
    file: UploadFile = File(...),
    job_description: str = Form(...),
):
    uploaded_filename = file.filename or ""

    try:
        if file.content_type != "application/pdf":
            return templates.TemplateResponse(
                request=request,
                name="index.html",
                context={
                    "results": None,
                    "error": "Only PDF files are allowed.",
                    "job_description": job_description,
                    "uploaded_filename": uploaded_filename,
                },
            )

        file_bytes = await file.read()

        if not file_bytes:
            return templates.TemplateResponse(
                request=request,
                name="index.html",
                context={
                    "results": None,
                    "error": "Uploaded file is empty.",
                    "job_description": job_description,
                    "uploaded_filename": uploaded_filename,
                },
            )

        extracted_text, extraction_method = extract_resume_text(file_bytes)

        if not extracted_text:
            return templates.TemplateResponse(
                request=request,
                name="index.html",
                context={
                    "results": None,
                    "error": "Could not extract readable text from the PDF.",
                    "job_description": job_description,
                    "uploaded_filename": uploaded_filename,
                },
            )

        cleaned_resume_text = clean_text(extracted_text)
        cleaned_job_text = clean_text(job_description)

        keywords = extract_keywords(cleaned_resume_text, top_n=15)

        comparison = compare_resume_to_job_section_aware(
            raw_resume_text=extracted_text,
            resume_text=cleaned_resume_text,
            raw_job_text=job_description,
            skill_map=SKILL_SYNONYMS,
        )

        skill_match_percentage = calculate_section_aware_skill_match_percentage(
            matched_skills=comparison["matched_skills"],
            job_skill_weights=comparison["job_skill_weights"],
            skill_evidence_map=comparison["skill_evidence_map"],
        )

        semantic_similarity_percentage = calculate_semantic_similarity(
            cleaned_resume_text,
            cleaned_job_text,
        )

        combined_match_score = calculate_combined_match_score(
            skill_match_percentage=skill_match_percentage,
            semantic_similarity_percentage=semantic_similarity_percentage,
        )

        formatting_issues = analyze_formatting_issues(
        extracted_text,
        detected_resume_skills=comparison["resume_skills"],
        )
        match_label = get_match_label(combined_match_score)
        match_css_class = get_match_css_class(match_label)

        recommendations = generate_recommendations(
            missing_skills=comparison["missing_skills"],
            formatting_issues=formatting_issues,
            required_missing_skills=comparison["required_missing_skills"],
            preferred_missing_skills=comparison["preferred_missing_skills"],
        )

        feedback_summary = generate_feedback_summary(
            job_match_percentage=combined_match_score,
            matched_skills=comparison["matched_skills"],
            missing_skills=comparison["missing_skills"],
            formatting_issues=formatting_issues,
            required_missing_skills=comparison["required_missing_skills"],
        )

        top_strengths = generate_top_strengths(
            matched_skills=comparison["matched_skills"],
            matched_skill_evidence=comparison["matched_skill_evidence"],
            semantic_similarity_percentage=semantic_similarity_percentage,
        )

        top_risks = generate_top_risks(
            required_missing_skills=comparison["required_missing_skills"],
            preferred_missing_skills=comparison["preferred_missing_skills"],
            formatting_issues=formatting_issues,
        )

        results = {
            "filename": uploaded_filename,
            "resume_keywords": keywords,
            "resume_skills": comparison["resume_skills"],
            "job_skills": comparison["job_skills"],
            "required_skills": comparison["required_skills"],
            "preferred_skills": comparison["preferred_skills"],
            "matched_skills": comparison["matched_skills"],
            "missing_skills": comparison["missing_skills"],
            "required_missing_skills": comparison["required_missing_skills"],
            "preferred_missing_skills": comparison["preferred_missing_skills"],
            "matched_skill_evidence": comparison["matched_skill_evidence"],
            "formatting_issues": formatting_issues,
            "skill_match_percentage": skill_match_percentage,
            "semantic_similarity_percentage": semantic_similarity_percentage,
            "job_match_percentage": combined_match_score,
            "match_label": match_label,
            "match_css_class": match_css_class,
            "recommendations": recommendations,
            "feedback_summary": feedback_summary,
            "top_strengths": top_strengths,
            "top_risks": top_risks,
            "extraction_method": extraction_method,
        }

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "results": results,
                "error": None,
                "job_description": job_description,
                "uploaded_filename": uploaded_filename,
            },
        )

    except Exception as e:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "results": None,
                "error": f"Something went wrong: {str(e)}",
                "job_description": job_description,
                "uploaded_filename": uploaded_filename,
            },
        )


@app.post("/download-report")
async def download_report(report_payload: str = Form(...)):
    report_data = json.loads(report_payload)
    pdf_buffer = generate_pdf_report(report_data)

    safe_name = report_data.get("filename", "resume").replace(".pdf", "")
    output_name = f"{safe_name}_analysis_report.pdf"

    return StreamingResponse(
        iter([pdf_buffer.getvalue()]),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{output_name}"'
        },
    )