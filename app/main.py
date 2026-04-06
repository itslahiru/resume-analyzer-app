import json
import os

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

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
    generate_comparison_insights,
    extract_resume_text,
)
from app.report_generator import generate_pdf_report
from app.skills import SKILL_SYNONYMS
from app.database import (
    init_db,
    save_analysis,
    get_all_analyses,
    get_analysis_by_id,
    get_analyses_by_ids,
    get_report_payload_by_id,
    delete_analysis,
    search_analyses,
    create_user,
    get_user_by_username,
    get_user_by_id,
)
from app.auth import hash_password, verify_password

app = FastAPI()

SESSION_SECRET = os.getenv("SESSION_SECRET", "dev-secret-change-me")
app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET,
    same_site="lax",
    https_only=False,
)

init_db()

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")


def get_match_css_class(match_label: str) -> str:
    if match_label == "Strong Match":
        return "match-strong"
    if match_label == "Moderate Match":
        return "match-moderate"
    return "match-weak"


def get_current_user_from_session(request: Request):
    user_id = request.session.get("user_id")
    if not user_id:
        return None

    user = get_user_by_id(int(user_id))
    if user is None:
        request.session.clear()
        return None

    return user


def require_login(request: Request):
    user = get_current_user_from_session(request)
    if user is None:
        return RedirectResponse(url="/login", status_code=303)
    return user


@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    current_user = get_current_user_from_session(request)
    if current_user:
        return RedirectResponse(url="/", status_code=303)

    return templates.TemplateResponse(
        request=request,
        name="register.html",
        context={"error": None},
    )


@app.post("/register", response_class=HTMLResponse)
async def register_user(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...),
):
    username = username.strip().lower()

    if len(username) < 3:
        return templates.TemplateResponse(
            request=request,
            name="register.html",
            context={"error": "Username must be at least 3 characters long."},
        )

    if len(password) < 6:
        return templates.TemplateResponse(
            request=request,
            name="register.html",
            context={"error": "Password must be at least 6 characters long."},
        )

    if password != confirm_password:
        return templates.TemplateResponse(
            request=request,
            name="register.html",
            context={"error": "Passwords do not match."},
        )

    existing_user = get_user_by_username(username)
    if existing_user:
        return templates.TemplateResponse(
            request=request,
            name="register.html",
            context={"error": "That username is already taken."},
        )

    user_id = create_user(username, hash_password(password))
    request.session["user_id"] = user_id
    request.session["username"] = username

    return RedirectResponse(url="/", status_code=303)


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    current_user = get_current_user_from_session(request)
    if current_user:
        return RedirectResponse(url="/", status_code=303)

    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={"error": None},
    )


@app.post("/login", response_class=HTMLResponse)
async def login_user(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
):
    username = username.strip().lower()
    user = get_user_by_username(username)

    if user is None or not verify_password(password, user["password_hash"]):
        return templates.TemplateResponse(
            request=request,
            name="login.html",
            context={"error": "Invalid username or password."},
        )

    request.session["user_id"] = user["id"]
    request.session["username"] = user["username"]

    return RedirectResponse(url="/", status_code=303)


@app.post("/logout")
async def logout_user(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login", status_code=303)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    current_user = require_login(request)
    if isinstance(current_user, RedirectResponse):
        return current_user

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "results": None,
            "error": None,
            "job_description": "",
            "uploaded_filename": "",
            "current_user": current_user,
        },
    )


@app.post("/upload", response_class=HTMLResponse)
async def upload_resume(
    request: Request,
    file: UploadFile = File(...),
    job_description: str = Form(...),
):
    current_user = require_login(request)
    if isinstance(current_user, RedirectResponse):
        return current_user

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
                    "current_user": current_user,
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
                    "current_user": current_user,
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
                    "current_user": current_user,
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

        analysis_id = save_analysis(results, job_description, current_user["id"])
        results["analysis_id"] = analysis_id

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "results": results,
                "error": None,
                "job_description": job_description,
                "uploaded_filename": uploaded_filename,
                "current_user": current_user,
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
                "current_user": current_user,
            },
        )


@app.post("/download-report")
async def download_report(
    request: Request,
    report_payload: str = Form(...),
):
    current_user = require_login(request)
    if isinstance(current_user, RedirectResponse):
        return current_user

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


@app.get("/history", response_class=HTMLResponse)
async def history_page(
    request: Request,
    search_text: str = "",
    match_label: str = "",
    min_score: str = "",
):
    current_user = require_login(request)
    if isinstance(current_user, RedirectResponse):
        return current_user

    parsed_min_score = None
    if min_score.strip():
        try:
            parsed_min_score = float(min_score)
        except ValueError:
            parsed_min_score = None

    analyses = search_analyses(
        user_id=current_user["id"],
        search_text=search_text.strip(),
        match_label=match_label.strip(),
        min_score=parsed_min_score,
    )

    return templates.TemplateResponse(
        request=request,
        name="history.html",
        context={
            "analyses": analyses,
            "error": None,
            "filters": {
                "search_text": search_text,
                "match_label": match_label,
                "min_score": min_score,
            },
            "current_user": current_user,
        },
    )


@app.get("/history/{analysis_id}", response_class=HTMLResponse)
async def history_detail_page(request: Request, analysis_id: int):
    current_user = require_login(request)
    if isinstance(current_user, RedirectResponse):
        return current_user

    analysis = get_analysis_by_id(analysis_id, current_user["id"])

    if analysis is None:
        return templates.TemplateResponse(
            request=request,
            name="history_detail.html",
            context={
                "analysis": None,
                "error": "Analysis not found.",
                "current_user": current_user,
            },
        )

    return templates.TemplateResponse(
        request=request,
        name="history_detail.html",
        context={
            "analysis": analysis,
            "error": None,
            "current_user": current_user,
        },
    )


@app.get("/history/{analysis_id}/download-report")
async def download_history_report(request: Request, analysis_id: int):
    current_user = require_login(request)
    if isinstance(current_user, RedirectResponse):
        return current_user

    report_data = get_report_payload_by_id(analysis_id, current_user["id"])

    if report_data is None:
        return RedirectResponse(url="/history", status_code=303)

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


@app.post("/history/{analysis_id}/delete")
async def delete_history_entry(request: Request, analysis_id: int):
    current_user = require_login(request)
    if isinstance(current_user, RedirectResponse):
        return current_user

    delete_analysis(analysis_id, current_user["id"])
    return RedirectResponse(url="/history", status_code=303)


@app.post("/history/compare", response_class=HTMLResponse)
async def compare_history_entries(
    request: Request,
    selected_ids: list[int] = Form(...),
):
    current_user = require_login(request)
    if isinstance(current_user, RedirectResponse):
        return current_user

    if len(selected_ids) != 2:
        analyses = get_all_analyses(current_user["id"])
        return templates.TemplateResponse(
            request=request,
            name="history.html",
            context={
                "analyses": analyses,
                "error": "Please select exactly 2 analyses to compare.",
                "filters": {
                    "search_text": "",
                    "match_label": "",
                    "min_score": "",
                },
                "current_user": current_user,
            },
        )

    analyses = get_analyses_by_ids(selected_ids, current_user["id"])

    if len(analyses) != 2:
        return templates.TemplateResponse(
            request=request,
            name="compare.html",
            context={
                "left_analysis": None,
                "right_analysis": None,
                "comparison_insights": None,
                "error": "Could not load both selected analyses.",
                "current_user": current_user,
            },
        )

    left_analysis = analyses[0]
    right_analysis = analyses[1]

    comparison_insights = generate_comparison_insights(
        left_analysis=left_analysis,
        right_analysis=right_analysis,
    )

    return templates.TemplateResponse(
        request=request,
        name="compare.html",
        context={
            "left_analysis": left_analysis,
            "right_analysis": right_analysis,
            "comparison_insights": comparison_insights,
            "error": None,
            "current_user": current_user,
        },
    )