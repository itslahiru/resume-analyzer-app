import re
from io import BytesIO
from pathlib import Path
from functools import lru_cache

import fitz  # PyMuPDF
import pytesseract
import yake
from PIL import Image
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util


REQUIRED_SECTION_MARKERS = (
    "required skills",
    "requirements",
    "must have",
    "must-have",
    "essential skills",
    "mandatory skills",
    "required qualifications",
)

PREFERRED_SECTION_MARKERS = (
    "preferred skills",
    "preferred qualifications",
    "nice to have",
    "nice-to-have",
    "bonus skills",
    "desired skills",
)

SECTION_RESET_MARKERS = (
    "responsibilities",
    "what you will do",
    "duties",
    "about the role",
    "benefits",
)


def configure_tesseract() -> None:
    default_windows_path = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    if default_windows_path.exists():
        pytesseract.pytesseract.tesseract_cmd = str(default_windows_path)


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")


def clean_text(text: str) -> str:
    if not text:
        return ""

    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s\-\+\#\./@]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def extract_keywords(text: str, top_n: int = 15) -> list[str]:
    if not text:
        return []

    kw_extractor = yake.KeywordExtractor(
        lan="en",
        n=3,
        dedupLim=0.9,
        top=top_n,
    )

    keywords = kw_extractor.extract_keywords(text)
    return [keyword for keyword, score in keywords]


def _alias_pattern(alias: str) -> str:
    """
    Safer than \\b for skills like c++, c#, node.js.
    """
    return rf"(?<![a-z0-9]){re.escape(alias.lower())}(?![a-z0-9])"


def _text_contains_alias(text: str, aliases: list[str]) -> bool:
    normalized_text = clean_text(text)

    for alias in aliases:
        pattern = _alias_pattern(alias)
        if re.search(pattern, normalized_text):
            return True

    return False


def find_skills_in_text(text: str, skill_map: dict[str, list[str]]) -> list[str]:
    found_skills = []

    if not text:
        return found_skills

    for canonical_skill, aliases in skill_map.items():
        if _text_contains_alias(text, aliases):
            found_skills.append(canonical_skill)

    return sorted(set(found_skills))


def extract_weighted_job_skills(raw_job_text: str, skill_map: dict[str, list[str]]) -> dict:
    """
    Detect job skills and assign weights based on which section they appear in.

    required  -> weight 1.0
    preferred -> weight 0.5
    neutral   -> weight 0.75
    """
    skill_weights: dict[str, float] = {}
    required_skills = set()
    preferred_skills = set()
    neutral_skills = set()

    if not raw_job_text:
        return {
            "job_skills": [],
            "required_skills": [],
            "preferred_skills": [],
            "neutral_skills": [],
            "job_skill_weights": {},
        }

    lines = [line.strip() for line in raw_job_text.splitlines() if line.strip()]
    current_bucket = "neutral"

    for line in lines:
        lowered_line = line.lower()

        if any(marker in lowered_line for marker in REQUIRED_SECTION_MARKERS):
            current_bucket = "required"
            continue

        if any(marker in lowered_line for marker in PREFERRED_SECTION_MARKERS):
            current_bucket = "preferred"
            continue

        if any(marker in lowered_line for marker in SECTION_RESET_MARKERS):
            current_bucket = "neutral"
            continue

        found_in_line = find_skills_in_text(line, skill_map)
        if not found_in_line:
            continue

        if current_bucket == "required":
            weight = 1.0
        elif current_bucket == "preferred":
            weight = 0.5
        else:
            weight = 0.75

        for skill in found_in_line:
            skill_weights[skill] = max(skill_weights.get(skill, 0.0), weight)

            if current_bucket == "required":
                required_skills.add(skill)
            elif current_bucket == "preferred":
                preferred_skills.add(skill)
            else:
                neutral_skills.add(skill)

    # Fallback: if nothing was found line-by-line, scan the whole text
    if not skill_weights:
        fallback_skills = find_skills_in_text(raw_job_text, skill_map)
        for skill in fallback_skills:
            skill_weights[skill] = 1.0
            required_skills.add(skill)

    return {
        "job_skills": sorted(skill_weights.keys()),
        "required_skills": sorted(required_skills),
        "preferred_skills": sorted(preferred_skills),
        "neutral_skills": sorted(neutral_skills),
        "job_skill_weights": skill_weights,
    }


def compare_resume_to_job_weighted(
    resume_text: str,
    raw_job_text: str,
    skill_map: dict[str, list[str]],
) -> dict:
    resume_skills = find_skills_in_text(resume_text, skill_map)
    weighted_job_data = extract_weighted_job_skills(raw_job_text, skill_map)

    job_skills = weighted_job_data["job_skills"]
    required_skills = weighted_job_data["required_skills"]
    preferred_skills = weighted_job_data["preferred_skills"]
    neutral_skills = weighted_job_data["neutral_skills"]
    job_skill_weights = weighted_job_data["job_skill_weights"]

    matched_skills = sorted(set(resume_skills) & set(job_skills))
    missing_skills = sorted(set(job_skills) - set(resume_skills))

    required_missing_skills = sorted(set(required_skills) - set(resume_skills))
    preferred_missing_skills = sorted(set(preferred_skills) - set(resume_skills))

    return {
        "resume_skills": resume_skills,
        "job_skills": job_skills,
        "required_skills": required_skills,
        "preferred_skills": preferred_skills,
        "neutral_skills": neutral_skills,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "required_missing_skills": required_missing_skills,
        "preferred_missing_skills": preferred_missing_skills,
        "job_skill_weights": job_skill_weights,
    }


def calculate_weighted_skill_match_percentage(
    matched_skills: list[str],
    job_skill_weights: dict[str, float],
) -> float:
    if not job_skill_weights:
        return 0.0

    total_weight = sum(job_skill_weights.values())
    matched_weight = sum(job_skill_weights.get(skill, 0.0) for skill in matched_skills)

    score = (matched_weight / total_weight) * 100
    return round(score, 2)


def calculate_semantic_similarity(text_a: str, text_b: str) -> float:
    if not text_a or not text_b:
        return 0.0

    model = get_embedding_model()
    embeddings = model.encode([text_a, text_b], convert_to_tensor=True)

    similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
    normalized_score = max(0.0, min(100.0, similarity * 100))

    return round(normalized_score, 2)


def calculate_combined_match_score(
    skill_match_percentage: float,
    semantic_similarity_percentage: float,
    skill_weight: float = 0.7,
    semantic_weight: float = 0.3,
) -> float:
    combined = (
        skill_match_percentage * skill_weight
        + semantic_similarity_percentage * semantic_weight
    )
    return round(combined, 2)


def analyze_formatting_issues(raw_text: str) -> list[str]:
    issues = []

    if not raw_text or len(raw_text.strip()) < 100:
        issues.append("Resume content appears too short.")

    email_pattern = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    if not re.search(email_pattern, raw_text):
        issues.append("Missing email address.")

    phone_pattern = r"(\+?\d[\d\s\-]{7,}\d)"
    if not re.search(phone_pattern, raw_text):
        issues.append("Missing phone number.")

    lowered = raw_text.lower()

    if "education" not in lowered:
        issues.append("Education section not clearly found.")

    if "experience" not in lowered and "work experience" not in lowered:
        issues.append("Work experience section not clearly found.")

    if "skills" not in lowered:
        issues.append("Skills section not clearly found.")

    if len(raw_text.split()) > 2000:
        issues.append("Resume appears unusually long or poorly extracted.")

    return issues


def get_match_label(job_match_percentage: float) -> str:
    if job_match_percentage >= 75:
        return "Strong Match"
    elif job_match_percentage >= 45:
        return "Moderate Match"
    return "Weak Match"


def generate_recommendations(
    missing_skills: list[str],
    formatting_issues: list[str],
    required_missing_skills: list[str] | None = None,
    preferred_missing_skills: list[str] | None = None,
) -> list[str]:
    required_missing_skills = required_missing_skills or []
    preferred_missing_skills = preferred_missing_skills or []

    recommendations = []

    if required_missing_skills:
        recommendations.append(
            f"Address the missing must-have skills first: {', '.join(required_missing_skills[:5])}."
        )

    if preferred_missing_skills:
        recommendations.append(
            f"If they genuinely fit your background, strengthen these nice-to-have skills: {', '.join(preferred_missing_skills[:5])}."
        )

    if missing_skills and not required_missing_skills and not preferred_missing_skills:
        recommendations.append(
            f"Prioritize these missing skills: {', '.join(missing_skills[:5])}."
        )

    if formatting_issues:
        recommendations.append(
            "Improve resume structure by fixing the identified formatting/content issues."
        )

    if not missing_skills and not formatting_issues:
        recommendations.append(
            "Your resume aligns well with the job description and has no major obvious issues."
        )

    return recommendations


def generate_feedback_summary(
    job_match_percentage: float,
    matched_skills: list[str],
    missing_skills: list[str],
    formatting_issues: list[str],
    required_missing_skills: list[str] | None = None,
) -> str:
    label = get_match_label(job_match_percentage)
    required_missing_skills = required_missing_skills or []

    summary_parts = [
        f"This resume is a {label.lower()} for the target role with a match score of {job_match_percentage}%.",
    ]

    if matched_skills:
        summary_parts.append(
            f"Key matched skills include: {', '.join(matched_skills[:5])}."
        )

    if required_missing_skills:
        summary_parts.append(
            f"The biggest concern is missing must-have skills such as: {', '.join(required_missing_skills[:5])}."
        )
    elif missing_skills:
        summary_parts.append(
            f"The main missing skills are: {', '.join(missing_skills[:5])}."
        )

    if formatting_issues:
        summary_parts.append(
            f"There are {len(formatting_issues)} noticeable formatting or structure issues that should be reviewed."
        )

    return " ".join(summary_parts)


def extract_text_from_pdf_bytes(file_bytes: bytes) -> str:
    pdf_stream = BytesIO(file_bytes)
    reader = PdfReader(pdf_stream)

    extracted_pages = []

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            extracted_pages.append(page_text)

    return "\n".join(extracted_pages).strip()


def extract_text_with_ocr(file_bytes: bytes, dpi: int = 250) -> str:
    configure_tesseract()

    ocr_pages = []
    doc = fitz.open(stream=file_bytes, filetype="pdf")

    for page in doc:
        pix = page.get_pixmap(dpi=dpi)
        img_bytes = pix.tobytes("png")
        image = Image.open(BytesIO(img_bytes))
        page_text = pytesseract.image_to_string(image)

        if page_text:
            ocr_pages.append(page_text)

    doc.close()
    return "\n".join(ocr_pages).strip()


def extract_resume_text(file_bytes: bytes) -> tuple[str, str]:
    normal_text = extract_text_from_pdf_bytes(file_bytes)

    if len(normal_text.strip()) >= 120:
        return normal_text, "embedded_text"

    ocr_text = extract_text_with_ocr(file_bytes)

    if ocr_text:
        return ocr_text, "ocr"

    return normal_text, "embedded_text"