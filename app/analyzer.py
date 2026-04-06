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

RESUME_SECTION_HEADINGS = {
    "summary": [
        "summary",
        "professional summary",
        "profile",
        "about me",
    ],
    "skills": [
        "skills",
        "technical skills",
        "core skills",
        "key skills",
    ],
    "experience": [
        "experience",
        "work experience",
        "professional experience",
        "employment history",
    ],
    "education": [
        "education",
        "academic background",
        "qualifications",
    ],
    "projects": [
        "projects",
        "personal projects",
        "academic projects",
    ],
    "certifications": [
        "certifications",
        "licenses",
    ],
}

SECTION_CONFIDENCE_WEIGHTS = {
    "skills": 1.00,
    "experience": 0.95,
    "projects": 0.90,
    "summary": 0.75,
    "certifications": 0.75,
    "education": 0.60,
    "other": 0.50,
}

ACTION_VERBS = {
    "built", "developed", "designed", "implemented", "created", "led",
    "improved", "optimized", "managed", "delivered", "analyzed",
    "automated", "collaborated", "integrated", "deployed", "tested",
    "maintained", "supported", "launched", "engineered"
}

TECH_ROLE_HINTS = {
    "python", "java", "javascript", "typescript", "sql", "fastapi",
    "django", "flask", "react", "node.js", "aws", "docker", "kubernetes",
    "machine learning", "nlp", "pytorch", "tensorflow", "data analysis"
}


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


def normalize_heading(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


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
    return rf"(?<![a-z0-9]){re.escape(alias.lower())}(?![a-z0-9])"


def _text_contains_alias(text: str, aliases: list[str]) -> bool:
    normalized_text = clean_text(text)

    for alias in aliases:
        pattern = _alias_pattern(alias)
        if re.search(pattern, normalized_text):
            return True

    return False


def detect_resume_heading(line: str) -> str | None:
    normalized = normalize_heading(line)

    for section_name, headings in RESUME_SECTION_HEADINGS.items():
        if normalized in [normalize_heading(h) for h in headings]:
            return section_name

    return None


def split_resume_into_sections(raw_resume_text: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {
        "summary": [],
        "skills": [],
        "experience": [],
        "education": [],
        "projects": [],
        "certifications": [],
        "other": [],
    }

    if not raw_resume_text:
        return sections

    current_section = "other"
    lines = [line.strip() for line in raw_resume_text.splitlines() if line.strip()]

    for line in lines:
        detected = detect_resume_heading(line)
        if detected:
            current_section = detected
            continue

        sections[current_section].append(line)

    return sections


def extract_resume_skill_evidence(
    raw_resume_text: str,
    skill_map: dict[str, list[str]],
) -> dict[str, dict]:
    """
    For each detected skill, keep the best evidence snippet and section.
    """
    sections = split_resume_into_sections(raw_resume_text)
    evidence_map: dict[str, dict] = {}

    for section_name, lines in sections.items():
        section_weight = SECTION_CONFIDENCE_WEIGHTS.get(section_name, 0.50)

        for line in lines:
            for canonical_skill, aliases in skill_map.items():
                if _text_contains_alias(line, aliases):
                    existing = evidence_map.get(canonical_skill)

                    candidate = {
                        "skill": canonical_skill,
                        "section": section_name,
                        "snippet": line,
                        "confidence": section_weight,
                    }

                    if existing is None or candidate["confidence"] > existing["confidence"]:
                        evidence_map[canonical_skill] = candidate

    return evidence_map


def find_skills_in_text(text: str, skill_map: dict[str, list[str]]) -> list[str]:
    found_skills = []

    if not text:
        return found_skills

    for canonical_skill, aliases in skill_map.items():
        if _text_contains_alias(text, aliases):
            found_skills.append(canonical_skill)

    return sorted(set(found_skills))


def extract_weighted_job_skills(raw_job_text: str, skill_map: dict[str, list[str]]) -> dict:
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


def compare_resume_to_job_section_aware(
    raw_resume_text: str,
    resume_text: str,
    raw_job_text: str,
    skill_map: dict[str, list[str]],
) -> dict:
    resume_sections = split_resume_into_sections(raw_resume_text)
    evidence_map = extract_resume_skill_evidence(raw_resume_text, skill_map)

    resume_skills = sorted(evidence_map.keys())
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

    matched_skill_evidence = [evidence_map[skill] for skill in matched_skills if skill in evidence_map]

    return {
        "resume_sections": resume_sections,
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
        "skill_evidence_map": evidence_map,
        "matched_skill_evidence": matched_skill_evidence,
    }


def calculate_section_aware_skill_match_percentage(
    matched_skills: list[str],
    job_skill_weights: dict[str, float],
    skill_evidence_map: dict[str, dict],
) -> float:
    if not job_skill_weights:
        return 0.0

    total_weight = sum(job_skill_weights.values())
    matched_weight = 0.0

    for skill in matched_skills:
        job_weight = job_skill_weights.get(skill, 0.0)
        evidence_confidence = skill_evidence_map.get(skill, {}).get("confidence", 0.5)
        matched_weight += job_weight * evidence_confidence

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
    skill_weight: float = 0.75,
    semantic_weight: float = 0.25,
) -> float:
    combined = (
        skill_match_percentage * skill_weight
        + semantic_similarity_percentage * semantic_weight
    )
    return round(combined, 2)


def analyze_formatting_issues(raw_text: str, detected_resume_skills: list[str] | None = None) -> list[str]:
    issues = []
    detected_resume_skills = detected_resume_skills or []

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

    # Split into lines for more detailed heuristics
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

    # Dense line detection
    long_lines = [line for line in lines if len(line.split()) > 28]
    if len(long_lines) >= 4:
        issues.append("Resume contains several dense text lines; bullet-based formatting may improve readability.")

    # Bullet point detection
    bullet_like_lines = [
        line for line in lines
        if line.startswith(("•", "-", "*")) or re.match(r"^\d+\.", line)
    ]
    if len(bullet_like_lines) < 3:
        issues.append("Resume may lack clear bullet points for achievements and responsibilities.")

    # Action verb detection in likely achievement lines
    achievement_like_lines = [
        line for line in lines
        if len(line.split()) >= 6
    ]
    weak_action_lines = 0
    checked_lines = 0

    for line in achievement_like_lines:
        first_word = re.sub(r"^[•\-\*\d\.\)\(]+\s*", "", line).split(" ")[0].lower()
        if first_word:
            checked_lines += 1
            if first_word not in ACTION_VERBS:
                weak_action_lines += 1

    if checked_lines >= 4 and weak_action_lines / checked_lines > 0.6:
        issues.append("Many experience/project lines do not begin with strong action verbs.")

    # Quantified achievement detection
    quantified_markers = re.findall(r"\b\d+%|\b\d+\+|\$\d+|\b\d+\b", raw_text)
    if len(quantified_markers) < 2:
        issues.append("Resume includes few measurable achievements; adding metrics could strengthen impact.")

    # Technical profile link suggestion for technical resumes
    tech_overlap = set(detected_resume_skills) & TECH_ROLE_HINTS
    has_linkedin = "linkedin" in lowered
    has_github = "github" in lowered

    if tech_overlap and not (has_linkedin or has_github):
        issues.append("Technical resume may benefit from adding a LinkedIn or GitHub profile link.")

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

def generate_top_strengths(
    matched_skills: list[str],
    matched_skill_evidence: list[dict],
    semantic_similarity_percentage: float,
) -> list[str]:
    strengths = []

    if matched_skills:
        strengths.append(
            f"Strong overlap with key skills such as: {', '.join(matched_skills[:4])}."
        )

    if matched_skill_evidence:
        high_confidence = [
            item["skill"]
            for item in matched_skill_evidence
            if item.get("confidence", 0) >= 0.9
        ]
        if high_confidence:
            strengths.append(
                f"Several skills are supported by strong resume evidence in sections like Skills, Experience, or Projects: {', '.join(high_confidence[:4])}."
            )

    if semantic_similarity_percentage >= 70:
        strengths.append(
            "The resume is semantically well aligned with the job description overall."
        )
    elif semantic_similarity_percentage >= 55:
        strengths.append(
            "The resume shows moderate overall semantic relevance to the target role."
        )

    if not strengths:
        strengths.append("Some relevant alignment exists, but the strongest strengths are limited.")

    return strengths


def generate_top_risks(
    required_missing_skills: list[str],
    preferred_missing_skills: list[str],
    formatting_issues: list[str],
) -> list[str]:
    risks = []

    if required_missing_skills:
        risks.append(
            f"Important required skills are missing: {', '.join(required_missing_skills[:4])}."
        )

    if preferred_missing_skills:
        risks.append(
            f"Some preferred skills are not shown: {', '.join(preferred_missing_skills[:4])}."
        )

    if formatting_issues:
        risks.append(
            f"There are {len(formatting_issues)} formatting or structure concerns that may reduce resume quality."
        )

    if not risks:
        risks.append("No major high-priority risks were detected.")

    return risks

def generate_comparison_insights(left_analysis: dict, right_analysis: dict) -> dict:
    """
    Compare two saved analyses and generate high-level insights.
    """
    left_score = float(left_analysis.get("overall_match") or 0)
    right_score = float(right_analysis.get("overall_match") or 0)

    left_required_missing = len(left_analysis.get("required_missing_skills") or [])
    right_required_missing = len(right_analysis.get("required_missing_skills") or [])

    left_formatting_issues = len(left_analysis.get("formatting_issues") or [])
    right_formatting_issues = len(right_analysis.get("formatting_issues") or [])

    left_matched = len(left_analysis.get("matched_skills") or [])
    right_matched = len(right_analysis.get("matched_skills") or [])

    left_advantages = []
    right_advantages = []

    if left_score > right_score:
        left_advantages.append("Higher overall match score.")
    elif right_score > left_score:
        right_advantages.append("Higher overall match score.")

    if left_required_missing < right_required_missing:
        left_advantages.append("Fewer missing must-have skills.")
    elif right_required_missing < left_required_missing:
        right_advantages.append("Fewer missing must-have skills.")

    if left_formatting_issues < right_formatting_issues:
        left_advantages.append("Fewer formatting or structure issues.")
    elif right_formatting_issues < left_formatting_issues:
        right_advantages.append("Fewer formatting or structure issues.")

    if left_matched > right_matched:
        left_advantages.append("More matched skills overall.")
    elif right_matched > left_matched:
        right_advantages.append("More matched skills overall.")

    # Decide winner
    left_points = len(left_advantages)
    right_points = len(right_advantages)

    if left_points > right_points:
        winner_side = "left"
        winner_name = left_analysis.get("filename", "Left Resume")
        loser_name = right_analysis.get("filename", "Right Resume")
    elif right_points > left_points:
        winner_side = "right"
        winner_name = right_analysis.get("filename", "Right Resume")
        loser_name = left_analysis.get("filename", "Left Resume")
    else:
        # Tie-break using overall match
        if left_score > right_score:
            winner_side = "left"
            winner_name = left_analysis.get("filename", "Left Resume")
            loser_name = right_analysis.get("filename", "Right Resume")
        elif right_score > left_score:
            winner_side = "right"
            winner_name = right_analysis.get("filename", "Right Resume")
            loser_name = left_analysis.get("filename", "Left Resume")
        else:
            winner_side = "tie"
            winner_name = "Neither version clearly wins"
            loser_name = ""

    if winner_side == "tie":
        summary = (
            "Both resume versions are closely matched overall. "
            "Neither version has a clear advantage across the main comparison signals."
        )
        improvement_advice = (
            "Focus on reducing missing must-have skills, strengthening evidence for matched skills, "
            "and improving formatting clarity to create a clearer winner."
        )
    else:
        summary = (
            f"{winner_name} appears stronger overall than {loser_name}. "
            f"It performs better across the most important comparison signals."
        )

        weaker_analysis = right_analysis if winner_side == "left" else left_analysis
        weaker_required_missing = weaker_analysis.get("required_missing_skills") or []
        weaker_formatting = weaker_analysis.get("formatting_issues") or []

        advice_parts = []

        if weaker_required_missing:
            advice_parts.append(
                f"Address missing must-have skills first: {', '.join(weaker_required_missing[:4])}."
            )

        if weaker_formatting:
            advice_parts.append(
                "Improve formatting and structure issues to increase readability and professionalism."
            )

        if not advice_parts:
            advice_parts.append(
                "Strengthen project and experience evidence for the most relevant matched skills."
            )

        improvement_advice = " ".join(advice_parts)

    return {
        "winner_side": winner_side,
        "winner_name": winner_name,
        "summary": summary,
        "left_advantages": left_advantages,
        "right_advantages": right_advantages,
        "improvement_advice": improvement_advice,
    }

def extract_resume_text(file_bytes: bytes) -> tuple[str, str]:
    normal_text = extract_text_from_pdf_bytes(file_bytes)

    if len(normal_text.strip()) >= 120:
        return normal_text, "embedded_text"

    ocr_text = extract_text_with_ocr(file_bytes)

    if ocr_text:
        return ocr_text, "ocr"

    return normal_text, "embedded_text"