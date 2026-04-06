from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    ListFlowable,
    ListItem,
)


def _build_bullet_list(items: list[str], styles):
    if not items:
        return [Paragraph("None", styles["BodyText"])]

    bullet_items = []
    for item in items:
        bullet_items.append(
            ListItem(Paragraph(str(item), styles["BodyText"]))
        )

    return [ListFlowable(bullet_items, bulletType="bullet")]


def generate_pdf_report(report_data: dict) -> BytesIO:
    """
    Build the resume analysis report as a PDF and return it in memory.
    """
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1d4ed8"),
        spaceAfter=12,
    )

    section_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#111827"),
        spaceBefore=10,
        spaceAfter=6,
    )

    body_style = styles["BodyText"]

    story = []

    # Title
    story.append(Paragraph("Resume Analysis Report", title_style))
    story.append(Spacer(1, 8))

    # Summary block
    story.append(Paragraph("Overview", section_style))
    story.append(Paragraph(
        f"<b>File:</b> {report_data.get('filename', 'Unknown')}",
        body_style
    ))
    story.append(Paragraph(
        f"<b>Overall Match:</b> {report_data.get('job_match_percentage', 0)}%",
        body_style
    ))
    story.append(Paragraph(
        f"<b>Match Label:</b> {report_data.get('match_label', 'N/A')}",
        body_style
    ))
    story.append(Paragraph(
        f"<b>Skill Match:</b> {report_data.get('skill_match_percentage', 0)}%",
        body_style
    ))
    story.append(Paragraph(
        f"<b>Semantic Similarity:</b> {report_data.get('semantic_similarity_percentage', 0)}%",
        body_style
    ))
    story.append(Paragraph(
        f"<b>Extraction Method:</b> {report_data.get('extraction_method', 'N/A')}",
        body_style
    ))
    story.append(Spacer(1, 10))

    # Feedback summary
    story.append(Paragraph("Feedback Summary", section_style))
    story.append(Paragraph(
        report_data.get("feedback_summary", "No summary available."),
        body_style
    ))

    # Recommendations
    story.append(Paragraph("Recommendations", section_style))
    story.extend(_build_bullet_list(report_data.get("recommendations", []), styles))

    # Matched skills
    story.append(Paragraph("Matched Skills", section_style))
    story.extend(_build_bullet_list(report_data.get("matched_skills", []), styles))

    story.append(Paragraph("Missing Must-Have Skills", section_style))
    story.extend(_build_bullet_list(report_data.get("required_missing_skills", []), styles))

    story.append(Paragraph("Missing Preferred Skills", section_style))
    story.extend(_build_bullet_list(report_data.get("preferred_missing_skills", []), styles))

    # Missing skills
    story.append(Paragraph("Missing Skills", section_style))
    story.extend(_build_bullet_list(report_data.get("missing_skills", []), styles))

    # Formatting issues
    story.append(Paragraph("Formatting Issues", section_style))
    story.extend(_build_bullet_list(report_data.get("formatting_issues", []), styles))

    # Resume keywords
    story.append(Paragraph("Resume Keywords", section_style))
    story.extend(_build_bullet_list(report_data.get("resume_keywords", []), styles))

    # Resume skills
    story.append(Paragraph("Detected Resume Skills", section_style))
    story.extend(_build_bullet_list(report_data.get("resume_skills", []), styles))

    story.append(Paragraph("Required Job Skills", section_style))
    story.extend(_build_bullet_list(report_data.get("required_skills", []), styles))

    story.append(Paragraph("Preferred Job Skills", section_style))
    story.extend(_build_bullet_list(report_data.get("preferred_skills", []), styles))

    # Job skills
    story.append(Paragraph("Detected Job Skills", section_style))
    story.extend(_build_bullet_list(report_data.get("job_skills", []), styles))

    doc.build(story)
    buffer.seek(0)
    return buffer