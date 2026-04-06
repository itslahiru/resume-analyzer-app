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
    Table,
    TableStyle,
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


def _section_title(text: str, style):
    return [Spacer(1, 10), Paragraph(text, style), Spacer(1, 4)]


def generate_pdf_report(report_data: dict) -> BytesIO:
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
        fontSize=21,
        leading=25,
        textColor=colors.HexColor("#1d4ed8"),
        spaceAfter=10,
    )

    subtitle_style = ParagraphStyle(
        "CustomSubtitle",
        parent=styles["BodyText"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#6b7280"),
        spaceAfter=10,
    )

    section_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#111827"),
        spaceBefore=8,
        spaceAfter=6,
    )

    body_style = ParagraphStyle(
        "BodyStyle",
        parent=styles["BodyText"],
        fontSize=10.2,
        leading=14,
        textColor=colors.HexColor("#1f2937"),
    )

    small_label_style = ParagraphStyle(
        "SmallLabel",
        parent=styles["BodyText"],
        fontSize=9.5,
        leading=12,
        textColor=colors.HexColor("#374151"),
    )

    story = []

    # Header
    story.append(Paragraph("Resume Analysis Report", title_style))
    story.append(Paragraph(
        "Automated resume-job matching report with skill analysis, semantic scoring, formatting checks, and recommendations.",
        subtitle_style
    ))

    # Summary table
    overview_data = [
        ["File", report_data.get("filename", "Unknown")],
        ["Overall Match", f"{report_data.get('job_match_percentage', 0)}%"],
        ["Match Label", report_data.get("match_label", "N/A")],
        ["Section-Aware Skill Match", f"{report_data.get('skill_match_percentage', 0)}%"],
        ["Semantic Similarity", f"{report_data.get('semantic_similarity_percentage', 0)}%"],
        ["Extraction Method", report_data.get("extraction_method", "N/A")],
    ]

    overview_table = Table(overview_data, colWidths=[160, 320])
    overview_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#111827")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d1d5db")),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eff6ff")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))

    story.extend(_section_title("Overview", section_style))
    story.append(overview_table)

    # Summary paragraph
    story.extend(_section_title("Feedback Summary", section_style))
    story.append(Paragraph(
        report_data.get("feedback_summary", "No summary available."),
        body_style
    ))

    # Strengths
    story.extend(_section_title("Top Strengths", section_style))
    story.extend(_build_bullet_list(report_data.get("top_strengths", []), styles))

    # Risks
    story.extend(_section_title("Top Risks", section_style))
    story.extend(_build_bullet_list(report_data.get("top_risks", []), styles))

    # Recommendations
    story.extend(_section_title("Recommendations", section_style))
    story.extend(_build_bullet_list(report_data.get("recommendations", []), styles))

    # Matched skills
    story.extend(_section_title("Matched Skills", section_style))
    story.extend(_build_bullet_list(report_data.get("matched_skills", []), styles))

    # Missing must-have skills
    story.extend(_section_title("Missing Must-Have Skills", section_style))
    story.extend(_build_bullet_list(report_data.get("required_missing_skills", []), styles))

    # Missing preferred skills
    story.extend(_section_title("Missing Preferred Skills", section_style))
    story.extend(_build_bullet_list(report_data.get("preferred_missing_skills", []), styles))

    # Evidence
    story.extend(_section_title("Matched Skill Evidence", section_style))
    matched_evidence = report_data.get("matched_skill_evidence", [])

    if matched_evidence:
        for item in matched_evidence:
            skill = item.get("skill", "Unknown skill")
            section = item.get("section", "unknown")
            confidence = item.get("confidence", 0)
            snippet = item.get("snippet", "")

            story.append(Paragraph(f"<b>{skill}</b>", body_style))
            story.append(Paragraph(
                f"Section: {section} | Confidence: {confidence}",
                small_label_style
            ))
            story.append(Paragraph(snippet, body_style))
            story.append(Spacer(1, 8))
    else:
        story.append(Paragraph("No evidence snippets available.", body_style))

    # Formatting issues
    story.extend(_section_title("Formatting Issues", section_style))
    story.extend(_build_bullet_list(report_data.get("formatting_issues", []), styles))

    # Resume keywords
    story.extend(_section_title("Resume Keywords", section_style))
    story.extend(_build_bullet_list(report_data.get("resume_keywords", []), styles))

    # Resume skills
    story.extend(_section_title("Detected Resume Skills", section_style))
    story.extend(_build_bullet_list(report_data.get("resume_skills", []), styles))

    # Required skills
    story.extend(_section_title("Required Job Skills", section_style))
    story.extend(_build_bullet_list(report_data.get("required_skills", []), styles))

    # Preferred skills
    story.extend(_section_title("Preferred Job Skills", section_style))
    story.extend(_build_bullet_list(report_data.get("preferred_skills", []), styles))

    doc.build(story)
    buffer.seek(0)
    return buffer