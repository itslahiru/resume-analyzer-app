import json
import sqlite3
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "resume_analyzer.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _safe_json_loads(value):
    if not value:
        return []
    try:
        return json.loads(value)
    except Exception:
        return []


def _get_existing_columns(conn, table_name: str) -> set[str]:
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    rows = cursor.fetchall()
    return {row["name"] for row in rows}


def _ensure_column_exists(conn, table_name: str, column_name: str, definition: str):
    existing_columns = _get_existing_columns(conn, table_name)
    if column_name not in existing_columns:
        cursor = conn.cursor()
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}")
        conn.commit()


def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            filename TEXT NOT NULL,
            analyzed_at TEXT NOT NULL,
            extraction_method TEXT,
            match_label TEXT,
            overall_match REAL,
            skill_match REAL,
            semantic_similarity REAL,
            feedback_summary TEXT,
            job_description TEXT,
            matched_skills TEXT,
            missing_skills TEXT,
            required_missing_skills TEXT,
            preferred_missing_skills TEXT,
            formatting_issues TEXT,
            top_strengths TEXT,
            top_risks TEXT,
            recommendations TEXT,
            report_payload TEXT
        )
    """)

    _ensure_column_exists(conn, "analyses", "user_id", "INTEGER")
    _ensure_column_exists(conn, "analyses", "report_payload", "TEXT")

    conn.commit()
    conn.close()


def create_user(username: str, password_hash: str) -> int:
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO users (username, password_hash, created_at)
        VALUES (?, ?, ?)
    """, (
        username,
        password_hash,
        datetime.now().isoformat(timespec="seconds"),
    ))

    user_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return user_id


def get_user_by_username(username: str) -> dict | None:
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT *
        FROM users
        WHERE username = ?
    """, (username,))

    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None

    return {
        "id": row["id"],
        "username": row["username"],
        "password_hash": row["password_hash"],
        "created_at": row["created_at"],
    }


def get_user_by_id(user_id: int) -> dict | None:
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT *
        FROM users
        WHERE id = ?
    """, (user_id,))

    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None

    return {
        "id": row["id"],
        "username": row["username"],
        "password_hash": row["password_hash"],
        "created_at": row["created_at"],
    }


def _row_to_dict(row) -> dict:
    row_dict = {
        "id": row["id"],
        "user_id": row["user_id"] if "user_id" in row.keys() else None,
        "filename": row["filename"],
        "analyzed_at": row["analyzed_at"],
        "extraction_method": row["extraction_method"],
        "match_label": row["match_label"],
        "overall_match": row["overall_match"],
        "skill_match": row["skill_match"],
        "semantic_similarity": row["semantic_similarity"],
        "feedback_summary": row["feedback_summary"],
        "job_description": row["job_description"],
        "matched_skills": _safe_json_loads(row["matched_skills"]),
        "missing_skills": _safe_json_loads(row["missing_skills"]),
        "required_missing_skills": _safe_json_loads(row["required_missing_skills"]),
        "preferred_missing_skills": _safe_json_loads(row["preferred_missing_skills"]),
        "formatting_issues": _safe_json_loads(row["formatting_issues"]),
        "top_strengths": _safe_json_loads(row["top_strengths"]),
        "top_risks": _safe_json_loads(row["top_risks"]),
        "recommendations": _safe_json_loads(row["recommendations"]),
        "report_payload": None,
    }

    if "report_payload" in row.keys():
        try:
            row_dict["report_payload"] = json.loads(row["report_payload"]) if row["report_payload"] else None
        except Exception:
            row_dict["report_payload"] = None

    return row_dict


def _build_fallback_report_payload(analysis: dict) -> dict:
    return {
        "filename": analysis.get("filename"),
        "job_match_percentage": analysis.get("overall_match", 0),
        "match_label": analysis.get("match_label", "N/A"),
        "skill_match_percentage": analysis.get("skill_match", 0),
        "semantic_similarity_percentage": analysis.get("semantic_similarity", 0),
        "feedback_summary": analysis.get("feedback_summary", ""),
        "top_strengths": analysis.get("top_strengths", []),
        "top_risks": analysis.get("top_risks", []),
        "recommendations": analysis.get("recommendations", []),
        "matched_skills": analysis.get("matched_skills", []),
        "missing_skills": analysis.get("missing_skills", []),
        "required_missing_skills": analysis.get("required_missing_skills", []),
        "preferred_missing_skills": analysis.get("preferred_missing_skills", []),
        "formatting_issues": analysis.get("formatting_issues", []),
        "extraction_method": analysis.get("extraction_method", "N/A"),
        "resume_keywords": [],
        "resume_skills": [],
        "required_skills": [],
        "preferred_skills": [],
        "matched_skill_evidence": [],
    }


def save_analysis(results: dict, job_description: str, user_id: int) -> int:
    conn = get_connection()
    cursor = conn.cursor()

    report_payload = json.dumps(results)

    cursor.execute("""
        INSERT INTO analyses (
            user_id,
            filename,
            analyzed_at,
            extraction_method,
            match_label,
            overall_match,
            skill_match,
            semantic_similarity,
            feedback_summary,
            job_description,
            matched_skills,
            missing_skills,
            required_missing_skills,
            preferred_missing_skills,
            formatting_issues,
            top_strengths,
            top_risks,
            recommendations,
            report_payload
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id,
        results.get("filename"),
        datetime.now().isoformat(timespec="seconds"),
        results.get("extraction_method"),
        results.get("match_label"),
        results.get("job_match_percentage"),
        results.get("skill_match_percentage"),
        results.get("semantic_similarity_percentage"),
        results.get("feedback_summary"),
        job_description,
        json.dumps(results.get("matched_skills", [])),
        json.dumps(results.get("missing_skills", [])),
        json.dumps(results.get("required_missing_skills", [])),
        json.dumps(results.get("preferred_missing_skills", [])),
        json.dumps(results.get("formatting_issues", [])),
        json.dumps(results.get("top_strengths", [])),
        json.dumps(results.get("top_risks", [])),
        json.dumps(results.get("recommendations", [])),
        report_payload,
    ))

    analysis_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return analysis_id


def get_all_analyses(user_id: int) -> list[dict]:
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT *
        FROM analyses
        WHERE user_id = ?
        ORDER BY id DESC
    """, (user_id,))

    rows = cursor.fetchall()
    conn.close()

    return [_row_to_dict(row) for row in rows]


def search_analyses(
    user_id: int,
    search_text: str = "",
    match_label: str = "",
    min_score: float | None = None,
) -> list[dict]:
    conn = get_connection()
    cursor = conn.cursor()

    query = "SELECT * FROM analyses WHERE user_id = ?"
    params = [user_id]

    if search_text:
        query += " AND filename LIKE ?"
        params.append(f"%{search_text}%")

    if match_label:
        query += " AND match_label = ?"
        params.append(match_label)

    if min_score is not None:
        query += " AND overall_match >= ?"
        params.append(min_score)

    query += " ORDER BY id DESC"

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    return [_row_to_dict(row) for row in rows]


def get_analysis_by_id(analysis_id: int, user_id: int) -> dict | None:
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT *
        FROM analyses
        WHERE id = ? AND user_id = ?
    """, (analysis_id, user_id))

    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None

    return _row_to_dict(row)


def get_analyses_by_ids(analysis_ids: list[int], user_id: int) -> list[dict]:
    if not analysis_ids:
        return []

    conn = get_connection()
    cursor = conn.cursor()

    placeholders = ",".join("?" for _ in analysis_ids)
    query = f"""
        SELECT *
        FROM analyses
        WHERE user_id = ? AND id IN ({placeholders})
        ORDER BY id DESC
    """

    cursor.execute(query, [user_id, *analysis_ids])
    rows = cursor.fetchall()
    conn.close()

    return [_row_to_dict(row) for row in rows]


def get_report_payload_by_id(analysis_id: int, user_id: int) -> dict | None:
    analysis = get_analysis_by_id(analysis_id, user_id)
    if analysis is None:
        return None

    if analysis.get("report_payload"):
        return analysis["report_payload"]

    return _build_fallback_report_payload(analysis)


def delete_analysis(analysis_id: int, user_id: int) -> None:
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        DELETE FROM analyses
        WHERE id = ? AND user_id = ?
    """, (analysis_id, user_id))

    conn.commit()
    conn.close()