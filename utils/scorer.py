"""
utils/scorer.py
---------------
Weighted scoring engine that combines multiple signals into one final score.

Weights:
  - TF-IDF / BERT similarity   : 70%
  - Skill match                : 10%
  - Experience match           : 10%
  - Education match            : 10%
"""

from skill_extractor import (
    extract_skills,
    compute_skill_gap,
    compute_experience_score,
    compute_education_score,
)

# ── Scoring weights (must sum to 1.0) ────────────────────────────────────────
WEIGHTS = {
    "similarity":  0.70,   # TF-IDF cosine or BERT cosine similarity
    "skills":      0.10,   # Skill taxonomy match %
    "experience":  0.10,   # Years of experience
    "education":   0.10,   # Education level
}


def compute_full_score(
    similarity_score: float,
    jd_text: str,
    resume_text: str,
) -> dict:
    """
    Compute a full weighted score for one resume against a JD.

    Args:
        similarity_score: raw cosine similarity (0.0 – 1.0) from TF-IDF or BERT
        jd_text:          cleaned JD text
        resume_text:      cleaned resume text

    Returns a dict with:
        - final_score       (0–100, weighted)
        - component scores  (similarity, skills, experience, education)
        - skill_gap         (matched / missing / extra skills)
        - breakdown         (human-readable per-component detail)
    """

    # ── 1. Similarity score (already 0–1, convert to 0–100) ──
    sim_score_100 = min(round(similarity_score * 100, 1), 100.0)

    # ── 2. Skill gap analysis ──
    jd_skills     = extract_skills(jd_text)
    resume_skills = extract_skills(resume_text)
    skill_gap     = compute_skill_gap(jd_skills, resume_skills)
    skill_score   = skill_gap["match_pct"]  # already 0–100

    # ── 3. Experience score ──
    exp_result  = compute_experience_score(jd_text, resume_text)
    exp_score   = exp_result["score"]

    # ── 4. Education score ──
    edu_result  = compute_education_score(resume_text, jd_text)
    edu_score   = edu_result["score"]

    # ── 5. Weighted final score ──
    final = (
        sim_score_100 * WEIGHTS["similarity"] +
        skill_score   * WEIGHTS["skills"] +
        exp_score     * WEIGHTS["experience"] +
        edu_score     * WEIGHTS["education"]
    )
    final = round(final, 1)

    # ── 6. Grade label ──
    if final >= 75:
        grade, grade_color = "Excellent", "#1D9E75"
    elif final >= 55:
        grade, grade_color = "Good",      "#BA7517"
    elif final >= 35:
        grade, grade_color = "Fair",      "#E24B4A"
    else:
        grade, grade_color = "Weak",      "#888780"

    return {
        "final_score": final,
        "grade":       grade,
        "grade_color": grade_color,

        # Per-component raw scores (0–100)
        "scores": {
            "similarity":  sim_score_100,
            "skills":      skill_score,
            "experience":  exp_score,
            "education":   edu_score,
        },

        # Weighted contributions to final score
        "contributions": {
            "similarity":  round(sim_score_100 * WEIGHTS["similarity"], 1),
            "skills":      round(skill_score   * WEIGHTS["skills"],      1),
            "experience":  round(exp_score     * WEIGHTS["experience"],  1),
            "education":   round(edu_score     * WEIGHTS["education"],   1),
        },

        # Skill gap detail
        "skill_gap": {
            "matched":   sorted(skill_gap["matched"]),
            "missing":   sorted(skill_gap["missing"]),
            "extra":     sorted(skill_gap["extra"]),
            "match_pct": skill_gap["match_pct"],
        },

        # Experience detail
        "experience": {
            "jd_required":  exp_result["jd_required"],
            "resume_years": exp_result["resume_years"],
            "status":       exp_result["status"],
        },

        # Education detail
        "education": {
            "resume_level": edu_result["resume_level"],
            "jd_level":     edu_result["jd_level"],
        },

        # Weights used (for display)
        "weights": WEIGHTS,
    }


def score_all_resumes(
    resumes:          list,
    similarity_scores: list,
    jd_text:          str,
    top_n:            int = 5,
) -> list:
    """
    Score all resumes and return sorted list of result dicts.

    Args:
        resumes:           list of resume dicts (filename, cleaned_text, raw_text)
        similarity_scores: list of cosine similarity floats (same order as resumes)
        jd_text:           cleaned JD text
        top_n:             number of top results to flag

    Returns:
        List of result dicts sorted by final_score descending.
        Each dict includes all fields from compute_full_score plus
        'filename', 'rank', and 'is_top'.
    """
    results = []

    for i, (resume, sim_score) in enumerate(zip(resumes, similarity_scores)):
        score_data = compute_full_score(
            similarity_score=float(sim_score),
            jd_text=jd_text,
            resume_text=resume["cleaned_text"],
        )
        score_data["filename"] = resume["filename"]
        results.append(score_data)

    # Sort by final score descending
    results.sort(key=lambda x: x["final_score"], reverse=True)

    # Assign rank and top flag
    for i, r in enumerate(results):
        r["rank"]   = i + 1
        r["is_top"] = i < top_n

    return results
