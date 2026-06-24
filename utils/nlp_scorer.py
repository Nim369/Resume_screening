"""
utils/nlp_scorer.py
-------------------
Scoring engine for Method 3: Full NLP Pipeline.

Implements the exact scoring logic from the NLP notebook:

  Weight breakdown (6 signals):
    required_skills  : 35%  — semantic skill coverage (embedding-based)
    reranker         : 25%  — Cross-Encoder score (JD + resume read together)
    experience       : 15%  — LLM-extracted years vs JD requirement
    education        : 10%  — degree level matching
    semantic         : 10%  — BERT cosine similarity
    projects         : 5%   — project relevance to JD

Key improvements over your existing scorer.py:
  1. Skill matching uses BERT embeddings (not just keyword lookup)
     → "PyTorch" can match "deep learning framework" (cosine sim > 0.68)
  2. Experience uses LLM-extracted structured data
     → distinguishes internship vs full-time, understands "6 months"
  3. Cross-Encoder reranker adds a 25% weight that reads both texts together
  4. Top 8 JD skills get 2x weight (they matter most)
"""

import math
import numpy as np


# ── Normalization helpers ─────────────────────────────────────────────────────

def normalize_degree(deg: str) -> str:
    """Normalize degree names to canonical forms for comparison."""
    if not deg:
        return ""
    d = str(deg).lower().strip()
    if any(x in d for x in ["btech", "b.tech", "bachelor", "b.e.", " be "]):
        return "bachelor"
    if any(x in d for x in ["mtech", "m.tech", "master", "m.e.", " me ", "ms"]):
        return "master"
    if any(x in d for x in ["mca"]):
        return "mca"
    if any(x in d for x in ["mba"]):
        return "mba"
    if any(x in d for x in ["phd", "ph.d", "doctorate"]):
        return "doctorate"
    if any(x in d for x in ["b.sc", "bsc"]):
        return "bsc"
    if any(x in d for x in ["m.sc", "msc"]):
        return "msc"
    return d


def normalize_skill(skill: str) -> str:
    """Normalize common skill abbreviations."""
    s = str(skill).lower().strip()
    synonyms = {
        "ml":      "machine learning",
        "ai":      "artificial intelligence",
        "nlp":     "natural language processing",
        "cv":      "computer vision",
        "dl":      "deep learning",
        "react":   "react.js",
        "node":    "node.js",
        "aws":     "cloud platform",
        "gcp":     "cloud platform",
        "azure":   "cloud platform",
        "postgres":"postgresql",
        "mongo":   "mongodb",
        "js":      "javascript",
        "ts":      "typescript",
    }
    return synonyms.get(s, s)


# ── Individual score functions ────────────────────────────────────────────────

def semantic_skill_coverage(
    jd_skills:        list,
    candidate_skills: list,
    embedding_model=None,
) -> tuple:
    """
    Compute skill coverage score with optional embedding-based fuzzy match.

    Logic (from NLP notebook):
      1. Exact match first (fastest)
      2. If no exact match AND embedding_model provided:
         → embed both skill names → cosine similarity
         → threshold 0.68 = match
         → score scales with similarity above floor of 0.35
      3. Top 8 JD skills get 2× weight

    Args:
        jd_skills        : skills required by JD
        candidate_skills : skills found in resume
        embedding_model  : SentenceTransformer model (optional)
                           If None, only exact matching is used.

    Returns:
        Tuple: (score 0-1, matched_skills list, missing_skills list)
    """
    if not jd_skills:
        return 1.0, [], []
    if not candidate_skills:
        return 0.0, [], list(jd_skills)

    req_norm  = [normalize_skill(s) for s in jd_skills]
    cand_norm = [normalize_skill(s) for s in candidate_skills]
    cand_set  = set(cand_norm)

    scores        = []
    matched       = []
    total_weight  = 0.0

    for idx, (orig_skill, norm_req) in enumerate(zip(jd_skills, req_norm)):
        weight       = 2.0 if idx < 8 else 1.0
        total_weight += weight

        # ── Exact match ──
        if norm_req in cand_set:
            scores.append(1.0 * weight)
            matched.append(orig_skill)
            continue

        # ── Embedding-based fuzzy match ──
        if embedding_model is not None:
            try:
                from sentence_transformers import util as st_util
                req_emb  = embedding_model.encode([norm_req],  convert_to_tensor=True)
                cand_emb = embedding_model.encode(cand_norm,   convert_to_tensor=True)
                sims     = st_util.cos_sim(req_emb[0], cand_emb)[0].cpu().numpy()
                best_sim = float(np.max(sims)) if len(sims) else 0.0

                # Scale: below 0.35 = 0, at 1.0 = 1.0
                floor      = 0.35
                scaled_sim = max(0.0, (best_sim - floor) / (1.0 - floor))
                scores.append(scaled_sim * weight)

                if best_sim > 0.68:
                    matched.append(orig_skill)
                continue
            except Exception:
                pass

        # ── No match ──
        scores.append(0.0)

    final   = sum(scores) / total_weight if total_weight > 0 else 0.0
    missing = [s for s in jd_skills if s not in matched]
    return round(float(final), 4), matched, missing


def experience_score_nlp(jd_years: float, resume_years: float) -> float:
    """
    Experience scoring from the NLP notebook.

    - Meets requirement        → 1.0
    - Exceeds by up to 20%    → up to 1.2 (bonus)
    - Below requirement        → proportional
    - JD asks for 0 (fresher) → reward any experience (0.5 to 1.0)
    """
    if jd_years <= 0:
        return float(min(0.5 + (resume_years * 0.5), 1.0))

    ratio = resume_years / jd_years
    if ratio >= 1.0:
        # Meets or exceeds — cap bonus at 1.2
        score = 1.0 + (min(ratio - 1.0, 0.2) * 0.5)
    else:
        score = ratio

    return float(round(min(score, 1.2), 3))


def education_score_nlp(jd_education: list, resume_education: list) -> float:
    """
    Education scoring from the NLP notebook.

    Normalizes both sides and checks for match.
    Returns 1.0 if matched, 0.0 if not. Binary.
    """
    if not jd_education:
        return 1.0   # JD didn't specify — full marks

    jd_norms  = {normalize_degree(e) for e in jd_education}
    res_norms = {normalize_degree(e) for e in resume_education}

    # Direct match
    if jd_norms & res_norms:
        return 1.0

    # Substring fallback (e.g. "bachelor of technology" contains "bachelor")
    for req in jd_norms:
        for res in res_norms:
            if req in res or res in req:
                return 1.0

    return 0.0


def project_score_nlp(jd_text: str, projects: list, embedding_model) -> float:
    """
    Project relevance score from the NLP notebook.

    Encodes all project descriptions and computes their similarity to the JD.
    Score = 0.5×best + 0.3×avg_top3 + 0.2×fraction_above_0.4

    Returns 0.0 if no projects or no embedding model.
    """
    if not projects or embedding_model is None:
        return 0.0

    try:
        from sentence_transformers import util as st_util
        jd_vec   = embedding_model.encode(jd_text,    convert_to_tensor=True)
        proj_vecs= embedding_model.encode(projects,   convert_to_tensor=True)
        sims     = st_util.cos_sim(jd_vec, proj_vecs)[0].cpu().numpy()

        best     = float(np.max(sims))
        top_k    = sorted(sims, reverse=True)[:3]
        avg_top  = float(np.mean(top_k))
        relevant = [s for s in sims if s > 0.4]
        rel_ratio= len(relevant) / len(sims)

        score = 0.5 * best + 0.3 * avg_top + 0.2 * rel_ratio
        return round(float(score), 4)
    except Exception:
        return 0.0


# ── Main scoring function ─────────────────────────────────────────────────────

# Fixed weights (from score_candidate() in NLP notebook)
NLP_WEIGHTS = {
    "required_skills": 0.35,
    "reranker":        0.25,
    "experience":      0.15,
    "education":       0.10,
    "semantic":        0.10,
    "projects":        0.05,
}


def score_candidate_nlp(
    jd_text:          str,
    jd_struct:        dict,
    resume_struct:    dict,
    resume_text:      str,
    sim_score:        float,
    rerank_score:     float,
    embedding_model=  None,
) -> dict:
    """
    Compute the full NLP pipeline score for one candidate.

    Args:
        jd_text         : cleaned JD text
        jd_struct       : LLM-extracted JD structure
        resume_struct   : LLM-extracted resume structure
        resume_text     : cleaned resume text
        sim_score       : BERT cosine similarity (0-1) from bi-encoder
        rerank_score    : Cross-Encoder score (0-1)
        embedding_model : SentenceTransformer for skill fuzzy match + project score

    Returns:
        Dict with final_score, grade, all sub-scores, skill gap, and experience detail.
    """
    jd_skills   = jd_struct.get("skills",            [])
    jd_edu      = jd_struct.get("education",          [])
    jd_years    = jd_struct.get("experience_years",   0.0)
    res_skills  = resume_struct.get("skills",         [])
    res_edu     = resume_struct.get("education",      [])
    res_years   = resume_struct.get("experience_years", 0.0)
    res_projects= resume_struct.get("projects",       [])

    # ── Individual scores ──
    skill_score, matched, missing = semantic_skill_coverage(
        jd_skills, res_skills, embedding_model
    )
    exp_score  = experience_score_nlp(jd_years,  res_years)
    edu_score  = education_score_nlp(jd_edu,     res_edu)
    proj_score = project_score_nlp(jd_text,      res_projects, embedding_model)

    # All scores normalized to 0-100 for display
    skill_100  = round(skill_score  * 100, 1)
    exp_100    = round(min(exp_score, 1.0) * 100, 1)
    edu_100    = round(edu_score    * 100, 1)
    proj_100   = round(proj_score   * 100, 1)
    sim_100    = round(sim_score    * 100, 1)
    rerank_100 = round(rerank_score * 100, 1)

    # ── Weighted final score ──
    w = NLP_WEIGHTS
    final = (
        skill_score   * w["required_skills"] +
        rerank_score  * w["reranker"]        +
        exp_score     * w["experience"]      +
        edu_score     * w["education"]       +
        sim_score     * w["semantic"]        +
        proj_score    * w["projects"]
    ) * 100   # convert to 0-100 scale

    final = round(min(float(final), 100.0), 1)

    # ── Grade ──
    if final >= 75:
        grade, grade_color = "Excellent", "#059669"
    elif final >= 55:
        grade, grade_color = "Good",      "#d97706"
    elif final >= 35:
        grade, grade_color = "Fair",      "#dc2626"
    else:
        grade, grade_color = "Weak",      "#9ca3af"

    # ── Extra skills (in resume but not in JD) ──
    jd_set   = {normalize_skill(s) for s in jd_skills}
    res_set  = {normalize_skill(s) for s in res_skills}
    extra    = sorted([s for s in res_skills if normalize_skill(s) not in jd_set])

    return {
        "final_score": final,
        "grade":       grade,
        "grade_color": grade_color,
        "method":      "NLP Pipeline",

        # Per-component scores (0-100 for display)
        "scores": {
            "similarity":  sim_100,
            "skills":      skill_100,
            "experience":  exp_100,
            "education":   edu_100,
            "reranker":    rerank_100,
            "projects":    proj_100,
        },

        # Weighted contributions to final score
        "contributions": {
            "similarity":  round(sim_100    * w["semantic"],         1),
            "skills":      round(skill_100  * w["required_skills"],  1),
            "experience":  round(exp_100    * w["experience"],       1),
            "education":   round(edu_100    * w["education"],        1),
            "reranker":    round(rerank_100 * w["reranker"],         1),
            "projects":    round(proj_100   * w["projects"],         1),
        },

        # Skill gap detail
        "skill_gap": {
            "matched":   sorted(matched),
            "missing":   sorted(missing),
            "extra":     extra[:10],
            "match_pct": round(len(matched) / len(jd_skills) * 100, 1) if jd_skills else 0.0,
        },

        # Experience detail
        "experience": {
            "jd_required":  jd_years,
            "resume_years": res_years,
            "status":       (
                "exceeds" if res_years > jd_years
                else "meets" if res_years == jd_years
                else "below"
            ),
        },

        # Education detail
        "education": {
            "resume_level": ", ".join(res_edu) if res_edu else "not mentioned",
            "jd_level":     ", ".join(jd_edu)  if jd_edu  else "not specified",
        },

        "weights": NLP_WEIGHTS,
    }


def score_all_nlp(
    resumes:          list,
    resume_structs:   list,
    jd_text:          str,
    jd_struct:        dict,
    sim_scores:       list,
    rerank_scores:    list,
    embedding_model=  None,
    top_n:            int = 5,
) -> list:
    """
    Score all resumes using the NLP pipeline and return ranked results.

    Args:
        resumes         : list of resume dicts (filename, cleaned_text)
        resume_structs  : list of LLM-extracted structures per resume
        jd_text         : cleaned JD text
        jd_struct       : LLM-extracted JD structure
        sim_scores      : BERT cosine similarity scores (one per resume)
        rerank_scores   : Cross-Encoder scores (one per resume)
        embedding_model : SentenceTransformer for skill fuzzy match
        top_n           : number of top results to flag

    Returns:
        List of result dicts sorted by final_score descending.
        Each dict has 'filename', 'rank', 'is_top' added.
    """
    results = []

    for resume, struct, sim, rerank in zip(
        resumes, resume_structs, sim_scores, rerank_scores
    ):
        score_data = score_candidate_nlp(
            jd_text        = jd_text,
            jd_struct      = jd_struct,
            resume_struct  = struct,
            resume_text    = resume["cleaned_text"],
            sim_score      = float(sim),
            rerank_score   = float(rerank),
            embedding_model= embedding_model,
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
