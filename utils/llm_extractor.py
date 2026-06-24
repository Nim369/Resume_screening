"""
utils/llm_extractor.py
----------------------
LLM-based structured extraction from resumes using GPT-4o-mini
via OpenRouter API (same approach as the NLP notebook).

What it extracts (JSON):
  - skills[]              : list of technical skills
  - experience[]          : [{role, type, duration_years}] — LLM understands
                             "6 month internship" = 0.5 years automatically
  - education[]           : degree names only
  - projects[]            : short project descriptions
  - semantic_summary      : 2-3 sentence professional summary

Why this is better than keyword/regex (your current approach):
  - Regex: looks for "3+ years" pattern → misses "worked at XYZ for 2 years"
  - LLM  : reads the whole sentence and understands context and duration
  - Regex: skill taxonomy → only finds what's in your hardcoded list
  - LLM  : finds any skill regardless of your taxonomy

Cost: ~$0.001 per resume (very cheap via OpenRouter free tier)
Requires: pip install openai
Requires: OpenRouter API key (free at openrouter.ai)
"""

import re
import json
import openai


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_prompt(text: str, is_jd: bool = False) -> str:
    """
    Build the strict extraction prompt used in the NLP notebook.
    Same prompt, adapted to work for both resume and JD extraction.
    """
    role    = "expert resume parser" if not is_jd else "expert job description analyzer"
    subject = "resume" if not is_jd else "job description"
    exp_rule = (
        "Include ONLY professional work experience entries. "
        "Include: Full-time jobs, internships (with role, type, duration_years). "
        "EXCLUDE: Education (BTech, MTech, degrees), academic projects, training courses."
    ) if not is_jd else (
        "Include minimum required years of experience as a single entry."
    )

    return f"""You are an {role}.

Analyze the {subject} text provided below and extract both semantic descriptions and structured data.

STRICT RULES FOR EXTRACTION:
1. "experience": {exp_rule}
   - Only count actual job/internship roles as experience.

2. "skills": Extract technical skills only (programming languages, tools, frameworks, libraries).
   - Normalize names (e.g., ML -> Machine Learning, JS -> JavaScript).

3. "education": Extract degree names only (BTech, MTech, MCA, MBA, PhD, etc.).

4. "projects": Include project titles or short descriptions (1 sentence each).

5. "semantic_summary": Provide a 2-3 sentence professional summary focusing on the candidate's technical profile.

OUTPUT FORMAT (STRICT JSON ONLY — no markdown, no preamble):
{{
  "skills": ["list of strings"],
  "experience": [
    {{
      "role": "title",
      "type": "internship or full-time",
      "duration_years": number
    }}
  ],
  "education": ["degree names"],
  "projects": ["short descriptions"],
  "semantic_summary": "summary text"
}}

{subject.capitalize()} text:
{text[:4000]}
"""


# ── JSON cleaning ─────────────────────────────────────────────────────────────

def _clean_json_response(raw: str) -> str:
    """Remove markdown code fences that LLMs sometimes wrap around JSON."""
    raw = raw.strip()
    raw = re.sub(r'^```json\s*', '', raw, flags=re.MULTILINE)
    raw = re.sub(r'^```\s*',     '', raw, flags=re.MULTILINE)
    raw = re.sub(r'```$',        '', raw, flags=re.MULTILINE)
    return raw.strip()


# ── Validation layer ──────────────────────────────────────────────────────────

def _validate_experience_years(experience_list: list) -> float:
    """
    Sum duration_years from all experience entries.
    Caps at 10 years (sanity check — same as NLP notebook).
    """
    total = 0.0
    for entry in experience_list:
        if isinstance(entry, dict):
            try:
                years = float(entry.get("duration_years", 0))
                if 0 < years < 50:   # sanity check per entry
                    total += years
            except (ValueError, TypeError):
                pass

    if total > 10:
        return 5.0   # cap unrealistic values
    if total < 0:
        return 0.0
    return round(total, 2)


def _validate_struct(raw_struct: dict, resume_text: str, is_jd: bool = False) -> dict:
    """
    Clean and validate the LLM output.
    Returns a normalized dict ready for scoring.
    """
    skills   = [s.strip() for s in raw_struct.get("skills",    []) if isinstance(s, str) and s.strip()]
    edu      = [e.strip() for e in raw_struct.get("education", []) if isinstance(e, str) and e.strip()]
    projects = [p.strip() for p in raw_struct.get("projects",  []) if isinstance(p, str) and p.strip()]
    exp_list = raw_struct.get("experience", [])

    return {
        "skills":            skills,
        "experience_years":  _validate_experience_years(exp_list),
        "experience_breakdown": exp_list,
        "education":         edu,
        "projects":          projects,
        "semantic_summary":  raw_struct.get("semantic_summary", ""),
    }


# ── Main extraction function ──────────────────────────────────────────────────

def extract_via_llm(
    text:     str,
    api_key:  str,
    is_jd:    bool = False,
    base_url: str  = "https://openrouter.ai/api/v1",
    model:    str  = "openai/gpt-4o-mini",
) -> dict:
    """
    Extract structured data from a resume or JD using GPT-4o-mini.

    Args:
        text    : raw or cleaned text of the resume / JD
        api_key : OpenRouter API key (get free at openrouter.ai)
        is_jd   : True if extracting from a Job Description
        base_url: API base URL (default: OpenRouter)
        model   : model to use (default: gpt-4o-mini)

    Returns:
        Validated dict with keys:
          skills, experience_years, experience_breakdown,
          education, projects, semantic_summary
    """
    empty = {
        "skills": [], "experience_years": 0.0,
        "experience_breakdown": [], "education": [],
        "projects": [], "semantic_summary": "",
    }

    if not api_key or not api_key.strip():
        print("[WARNING] No API key provided — skipping LLM extraction.")
        return empty

    try:
        client = openai.OpenAI(base_url=base_url, api_key=api_key.strip())
        prompt = _build_prompt(text, is_jd=is_jd)

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.1,   # low temperature = more consistent JSON output
        )

        raw = response.choices[0].message.content
        raw = _clean_json_response(raw)
        parsed = json.loads(raw)
        return _validate_struct(parsed, text, is_jd=is_jd)

    except json.JSONDecodeError as e:
        print(f"[ERROR] LLM returned invalid JSON: {e}")
        return empty
    except Exception as e:
        print(f"[ERROR] LLM extraction failed: {e}")
        return empty


# ── Batch extraction ──────────────────────────────────────────────────────────

def extract_all_via_llm(
    resumes:  list,
    jd_text:  str,
    api_key:  str,
    progress_callback=None,
) -> tuple:
    """
    Extract structured data from all resumes + JD using LLM.

    Args:
        resumes           : list of resume dicts (from parser.load_all_resumes)
        jd_text           : cleaned JD text
        api_key           : OpenRouter API key
        progress_callback : optional callable(current, total) for progress UI

    Returns:
        Tuple: (resume_structs list, jd_struct dict)
    """
    resume_structs = []
    total = len(resumes)

    for i, resume in enumerate(resumes):
        print(f"  [LLM] Extracting resume {i+1}/{total}: {resume['filename']}")
        struct = extract_via_llm(resume["cleaned_text"], api_key, is_jd=False)
        resume_structs.append(struct)
        if progress_callback:
            progress_callback(i + 1, total)

    print("  [LLM] Extracting JD structure...")
    jd_struct = extract_via_llm(jd_text, api_key, is_jd=True)

    return resume_structs, jd_struct
