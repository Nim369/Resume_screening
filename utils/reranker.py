"""
utils/reranker.py
-----------------
Cross-Encoder reranker from the NLP notebook approach.

Model  : cross-encoder/ms-marco-MiniLM-L-6-v2
Size   : ~91 MB (downloaded once, cached automatically)
Cost   : Free — no API key needed
Purpose: Re-scores the top candidates by reading JD + resume TOGETHER
         as a pair — much more accurate than cosine similarity alone.

How it differs from BERT cosine similarity:
  - BERT (bi-encoder): encodes JD and resume SEPARATELY, compares vectors
  - Cross-Encoder    : reads BOTH texts at the same time as one input
                       → understands their relationship directly
                       → significantly more accurate for ranking
"""

import math
from sentence_transformers import CrossEncoder

RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def load_reranker() -> CrossEncoder:
    """
    Load and return the Cross-Encoder reranker model.
    Called once and cached in Streamlit via @st.cache_resource.

    Returns:
        CrossEncoder model ready for prediction.
    """
    print(f"[INFO] Loading reranker model: {RERANKER_MODEL_NAME}")
    model = CrossEncoder(RERANKER_MODEL_NAME)
    print("[INFO] Reranker model loaded.")
    return model


def rerank_score(model: CrossEncoder, jd_text: str, resume_text: str) -> float:
    """
    Compute a reranker relevance score for one (JD, resume) pair.

    The raw logit from the model is passed through a sigmoid with
    temperature 1.5 (same as the NLP notebook) to produce a 0–1 score.

    Args:
        model       : loaded CrossEncoder model
        jd_text     : cleaned job description text
        resume_text : cleaned resume text

    Returns:
        Float between 0.0 and 1.0.
        Higher = more relevant match.
    """
    raw_score = model.predict([jd_text, resume_text])
    # Sigmoid with temperature 1.5 — spreads the scores more than raw sigmoid
    # temp < 1.0 → squashes scores toward 0.5
    # temp > 1.0 → spreads scores (more differentiation between candidates)
    return float(1 / (1 + math.exp(-raw_score / 1.5)))


def rerank_all(
    model: CrossEncoder,
    jd_text: str,
    resume_texts: list,
    top_n: int = 20,
) -> list:
    """
    Rerank the top_n candidates using the Cross-Encoder.

    For efficiency, only rerank the top candidates (not all 200).
    The bi-encoder (BERT cosine) pre-selects the top_n, then the
    Cross-Encoder re-scores them with higher accuracy.

    Args:
        model        : loaded CrossEncoder model
        jd_text      : cleaned job description text
        resume_texts : list of cleaned resume texts (all resumes)
        top_n        : how many to rerank (default: rerank top 20 only)

    Returns:
        List of floats — reranker score for EACH resume (same order as input).
        Resumes outside top_n get score 0.0.
    """
    scores = [0.0] * len(resume_texts)
    limit  = min(top_n, len(resume_texts))

    for i in range(limit):
        scores[i] = rerank_score(model, jd_text, resume_texts[i])

    return scores
