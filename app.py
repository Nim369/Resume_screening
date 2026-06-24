"""
app/app.py
----------
ResumeIQ — Enhanced Resume Screening System

Methods available:
  1. TF-IDF        : fast, keyword-based
  2. BERT          : accurate, semantic embeddings
  3. NLP Pipeline  : LLM extraction + BERT + Cross-Encoder reranker (NEW)

Changes from previous version:
  ─────────────────────────────────────────────────────────────────────────────
  [NEW] Imports: reranker, llm_extractor, nlp_scorer added
  [NEW] Sidebar: third method radio option + API key input
  [NEW] load_reranker() cached with @st.cache_resource
  [NEW] run_nlp_pipeline() function — full 4-step NLP pipeline
  [NEW] make_comparison_bar_nlp() — 6-component stacked bar for NLP results
  [NEW] make_radar_chart_nlp() — 6-axis radar for NLP results
  [NEW] Results section: NLP method renders reranker + projects scores too
  [CHANGED] Method selection logic updated to handle 3 methods
  ─────────────────────────────────────────────────────────────────────────────
"""

import sys
import os
import tempfile
import shutil

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── Path setup ──────────────────────────────────────────────────────────────
if os.path.exists(os.path.join(os.path.dirname(__file__), 'utils')):
    UTILS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils'))
else:
    UTILS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils'))
sys.path.insert(0, UTILS_DIR)

from parser          import load_all_resumes, load_job_description
from skill_extractor import extract_skills
from scorer          import score_all_resumes

# ── NEW IMPORTS ──────────────────────────────────────────────────────────────
from reranker        import load_reranker, rerank_all
from llm_extractor   import extract_all_via_llm
from nlp_scorer      import score_all_nlp

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ResumeIQ — Smart Resume Screener",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { padding-top: 1rem; }
    .block-container { padding-top: 1rem; padding-bottom: 2rem; }
    .pill { display:inline-block;padding:3px 10px;border-radius:12px;font-size:12px;margin:2px 3px 2px 0;font-weight:500; }
    .pill-green { background:#d1fae5;color:#065f46; }
    .pill-red   { background:#fee2e2;color:#991b1b; }
    .pill-blue  { background:#dbeafe;color:#1e40af; }
    .pill-gray  { background:#f1f5f9;color:#475569; }
    .pill-purp  { background:#ede9fe;color:#4c1d95; }
    #MainMenu { visibility:hidden; }
    footer    { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════

def get_medal(rank):
    return {1:"🥇",2:"🥈",3:"🥉"}.get(rank, f"#{rank}")

def grade_color(grade):
    return {"Excellent":"#059669","Good":"#d97706","Fair":"#dc2626","Weak":"#9ca3af"}.get(grade,"#6b7280")

def render_skill_pills(skills, pill_class):
    if not skills:
        return "<span style='color:#9ca3af;font-size:12px'>None</span>"
    return " ".join(f'<span class="pill {pill_class}">{s}</span>' for s in skills)

def score_bar_html(label, score, weight, color):
    contrib = round(score * weight, 1)
    return f"""
    <div style='margin-bottom:10px'>
      <div style='display:flex;justify-content:space-between;font-size:12px;color:#6b7280;margin-bottom:3px'>
        <span>{label}</span>
        <span>{score:.0f}/100 &nbsp;·&nbsp; contributes <b>{contrib:.1f}pts</b></span>
      </div>
      <div style='background:#f1f5f9;border-radius:6px;height:10px;overflow:hidden'>
        <div style='background:{color};width:{score}%;height:100%;border-radius:6px'></div>
      </div>
    </div>"""


# ── Radar chart (4-axis — for TF-IDF and BERT methods) ───────────────────────
def make_radar_chart(result):
    categories  = ["Similarity", "Skills", "Experience", "Education"]
    scores      = [result["scores"][k] for k in ["similarity","skills","experience","education"]]
    N = len(categories)
    angles      = [n / float(N) * 2 * 3.14159 for n in range(N)] + [0]
    scores_plot = [s / 100 for s in scores] + [scores[0] / 100]
    fig, ax = plt.subplots(figsize=(3,3), subplot_kw=dict(polar=True))
    fig.patch.set_alpha(0); ax.set_facecolor("none")
    ax.plot(angles, scores_plot, color="#6366f1", linewidth=1.5)
    ax.fill(angles, scores_plot, alpha=0.2, color="#6366f1")
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(categories, size=7, color="#374151")
    ax.set_yticks([0.25,0.5,0.75,1.0]); ax.set_yticklabels(["25","50","75","100"], size=6, color="#9ca3af")
    ax.grid(color="#e5e7eb", linewidth=0.5); ax.spines["polar"].set_visible(False)
    plt.tight_layout()
    return fig


# ── [NEW] Radar chart (6-axis — for NLP Pipeline method) ─────────────────────
def make_radar_chart_nlp(result):
    categories  = ["Skills","Reranker","Experience","Education","Semantic","Projects"]
    score_keys  = ["skills","reranker","experience","education","similarity","projects"]
    scores      = [result["scores"].get(k, 0) for k in score_keys]
    N = len(categories)
    angles      = [n / float(N) * 2 * 3.14159 for n in range(N)] + [0]
    scores_plot = [s / 100 for s in scores] + [scores[0] / 100]
    fig, ax = plt.subplots(figsize=(3,3), subplot_kw=dict(polar=True))
    fig.patch.set_alpha(0); ax.set_facecolor("none")
    ax.plot(angles, scores_plot, color="#7c3aed", linewidth=1.5)
    ax.fill(angles, scores_plot, alpha=0.2, color="#7c3aed")
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(categories, size=6, color="#374151")
    ax.set_yticks([0.25,0.5,0.75,1.0]); ax.set_yticklabels(["25","50","75","100"], size=6, color="#9ca3af")
    ax.grid(color="#e5e7eb", linewidth=0.5); ax.spines["polar"].set_visible(False)
    plt.tight_layout()
    return fig


# ── Stacked bar chart (4-component — TF-IDF / BERT) ─────────────────────────
def make_comparison_bar(results, top_n):
    top     = results[:top_n]
    labels  = [r["filename"][:28]+("…" if len(r["filename"])>28 else "") for r in top]
    y       = range(len(top))
    fig, ax = plt.subplots(figsize=(9, max(3, top_n*0.8)))
    fig.patch.set_alpha(0); ax.set_facecolor("none")
    sv = [r["contributions"]["similarity"] for r in top]
    sk = [r["contributions"]["skills"]     for r in top]
    ex = [r["contributions"]["experience"] for r in top]
    ed = [r["contributions"]["education"]  for r in top]
    ax.barh(y, sv, color="#6366f1", label="Similarity (40%)", height=0.55)
    ax.barh(y, sk, left=sv,                      color="#10b981", label="Skills (30%)",     height=0.55)
    ax.barh(y, ex, left=[a+b for a,b in zip(sv,sk)],       color="#f59e0b", label="Experience (20%)", height=0.55)
    ax.barh(y, ed, left=[a+b+c for a,b,c in zip(sv,sk,ex)],color="#3b82f6", label="Education (10%)", height=0.55)
    ax.set_yticks(list(y)); ax.set_yticklabels([f"{get_medal(r['rank'])} {l}" for r,l in zip(top,labels)], fontsize=8)
    ax.set_xlabel("Final Score (points)", fontsize=8, color="#6b7280"); ax.set_xlim(0,100)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.legend(loc="lower right", fontsize=7, framealpha=0); ax.invert_yaxis()
    plt.tight_layout()
    return fig


# ── [NEW] Stacked bar chart (6-component — NLP Pipeline) ─────────────────────
def make_comparison_bar_nlp(results, top_n):
    top     = results[:top_n]
    labels  = [r["filename"][:28]+("…" if len(r["filename"])>28 else "") for r in top]
    y       = range(len(top))
    fig, ax = plt.subplots(figsize=(9, max(3, top_n*0.8)))
    fig.patch.set_alpha(0); ax.set_facecolor("none")
    c = results[0]["contributions"]

    def vals(key): return [r["contributions"].get(key, 0) for r in top]

    sk = vals("skills"); rr = vals("reranker"); ex = vals("experience")
    ed = vals("education"); sm = vals("similarity"); pr = vals("projects")

    ax.barh(y, sk, color="#7c3aed", label="Skills (35%)",     height=0.55)
    ax.barh(y, rr, left=sk,                          color="#059669", label="Reranker (25%)",  height=0.55)
    ax.barh(y, ex, left=[a+b for a,b in zip(sk,rr)],           color="#f59e0b", label="Experience (15%)",height=0.55)
    ax.barh(y, ed, left=[a+b+c for a,b,c in zip(sk,rr,ex)],    color="#3b82f6", label="Education (10%)", height=0.55)
    ax.barh(y, sm, left=[a+b+c+d for a,b,c,d in zip(sk,rr,ex,ed)],   color="#d85a30", label="Semantic (10%)",  height=0.55)
    ax.barh(y, pr, left=[a+b+c+d+e for a,b,c,d,e in zip(sk,rr,ex,ed,sm)],color="#888780", label="Projects (5%)",   height=0.55)

    ax.set_yticks(list(y)); ax.set_yticklabels([f"{get_medal(r['rank'])} {l}" for r,l in zip(top,labels)], fontsize=8)
    ax.set_xlabel("Final Score (points)", fontsize=8, color="#6b7280"); ax.set_xlim(0,100)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.legend(loc="lower right", fontsize=6, framealpha=0); ax.invert_yaxis()
    plt.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════════════════
# SCORING ENGINES
# ════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def run_tfidf(resume_texts, jd_text):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    vectorizer   = TfidfVectorizer(ngram_range=(1,2), min_df=1, max_df=0.85,
                                   sublinear_tf=True, stop_words="english")
    all_texts    = [jd_text] + list(resume_texts)
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    return cos_sim(tfidf_matrix[0], tfidf_matrix[1:])[0]


@st.cache_resource(show_spinner=False)
def load_bert_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("BAAI/bge-base-en-v1.5")


def run_bert(resume_texts, jd_text):
    import faiss
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    model          = load_bert_model()
    res_embeddings = model.encode(list(resume_texts), batch_size=32,
                                  show_progress_bar=False, convert_to_numpy=True)
    jd_embedding   = model.encode([jd_text], convert_to_numpy=True)
    res_norm = res_embeddings.astype("float32"); faiss.normalize_L2(res_norm)
    jd_norm  = jd_embedding.astype("float32");  faiss.normalize_L2(jd_norm)
    return cos_sim(jd_norm, res_norm)[0]


# ── [NEW] Load Cross-Encoder reranker (cached — loaded only once) ─────────────
@st.cache_resource(show_spinner=False)
def load_reranker_cached():
    return load_reranker()


# ── [NEW] Full NLP Pipeline runner ────────────────────────────────────────────
def run_nlp_pipeline(resumes, resume_texts, jd_text, api_key, status, progress):
    """
    4-step pipeline from the NLP notebook:
      Step 1: BERT similarity (bi-encoder)
      Step 2: LLM extraction (GPT-4o-mini via OpenRouter)
      Step 3: Cross-Encoder reranking
      Step 4: Combined weighted scoring (6 signals)
    """
    # Step 1 — BERT similarity
    status.caption("🤖 Step 1/4 — Computing BERT similarity...")
    progress.progress(35)
    sim_scores = run_bert(resume_texts, jd_text)

    # Step 2 — LLM extraction
    status.caption("🧠 Step 2/4 — LLM extracting skills + experience from resumes...")
    status.caption("   (this takes ~5s per resume — please wait)")
    progress.progress(50)
    resume_structs, jd_struct = extract_all_via_llm(resumes, jd_text, api_key)

    # Step 3 — Cross-Encoder reranking (top 20 only for speed)
    status.caption("⚡ Step 3/4 — Cross-Encoder reranking top candidates...")
    progress.progress(70)
    reranker_model = load_reranker_cached()
    rerank_scores  = rerank_all(reranker_model, jd_text, list(resume_texts), top_n=20)

    # Step 4 — Final scoring
    status.caption("🧮 Step 4/4 — Computing final weighted scores...")
    progress.progress(85)
    embedding_model = load_bert_model()

    return resume_structs, jd_struct, sim_scores, rerank_scores, embedding_model


# ════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ════════════════════════════════════════════════════════════════════════════

def main():
    st.markdown("## 📄 ResumeIQ — Smart Resume Screener")
    st.markdown("Upload resumes and a job description to find the best candidates.")
    st.divider()

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")

        # [CHANGED] — now 3 options instead of 2
        method = st.radio(
            "Matching Method",
            [
                "TF-IDF (fast, keyword-based)",
                "BERT (accurate, semantic)",
                "NLP Pipeline (LLM + Reranker)",   # NEW
            ],
            help=(
                "TF-IDF: fastest. "
                "BERT: understands meaning. "
                "NLP Pipeline: most accurate — uses GPT-4o-mini extraction + Cross-Encoder reranking."
            )
        )
        use_bert = method.startswith("BERT")
        use_nlp  = method.startswith("NLP")    # NEW

        top_n = st.slider("Top N candidates", min_value=3, max_value=10, value=5)

        # [NEW] — API key input shown only when NLP Pipeline is selected
        api_key = ""
        if use_nlp:
            st.divider()
            st.markdown("### 🔑 OpenRouter API Key")
            api_key = st.text_input(
                "Paste your API key",
                type="password",
                placeholder="sk-or-v1-...",
                help="Get a free key at openrouter.ai — needed for GPT-4o-mini extraction"
            )
            if not api_key:
                st.warning("⚠️ API key required for NLP Pipeline method.")
            else:
                st.success("✅ API key provided")

        st.divider()
        st.markdown("### 📊 Score Weights")
        if use_nlp:
            # [NEW] — Show NLP weights when NLP method is selected
            st.markdown("""
            | Component | Weight |
            |---|---|
            | Skill match | 35% |
            | Reranker | 25% |
            | Experience | 15% |
            | Education | 10% |
            | Semantic | 10% |
            | Projects | 5% |
            """)
        else:
            st.markdown("""
            | Component | Weight |
            |---|---|
            | Similarity | 70% |
            | Skill match | 10% |
            | Experience | 10% |
            | Education | 10% |
            """)

        st.divider()
        st.caption("ResumeIQ v3.0 · PDF & DOCX · Streamlit")

    # ── Main inputs ──────────────────────────────────────────────────────────
    col_jd, col_resume = st.columns([1, 1], gap="large")
    with col_jd:
        st.markdown("#### 📝 Job Description")
        jd_input = st.text_area(
            "Paste your JD here", height=280, label_visibility="collapsed",
            placeholder="We are looking for a Python Developer with 3+ years experience in ML, FastAPI, and SQL..."
        )
    with col_resume:
        st.markdown("#### 📁 Upload Resumes")
        uploaded_files = st.file_uploader(
            "Drop PDF or DOCX files here", type=["pdf","docx"],
            accept_multiple_files=True, label_visibility="collapsed"
        )
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) uploaded")
            with st.expander("View files"):
                for f in uploaded_files:
                    st.caption(f"📄 {f.name} ({round(f.size/1024,1)} KB)")

    st.divider()

    # [CHANGED] — disabled if NLP method and no API key
    can_run = bool(jd_input.strip()) and bool(uploaded_files)
    if use_nlp and not api_key:
        can_run = False

    run_btn = st.button("🚀  Screen Resumes", type="primary",
                        disabled=not can_run, use_container_width=True)

    if not can_run and not run_btn:
        if use_nlp and not api_key:
            st.info("👆 Paste a Job Description, upload resumes, and enter your OpenRouter API key.")
        else:
            st.info("👆 Paste a Job Description and upload at least one resume to get started.")
        return

    if not run_btn:
        return

    # ── Processing ───────────────────────────────────────────────────────────
    progress = st.progress(0, "Starting...")
    status   = st.empty()
    tmp_dir  = tempfile.mkdtemp()

    try:
        for uf in uploaded_files:
            dest = os.path.join(tmp_dir, uf.name)
            with open(dest, "wb") as f:
                f.write(uf.read())

        status.caption("📂 Parsing resumes...")
        progress.progress(15)
        resumes = load_all_resumes(tmp_dir, clean=True)

        if not resumes:
            st.error("No text could be extracted. Check if files are scanned PDFs.")
            return

        status.caption("📝 Processing job description...")
        progress.progress(25)
        cleaned_jd   = load_job_description(jd_input)
        resume_texts = tuple(r["cleaned_text"] for r in resumes)

        # ── [CHANGED] — Three method branches ────────────────────────────────
        if use_nlp:
            # ── Method 3: Full NLP Pipeline (NEW) ────────────────────────────
            resume_structs, jd_struct, sim_scores, rerank_scores, emb_model = \
                run_nlp_pipeline(resumes, resume_texts, cleaned_jd, api_key, status, progress)

            status.caption("🧮 Computing final scores...")
            progress.progress(90)
            results = score_all_nlp(
                resumes         = resumes,
                resume_structs  = resume_structs,
                jd_text         = cleaned_jd,
                jd_struct       = jd_struct,
                sim_scores      = sim_scores,
                rerank_scores   = rerank_scores,
                embedding_model = emb_model,
                top_n           = top_n,
            )

        elif use_bert:
            # ── Method 2: BERT ───────────────────────────────────────────────
            status.caption("🤖 Loading BERT model (first time ~1 min)...")
            progress.progress(45)
            sim_scores = run_bert(resume_texts, cleaned_jd)
            status.caption("🧮 Computing scores...")
            progress.progress(70)
            results = score_all_resumes(resumes, sim_scores, cleaned_jd, top_n=top_n)

        else:
            # ── Method 1: TF-IDF ─────────────────────────────────────────────
            status.caption("⚡ Computing TF-IDF similarity...")
            progress.progress(45)
            sim_scores = run_tfidf(resume_texts, cleaned_jd)
            status.caption("🧮 Computing scores...")
            progress.progress(70)
            results = score_all_resumes(resumes, sim_scores, cleaned_jd, top_n=top_n)

        progress.progress(100)
        status.empty()
        progress.empty()

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # ════════════════════════════════════════════════════════════════════════
    # RESULTS
    # ════════════════════════════════════════════════════════════════════════

    top_results = [r for r in results if r["is_top"]]
    method_label = (
        "🧠 NLP Pipeline (LLM + Reranker)" if use_nlp else
        "🤖 BERT Semantic"                  if use_bert else
        "⚡ TF-IDF"
    )

    st.markdown(f"## 🏆 Top {top_n} Candidates  ·  {method_label}")

    # ── Summary table ────────────────────────────────────────────────────────
    with st.expander("📊 Summary table (all candidates)", expanded=False):
        if use_nlp:
            df = pd.DataFrame([{
                "Rank":        r["rank"],
                "File":        r["filename"],
                "Final Score": r["final_score"],
                "Grade":       r["grade"],
                "Skills":      r["scores"]["skills"],
                "Reranker":    r["scores"]["reranker"],
                "Experience":  r["scores"]["experience"],
                "Education":   r["scores"]["education"],
                "Semantic":    r["scores"]["similarity"],
                "Projects":    r["scores"]["projects"],
                "Matched Skills": len(r["skill_gap"]["matched"]),
                "Missing Skills": len(r["skill_gap"]["missing"]),
            } for r in results])
        else:
            df = pd.DataFrame([{
                "Rank":        r["rank"],
                "File":        r["filename"],
                "Final Score": r["final_score"],
                "Grade":       r["grade"],
                "Similarity":  r["scores"]["similarity"],
                "Skills":      r["scores"]["skills"],
                "Experience":  r["scores"]["experience"],
                "Education":   r["scores"]["education"],
                "Matched Skills": len(r["skill_gap"]["matched"]),
                "Missing Skills": len(r["skill_gap"]["missing"]),
            } for r in results])

        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download CSV", data=df.to_csv(index=False),
            file_name="resume_screening_results.csv", mime="text/csv"
        )

    # ── Comparison chart ─────────────────────────────────────────────────────
    st.markdown("#### Score Breakdown — Top Candidates")
    # [CHANGED] — use NLP chart when NLP method selected
    fig_bar = make_comparison_bar_nlp(results, top_n) if use_nlp else make_comparison_bar(results, top_n)
    st.pyplot(fig_bar, use_container_width=True)
    plt.close(fig_bar)

    st.divider()
    st.markdown("#### Candidate Profiles")

    # ── Per-candidate cards ──────────────────────────────────────────────────
    for r in top_results:
        medal  = get_medal(r["rank"])
        gcolor = grade_color(r["grade"])

        with st.container(border=True):
            h1, h2, h3 = st.columns([0.08, 0.6, 0.32])
            with h1:
                st.markdown(f"<div style='font-size:28px;text-align:center'>{medal}</div>",
                            unsafe_allow_html=True)
            with h2:
                st.markdown(f"**{r['filename']}**")
                st.caption(f"Rank #{r['rank']} of {len(results)} candidates")
            with h3:
                st.markdown(
                    f"<div style='text-align:right'>"
                    f"<span style='font-size:28px;font-weight:700;color:{gcolor}'>{r['final_score']}</span>"
                    f"<span style='font-size:14px;color:#9ca3af'>/100</span> &nbsp;"
                    f"<span style='background:{gcolor}20;color:{gcolor};padding:3px 10px;border-radius:12px;"
                    f"font-size:12px;font-weight:600'>{r['grade']}</span></div>",
                    unsafe_allow_html=True
                )

            st.markdown("")
            left, right = st.columns([0.65, 0.35])

            with left:
                st.markdown("**Score Breakdown**")
                # [CHANGED] — show 6 bars for NLP, 4 bars for others
                if use_nlp:
                    components = [
                        ("Skill Match (semantic)",     r["scores"]["skills"],      0.35, "#7c3aed"),
                        ("Cross-Encoder Reranker",     r["scores"]["reranker"],    0.25, "#059669"),
                        ("Experience (LLM extracted)", r["scores"]["experience"],  0.15, "#f59e0b"),
                        ("Education",                  r["scores"]["education"],   0.10, "#3b82f6"),
                        ("Semantic Similarity",        r["scores"]["similarity"],  0.10, "#d85a30"),
                        ("Project Relevance",          r["scores"]["projects"],    0.05, "#888780"),
                    ]
                else:
                    components = [
                        ("Similarity (keyword/semantic)", r["scores"]["similarity"], 0.40, "#6366f1"),
                        ("Skill Match",                   r["scores"]["skills"],      0.30, "#10b981"),
                        ("Experience",                    r["scores"]["experience"],  0.20, "#f59e0b"),
                        ("Education",                     r["scores"]["education"],   0.10, "#3b82f6"),
                    ]
                bars_html = "".join(score_bar_html(l, s, w, c) for l, s, w, c in components)
                st.markdown(bars_html, unsafe_allow_html=True)

                exp = r["experience"]
                if exp.get("jd_required", 0) > 0:
                    sc = "#10b981" if exp["status"] in ("meets","exceeds") else "#ef4444"
                    st.caption(
                        f"Experience: candidate has **{exp['resume_years']} yrs** "
                        f"(JD requires {exp['jd_required']} yrs) — "
                        f"<span style='color:{sc};font-weight:600'>{exp['status']}</span>",
                        unsafe_allow_html=True
                    )

                # [NEW] — Show reranker score separately for NLP method
                if use_nlp:
                    rr = r["scores"]["reranker"]
                    rr_color = "#059669" if rr >= 60 else "#d97706" if rr >= 40 else "#dc2626"
                    st.caption(
                        f"Cross-Encoder reranker score: "
                        f"<span style='color:{rr_color};font-weight:600'>{rr:.1f}/100</span> "
                        f"— reads JD + resume together as a pair",
                        unsafe_allow_html=True
                    )

            with right:
                # [CHANGED] — use 6-axis radar for NLP method
                fig_radar = make_radar_chart_nlp(r) if use_nlp else make_radar_chart(r)
                st.pyplot(fig_radar, use_container_width=True)
                plt.close(fig_radar)

            # ── Skill gap ─────────────────────────────────────────────────────
            st.markdown("**Skill Gap Analysis**")
            sg = r["skill_gap"]
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(
                    f"<div style='font-size:12px;color:#059669;font-weight:600;margin-bottom:4px'>"
                    f"✅ Matched ({len(sg['matched'])})</div>"
                    + render_skill_pills(sg['matched'], 'pill-green'),
                    unsafe_allow_html=True
                )
            with c2:
                st.markdown(
                    f"<div style='font-size:12px;color:#dc2626;font-weight:600;margin-bottom:4px'>"
                    f"❌ Missing ({len(sg['missing'])})</div>"
                    + render_skill_pills(sg['missing'], 'pill-red'),
                    unsafe_allow_html=True
                )
            with c3:
                st.markdown(
                    f"<div style='font-size:12px;color:#2563eb;font-weight:600;margin-bottom:4px'>"
                    f"➕ Extra Skills ({len(sg['extra'])})</div>"
                    + render_skill_pills(sg['extra'][:8], 'pill-blue'),
                    unsafe_allow_html=True
                )

            # [NEW] — Show LLM extracted projects for NLP method
            if use_nlp:
                idx = next((i for i, res in enumerate(resumes)
                            if res["filename"] == r["filename"]), None)
                if idx is not None:
                    structs_list = st.session_state.get("nlp_structs", [])

            st.markdown("")

    # ── JD skills summary ────────────────────────────────────────────────────
    st.divider()
    with st.expander("🔍 Skills detected in Job Description"):
        jd_skills_found = extract_skills(cleaned_jd)
        if jd_skills_found:
            st.markdown(render_skill_pills(sorted(jd_skills_found), "pill-gray"),
                        unsafe_allow_html=True)
            st.caption(f"Total: {len(jd_skills_found)} skills detected from JD")
        else:
            st.info("No skills detected. Try adding specific names like 'Python', 'Docker'.")


if __name__ == "__main__":
    main()
