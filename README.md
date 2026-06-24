---
title: Resume Screening
emoji: 📄
colorFrom: indigo
colorTo: blue
sdk: streamlit
sdk_version: 1.40.2
app_file: app.py
pinned: false
---

# 📄 ResumeIQ — Smart Resume Screening & Ranking Pipeline

ResumeIQ is an AI-powered resume screening and matching system that automatically evaluates candidates against a job description. It provides a visual dashboard for HR teams and hiring managers to analyze candidate qualifications, skill gaps, and experience fits.

The application supports three screening methodologies ranging from fast keyword matching to state-of-the-art semantic ranking and LLM-assisted analysis.

---

## ✨ Key Features

* **Three Screening Methodologies:**
  1. **TF-IDF (Keyword-Based):** Fast and lightweight, matching specific technical terms and key phrases.
  2. **BERT Semantic Embeddings:** Uses the `BAAI/bge-base-en-v1.5` bi-encoder to understand the contextual meaning and synonyms in resumes, even if the phrasing differs from the job description.
  3. **Advanced NLP Pipeline (Hybrid LLM + Reranker):** 
     - **LLM Extraction:** Extracts exact skills, education, and years of experience using `GPT-4o-mini` (via OpenRouter).
     - **Cross-Encoder Reranking:** Leverages a cross-encoder model to read the job description and resume together as a pair, capturing deep contextual match scoring.
     - **6-Signal Weighted Scoring:** Computes final scores based on skills match (35%), cross-encoder reranking (25%), experience (15%), education (10%), semantic similarity (10%), and project relevance (5%).
* **Skill Gap Analysis:** Displays matched, missing, and extra skills for each candidate using visual color-coded badges.
* **Interactive Visualizations:**
  * **Stacked Bar Charts:** Displays candidate score breakdowns side-by-side.
  * **6-Axis Radar Charts:** Visualizes multidimensional candidate profiles (Skills, Reranker, Experience, Education, Semantic, Projects).
* **Exportable Results:** Provides a downloadable candidate ranking summary table (CSV format).

---

## 📁 Repository Structure

```text
├── app.py                 # Main Streamlit web application
├── requirements.txt       # Python dependencies (optimized for Python 3.13)
├── utils/
│   ├── parser.py          # PDF and DOCX file parser
│   ├── skill_extractor.py # Regex-based fallback skill extractor
│   ├── scorer.py          # TF-IDF and BERT scoring engines
│   ├── reranker.py        # Cross-encoder model loader and scorer
│   ├── llm_extractor.py   # LLM extraction client for OpenRouter
│   └── nlp_scorer.py      # Weights and scoring logic for the NLP pipeline
└── README.md              # Project documentation
```

---

## 🚀 Quick Start (Local Run)

### 1. Prerequisites
Ensure you have **Python 3.10+** (fully supports Python 3.13) and `pip` installed.

### 2. Installation
Clone your repository, navigate to the folder, and install the dependencies:
```bash
git clone https://github.com/Nim369/Resume_screening.git
cd Resume_screening
pip install -r app/requirements.txt
```

### 3. Run the App
Launch the Streamlit app:
```bash
streamlit run app/app.py
```
The application will open automatically at `http://localhost:8501`.

---

## ⚙️ Configuration & API Keys

To use the **NLP Pipeline (LLM + Reranker)** mode:
1. Obtain an API key from [OpenRouter](https://openrouter.ai/).
2. Paste the API key into the sidebar text box in the web interface.
3. (Optional) For deployment, you can set the environment variable `OPENROUTER_API_KEY` on your cloud provider.

---

## ☁️ Deployment

### 🐳 Deploy to Hugging Face Spaces (Recommended)
Hugging Face Spaces offers **16 GB RAM** on its free tier, which is ideal for the BERT and Cross-Encoder neural models.

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) and create a new Space.
2. Select **Streamlit** as the SDK and choose the free CPU basic hardware tier.
3. Commit and push this codebase directly to the Hugging Face git remote repository.

### ⚡ Deploy to Streamlit Community Cloud
1. Push this codebase to a public GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
3. Click **New app**, select your repository, set the main file path to `app/app.py`, and click **Deploy**.
