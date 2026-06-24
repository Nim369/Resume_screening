"""
utils/skill_extractor.py
------------------------
Extracts skills from JD and resumes, computes skill gap analysis.
No external ML model needed — uses a curated keyword taxonomy.
"""

import re

# ─────────────────────────────────────────────────────────────────────────────
# SKILL TAXONOMY  — extend this dict to add more domains
# Each key = category, value = list of skill keywords/phrases
# ─────────────────────────────────────────────────────────────────────────────
SKILL_TAXONOMY = {
    # ── Programming Languages ──
    "Python":           ["python"],
    "Java":             ["java", "java 8", "java 11", "java 17", "spring boot", "spring"],
    "JavaScript":       ["javascript", "js", "node.js", "nodejs", "typescript", "ts"],
    "C++":              ["c++", "cpp"],
    "C":                ["c programming", " c "],
    "Go":               ["golang", " go "],
    "Rust":             ["rust"],
    "Scala":            ["scala"],
    "R":                [" r ", "r programming", "rstudio"],
    "PHP":              ["php", "laravel"],
    "Ruby":             ["ruby", "ruby on rails", "rails"],
    "Swift":            ["swift", "swiftui"],
    "Kotlin":           ["kotlin"],
    "MATLAB":           ["matlab"],

    # ── ML / AI ──
    "Machine Learning": ["machine learning", " ml "],
    "Deep Learning":    ["deep learning", " dl "],
    "NLP":              ["nlp", "natural language processing", "text mining"],
    "Computer Vision":  ["computer vision", "cv", "image recognition", "object detection"],
    "Reinforcement Learning": ["reinforcement learning", "rl", "q-learning"],
    "LLM":              ["llm", "large language model", "gpt", "chatgpt", "transformers"],
    "scikit-learn":     ["scikit-learn", "sklearn"],
    "TensorFlow":       ["tensorflow", "tf", "keras"],
    "PyTorch":          ["pytorch", "torch"],
    "Hugging Face":     ["huggingface", "hugging face", "transformers library"],
    "XGBoost":          ["xgboost", "xgb", "lightgbm", "lgbm", "catboost"],
    "OpenCV":           ["opencv", "cv2"],
    "YOLO":             ["yolo", "yolov5", "yolov8"],

    # ── Data Science ──
    "pandas":           ["pandas"],
    "NumPy":            ["numpy", "np"],
    "Matplotlib":       ["matplotlib", "seaborn", "plotly", "visualization"],
    "Statistics":       ["statistics", "statistical analysis", "hypothesis testing", "probability"],
    "Data Wrangling":   ["data wrangling", "data cleaning", "etl", "data preprocessing"],
    "Feature Engineering": ["feature engineering", "feature selection"],

    # ── Web Frameworks ──
    "FastAPI":          ["fastapi", "fast api"],
    "Flask":            ["flask"],
    "Django":           ["django"],
    "REST API":         ["rest api", "restful", "rest", "api development", "web api"],
    "GraphQL":          ["graphql"],
    "React":            ["react", "reactjs", "react.js"],
    "Angular":          ["angular", "angularjs"],
    "Vue":              ["vue", "vuejs", "vue.js"],
    "Next.js":          ["next.js", "nextjs"],

    # ── Databases ──
    "SQL":              ["sql", "mysql", "postgresql", "postgres", "sqlite", "oracle", "ms sql"],
    "NoSQL":            ["nosql", "mongodb", "mongo", "cassandra", "couchdb"],
    "Redis":            ["redis", "cache"],
    "Elasticsearch":    ["elasticsearch", "elastic search", "kibana"],
    "BigQuery":         ["bigquery", "big query"],

    # ── Cloud & DevOps ──
    "AWS":              ["aws", "amazon web services", "ec2", "s3", "lambda", "sagemaker"],
    "GCP":              ["gcp", "google cloud", "google cloud platform", "bigquery"],
    "Azure":            ["azure", "microsoft azure"],
    "Docker":           ["docker", "dockerfile", "containerization", "containers"],
    "Kubernetes":       ["kubernetes", "k8s", "kubectl"],
    "CI/CD":            ["ci/cd", "cicd", "jenkins", "github actions", "gitlab ci", "travis"],
    "Terraform":        ["terraform", "infrastructure as code", "iac"],
    "Linux":            ["linux", "ubuntu", "centos", "bash", "shell scripting"],

    # ── Data Engineering ──
    "Apache Spark":     ["apache spark", "pyspark", "spark"],
    "Apache Kafka":     ["kafka", "apache kafka"],
    "Airflow":          ["airflow", "apache airflow"],
    "Hadoop":           ["hadoop", "hdfs", "hive"],
    "dbt":              ["dbt", "data build tool"],

    # ── Tools & Practices ──
    "Git":              ["git", "github", "gitlab", "bitbucket", "version control"],
    "Jupyter":          ["jupyter", "jupyter notebook", "ipynb"],
    "Agile":            ["agile", "scrum", "kanban", "sprint"],
    "Unit Testing":     ["unit testing", "pytest", "junit", "tdd", "test driven"],
    "MLOps":            ["mlops", "ml pipeline", "model deployment", "model serving"],

    # ── Soft Skills ──
    "Communication":    ["communication", "written communication", "verbal communication"],
    "Teamwork":         ["teamwork", "team player", "collaboration", "cross-functional"],
    "Leadership":       ["leadership", "team lead", "tech lead", "mentoring"],
    "Problem Solving":  ["problem solving", "analytical", "critical thinking"],

    # ── Education ──
    "B.Tech / B.E.":    ["b.tech", "b.e.", "bachelor of technology", "bachelor of engineering", "btech"],
    "M.Tech / M.E.":    ["m.tech", "m.e.", "master of technology", "mtech"],
    "MCA":              ["mca", "master of computer"],
    "MBA":              ["mba"],
    "B.Sc":             ["b.sc", "bsc", "bachelor of science"],
    "M.Sc":             ["m.sc", "msc", "master of science"],
    "PhD":              ["phd", "doctorate", "ph.d"],
    "Computer Science": ["computer science", "cs", "cse", "information technology", " it "],
}

# Years of experience patterns
EXPERIENCE_PATTERNS = [
    r"(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s+)?(?:experience|exp)",
    r"(?:experience|exp)\s*(?:of\s+)?(\d+)\+?\s*(?:years?|yrs?)",
    r"(\d+)\s*-\s*\d+\s*(?:years?|yrs?)",
]

# Education level hierarchy (higher index = higher qualification)
EDUCATION_LEVELS = ["high school", "diploma", "b.sc", "bsc", "b.tech", "btech", "b.e.",
                    "mca", "m.sc", "msc", "m.tech", "mtech", "m.e.", "mba", "phd", "ph.d"]


def extract_skills(text: str) -> set:
    """
    Extract a set of skill names present in the given text.
    Returns canonical skill names (keys from SKILL_TAXONOMY).
    """
    text_lower = text.lower()
    found = set()

    for skill_name, keywords in SKILL_TAXONOMY.items():
        for kw in keywords:
            # Use word-boundary-like matching
            pattern = r'(?<![a-z0-9])' + re.escape(kw) + r'(?![a-z0-9])'
            if re.search(pattern, text_lower):
                found.add(skill_name)
                break  # Found one keyword for this skill, move on

    return found


def extract_years_experience(text: str) -> int:
    """
    Extract the maximum years of experience mentioned in the text.
    Returns 0 if none found.
    """
    text_lower = text.lower()
    max_years = 0

    for pattern in EXPERIENCE_PATTERNS:
        matches = re.findall(pattern, text_lower)
        for m in matches:
            try:
                years = int(m)
                if 0 < years < 50:  # Sanity check
                    max_years = max(max_years, years)
            except ValueError:
                pass

    return max_years


def compute_skill_gap(jd_skills: set, resume_skills: set) -> dict:
    """
    Compute skill gap between JD requirements and resume skills.

    Returns:
        {
            "matched":  set of skills present in both JD and resume,
            "missing":  set of skills JD requires but resume lacks,
            "extra":    set of skills resume has beyond JD requirements,
            "match_pct": float (0-100)
        }
    """
    matched = jd_skills & resume_skills
    missing = jd_skills - resume_skills
    extra   = resume_skills - jd_skills

    match_pct = (len(matched) / len(jd_skills) * 100) if jd_skills else 0.0

    return {
        "matched":   matched,
        "missing":   missing,
        "extra":     extra,
        "match_pct": round(match_pct, 1)
    }


def compute_experience_score(jd_text: str, resume_text: str) -> dict:
    """
    Compare years of experience in JD requirement vs resume.

    Returns:
        {
            "jd_required":    int (years required by JD),
            "resume_years":   int (years candidate has),
            "score":          float (0-100),
            "status":         str ("exceeds" | "meets" | "below")
        }
    """
    jd_yrs     = extract_years_experience(jd_text)
    resume_yrs = extract_years_experience(resume_text)

    if jd_yrs == 0:
        score  = 75.0   # JD didn't specify — give partial credit
        status = "not specified"
    elif resume_yrs >= jd_yrs:
        score  = 100.0
        status = "exceeds" if resume_yrs > jd_yrs else "meets"
    else:
        # Partial credit: e.g. 2 yrs when 3 required → 67%
        score  = round((resume_yrs / jd_yrs) * 100, 1)
        status = "below"

    return {
        "jd_required":  jd_yrs,
        "resume_years": resume_yrs,
        "score":        score,
        "status":       status
    }


def compute_education_score(resume_text: str, jd_text: str) -> dict:
    """
    Score education level. Checks if resume meets or exceeds JD's education requirement.

    Returns:
        {
            "resume_level": str,
            "jd_level":     str,
            "score":        float (0-100)
        }
    """
    text_r = resume_text.lower()
    text_j = jd_text.lower()

    # Find highest education level in resume
    resume_edu_idx = -1
    resume_edu_name = "not mentioned"
    for i, level in enumerate(EDUCATION_LEVELS):
        if level in text_r:
            resume_edu_idx = i
            resume_edu_name = level.upper()

    # Find education requirement in JD
    jd_edu_idx = -1
    jd_edu_name = "not specified"
    for i, level in enumerate(EDUCATION_LEVELS):
        if level in text_j:
            jd_edu_idx = i
            jd_edu_name = level.upper()

    if jd_edu_idx == -1:
        score = 75.0  # JD didn't specify education
    elif resume_edu_idx >= jd_edu_idx:
        score = 100.0
    elif resume_edu_idx == jd_edu_idx - 1:
        score = 70.0  # One level below
    else:
        score = max(0, 50.0 - (jd_edu_idx - resume_edu_idx) * 10)

    return {
        "resume_level": resume_edu_name,
        "jd_level":     jd_edu_name,
        "score":        score
    }
