# %%
# ------------------------------------------------------------
# Import: Standard Libraries + Data Handling + Streamlit + NLP
# ------------------------------------------------------------
import json                    # Einlesen von JSON-Dateien (LinkedIn CV Daten)
import re                      # Regex (z.B. für Text-Normalisierung und Abkürzungen zählen)
from pathlib import Path       # Plattform-unabhängige Pfade / File-Handling
from collections import Counter  # Häufigkeiten zählen (Top Tokens)
from hybrid_model_Variante_B.hybrid_predictor import HybridPredictor
   

import numpy as np             # Numerische Operationen (z.B. NaN checks, Arrays)
import pandas as pd            # DataFrames für strukturierte Daten
import streamlit as st         # Streamlit Framework für Dashboard/Frontend
import matplotlib.pyplot as plt  # Visualisierung (Histogramme, Barplots)

import joblib                 

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# CountVectorizer: Häufigkeiten von Tokens/N-Grams zählen
# TfidfVectorizer: TF-IDF Keywords extrahieren (informativer als reine Counts)


# -----------------------------
# Config / constants (aligned with your final RuleLabeler dictionaries)
# -----------------------------
# STOPWORDS: Wörter ohne Informationsgehalt (z.B. "and", "of", "der"), die bei Keyword-EDA entfernt werden
STOPWORDS = set(["and","of","for","und","der","die","das","in","mit","to","de","la","le","des","et","en","as"])

# Abkürzungen, die Senioritätslevel signalisieren können (aus finalem Code übernommen)
SENIORITY_ABBR = {
    "jr":"Junior", "sr":"Senior", "lead":"Lead", "chief":"Lead",
    "dir":"Director", "vp":"Director", "mgr":"Management"
}

# Abkürzungen, die oft Departments signalisieren (aus finalem Code übernommen)
DEPARTMENT_ABBR = {
    "it":"Information Technology",
    "hr":"Human Resources",
    "bd":"Business Development",
    "ops":"Operations"
}

# C-Level / Executive Abkürzungen (aus finalem Code übernommen)
C_LEVEL_ABBR = {
    "CEO":"Chief Executive Officer",
    "CFO":"Chief Financial Officer",
    "COO":"Chief Operating Officer",
    "CTO":"Chief Technology Officer",
    "CMO":"Chief Marketing Officer",
    "CIO":"Chief Information Officer",
    "CHRO":"Chief Human Resources Officer",
    "EVP":"Executive Vice President",
    "SVP":"Senior Vice President",
    "VP":"Vice President",
    "AVP":"Assistant / Associate Vice President",
}

# Standard Stopwords, die im Dashboard als Basis verwendet werden
DEFAULT_STOPWORDS = STOPWORDS.copy()

# Seniority Order (Rangfolge) für Career-History Analyse (Progression über Zeit)
SENIORITY_ORDER = ["Junior", "Professional", "Senior", "Lead", "Management", "Director"]


# -----------------------------
# Helpers
# -----------------------------
def safe_str(x) -> str:
    """
    Konvertiert Werte robust in Strings:
    - None oder NaN werden zu ""
    - ansonsten str(x)
    Zweck: verhindert Fehler bei fehlenden Feldern in JSON/CSV.
    """
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    return str(x)


def normalize_title(text: str) -> str:
    """
    Normalisiert Job Titles für NLP/EDA:
    - lowercase
    - Umlaut-Handling (ä->ae etc.)
    - Entfernt Gender-Endungen ('innen' / 'in')
    - Entfernt Sonderzeichen
    - Entfernt doppelte Leerzeichen

    Ergebnis wird in 'position_norm' gespeichert.
    """
    text = safe_str(text).lower()

    # German umlauts ersetzen, damit Modelle/Tokenizer konsistenter arbeiten
    text = (
        text.replace("ä", "ae")
            .replace("ö", "oe")
            .replace("ü", "ue")
            .replace("ß", "ss")
    )

    # Entfernt deutsche gendered forms (z.B. "entwicklerinnen", "entwicklerin")
    text = re.sub(r"(innen|in)\b", "", text)

    # Nur alphanumerische Zeichen + Spaces behalten
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # Mehrfach-Leerzeichen zu einem Space reduzieren
    text = re.sub(r"\s+", " ", text).strip()
    return text


def flatten_profiles_to_jobs(json_data) -> pd.DataFrame:
    """
    Robustly flatten LinkedIn CV JSON into a job-level DataFrame.
    Handles cases where elements are dict-profiles OR job-lists.
    """
    rows = []

    # Falls die JSON Root ein Dict ist: häufig gibt es Container Keys wie "profiles" / "data" / "items"
    if isinstance(json_data, dict):
        for k in ["profiles", "data", "items"]:
            if k in json_data and isinstance(json_data[k], list):
                json_data = json_data[k]
                break

    # Wenn Root keine Liste ist, kann nicht geflattet werden -> leeres DataFrame
    if not isinstance(json_data, list):
        return pd.DataFrame()

    for pi, prof in enumerate(json_data):

        if isinstance(prof, list):
            person_id = pi
            jobs = prof
        elif isinstance(prof, dict):
            person_id = prof.get("person_id", prof.get("id", pi))
            jobs = prof.get("jobs", prof.get("positions", prof.get("experience", [])))
        else:
            continue

        if not isinstance(jobs, list):
            continue

        for ji, job in enumerate(jobs):

            if not isinstance(job, dict):
                continue

            title = job.get("position", job.get("title", job.get("job_title", "")))
            status = job.get("status", job.get("current", ""))

            if isinstance(status, bool):
                status = "ACTIVE" if status else "INACTIVE"

            dept = job.get("department", job.get("domain", ""))
            sen = job.get("seniority", job.get("level", ""))

            start_date = job.get("start_date", job.get("startDate", job.get("from", "")))
            end_date = job.get("end_date", job.get("endDate", job.get("to", "")))

            rows.append({
                "person_id": person_id,
                "job_idx": ji,
                "status": safe_str(status),
                "position": safe_str(title),
                "position_norm": normalize_title(title),
                "department": safe_str(dept),
                "seniority": safe_str(sen),
                "start_date": safe_str(start_date),
                "end_date": safe_str(end_date),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["status"] = (
        df["status"]
        .astype(str)
        .str.upper()
        .replace({"CURRENT": "ACTIVE", "TRUE": "ACTIVE", "FALSE": "INACTIVE"})
    )
    return df


@st.cache_data(show_spinner=False)
def load_json_file(path: str):
    """
    Lädt JSON Datei vom Pfad.
    """
    p = Path(path)
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)
# -----------------------------
# Funktion zum CSV-Laden
# -----------------------------
@st.cache_data(show_spinner=False)
def load_csv_file(path: str) -> pd.DataFrame:
    """
    Lädt CSV-Datei vom Pfad in ein DataFrame.
    Gibt ein leeres DataFrame zurück, wenn die Datei nicht existiert.
    """
    p = Path(path)
    if not p.exists():
        st.warning(f"CSV-Datei nicht gefunden: {path}")
        return pd.DataFrame()
    return pd.read_csv(p)

# -----------------------------
# CSV-Dateien laden
# -----------------------------
df_department = load_csv_file("department-v2.csv")
df_seniority = load_csv_file("seniority-v2.csv")


def top_tokens(texts: pd.Series, stopwords: set, top_n=30):
    tokens = []
    for t in texts.dropna().astype(str):
        t = normalize_title(t)
        toks = [w for w in t.split() if (len(w) >= 2 and w not in stopwords)]
        tokens.extend(toks)
    return Counter(tokens).most_common(top_n)


def top_char_ngrams(texts: pd.Series, ngram_range=(3, 5), top_n=30):
    vec = CountVectorizer(analyzer="char_wb", ngram_range=ngram_range, lowercase=True, min_df=1)
    X = vec.fit_transform(texts.fillna("").astype(str))
    counts = np.asarray(X.sum(axis=0)).ravel()
    feats = vec.get_feature_names_out()
    pairs = list(zip(feats, counts))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:top_n]


def top_tfidf_terms(texts: pd.Series, top_n=30, stopwords=None):
    stopwords = list(stopwords) if stopwords is not None else None
    vec = TfidfVectorizer(
        lowercase=True,
        stop_words=stopwords,
        ngram_range=(1, 1),
        min_df=2
    )
    X = vec.fit_transform(texts.fillna("").astype(str))
    scores = np.asarray(X.mean(axis=0)).ravel()
    terms = vec.get_feature_names_out()
    pairs = sorted(zip(terms, scores), key=lambda x: x[1], reverse=True)[:top_n]
    return pd.DataFrame(pairs, columns=["term", "avg_tfidf"])


def top_bigrams(texts: pd.Series, stopwords: set, top_n=30):
    vec = CountVectorizer(
        lowercase=True,
        stop_words=list(stopwords),
        ngram_range=(2, 2),
        min_df=2
    )
    X = vec.fit_transform(texts.fillna("").astype(str))
    counts = np.asarray(X.sum(axis=0)).ravel()
    feats = vec.get_feature_names_out()
    pairs = sorted(zip(feats, counts), key=lambda x: x[1], reverse=True)[:top_n]
    return pd.DataFrame(pairs, columns=["bigram", "count"])


def top_keywords_per_label(df: pd.DataFrame, text_col: str, label_col: str, top_n=12, stopwords=None, min_rows=10):
    stopwords = list(stopwords) if stopwords is not None else None
    out_rows = []

    labels = (
        df[label_col]
        .replace("", np.nan)
        .dropna()
        .unique()
        .tolist()
    )

    for lab in sorted(labels):
        sub = df[df[label_col] == lab]
        if len(sub) < min_rows:
            continue

        vec = TfidfVectorizer(
            lowercase=True,
            stop_words=stopwords,
            ngram_range=(1, 1),
            min_df=2
        )
        X = vec.fit_transform(sub[text_col].fillna("").astype(str))
        scores = np.asarray(X.mean(axis=0)).ravel()
        terms = vec.get_feature_names_out()

        top = sorted(zip(terms, scores), key=lambda x: x[1], reverse=True)[:top_n]
        for term, score in top:
            out_rows.append({"label": lab, "term": term, "avg_tfidf": score})

    return pd.DataFrame(out_rows)


def count_abbreviations_in_titles(df: pd.DataFrame) -> pd.DataFrame:
    text = " ".join(df["position_norm"].fillna("").astype(str).tolist())

    def count_word(abbr: str) -> int:
        return len(re.findall(rf"\b{re.escape(abbr.lower())}\b", text))

    rows = []

    for abbr, label in SENIORITY_ABBR.items():
        rows.append({
            "category": "seniority_abbr",
            "abbr": abbr,
            "maps_to": label,
            "count": count_word(abbr)
        })

    for abbr, label in DEPARTMENT_ABBR.items():
        rows.append({
            "category": "department_abbr",
            "abbr": abbr,
            "maps_to": label,
            "count": count_word(abbr)
        })

    for abbr, longform in C_LEVEL_ABBR.items():
        rows.append({
            "category": "c_level",
            "abbr": abbr.lower(),
            "maps_to": longform,
            "count": count_word(abbr)
        })

    df_out = pd.DataFrame(rows)
    df_out = df_out.sort_values(["category", "count"], ascending=[True, False])
    return df_out


def compute_department_changes(df_jobs: pd.DataFrame) -> pd.DataFrame:
    out = []
    for pid, g in df_jobs.sort_values(["person_id", "job_idx"]).groupby("person_id"):
        depts = [d for d in g["department"].tolist() if d]
        changes = 0
        for i in range(1, len(depts)):
            if depts[i] != depts[i - 1]:
                changes += 1
        out.append({"person_id": pid, "dept_changes": changes, "jobs_count": len(g)})
    return pd.DataFrame(out)


def compute_seniority_increases(df_jobs: pd.DataFrame) -> pd.DataFrame:
    rank = {k: i for i, k in enumerate(SENIORITY_ORDER)}
    out = []
    for pid, g in df_jobs.sort_values(["person_id", "job_idx"]).groupby("person_id"):
        levels = [x for x in g["seniority"].tolist() if x in rank]
        inc = 0
        for i in range(1, len(levels)):
            if rank[levels[i]] > rank[levels[i - 1]]:
                inc += 1
        out.append({"person_id": pid, "sen_increases": inc, "jobs_count": len(g)})
    return pd.DataFrame(out)


#  HybridPredictor Loader
@st.cache_resource(show_spinner=False)
def load_hybrid_model():
    return HybridPredictor(model_dir="hybrid_model_Variante_B")



# -----------------------------
# Streamlit app
# -----------------------------
st.set_page_config(page_title="LinkedIn CV NLP EDA Dashboard", layout="wide")
st.title("Capstone Project")


with st.sidebar:
    st.header("Paths")

    annotated_path = st.text_input("Annotated JSON (evaluation)", value="linkedin-cvs-annotated.json")
    unlabeled_path = st.text_input("Unlabeled JSON (optional)", value="linkedin-cvs-not-annotated.json")

    st.divider()
    st.header("Pages")
    page = st.radio(
        "Navigation",
        ["Overview", "Labels", "Job Title NLP", "Career History", "Data Quality", "Prediction (Hybrid)"],
        index=0
    )

# Load data
annotated_json = load_json_file(annotated_path)
unlabeled_json = load_json_file(unlabeled_path)

df_ann_jobs = flatten_profiles_to_jobs(annotated_json) if annotated_json is not None else pd.DataFrame()
df_unl_jobs = flatten_profiles_to_jobs(unlabeled_json) if unlabeled_json is not None else pd.DataFrame()

df_ann_active = df_ann_jobs[df_ann_jobs["status"] == "ACTIVE"].copy() if not df_ann_jobs.empty else pd.DataFrame()
df_unl_active = df_unl_jobs[df_unl_jobs["status"] == "ACTIVE"].copy() if not df_unl_jobs.empty else pd.DataFrame()


# -----------------------------
# Page: Overview
# -----------------------------

if page == "Overview":
    st.markdown("This page gives a high-level summary of the annotated and unlabeled datasets.")
    st.divider()

    st.subheader("Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Annotated Profiles (CVs)", int(df_ann_jobs["person_id"].nunique()) if not df_ann_jobs.empty else 0)
    with c2:
        st.metric("Annotated Jobs", int(len(df_ann_jobs)) if not df_ann_jobs.empty else 0)
    with c3:
        st.metric("Annotated ACTIVE Jobs", int(len(df_ann_active)) if not df_ann_active.empty else 0)
    with c4:
        st.metric("Unlabeled ACTIVE Jobs", int(len(df_unl_active)) if not df_unl_active.empty else 0)

    st.markdown(
        """
**Target definition (project):** The prediction target is the job entry with `status == ACTIVE`, representing each person`s current role.  

        """
    )

    st.divider()
    st.subheader("Sample of ACTIVE Jobs")

    st.markdown(
        """
    **Identifier definitions:**

    - **person_id** identifies a unique person (LinkedIn profile).  
    All jobs with the same *person_id* belong to the same CV.

    - **job_idx** indicates the index of a job within a person’s career history.  
    It shows which job entry it is for that person (0 = first job, 1 = second job, etc.).
        """
    )


    if df_ann_active.empty:
        st.warning("No annotated ACTIVE jobs loaded. Check file path or JSON structure.")
    else:
        st.write("**Sample of ACTIVE jobs (annotated):**")
        st.dataframe(
            df_ann_active[["person_id", "job_idx", "position", "department", "seniority"]].head(25),
            use_container_width=True
        )

    if not df_unl_active.empty:
        st.write("**Sample of ACTIVE jobs (unlabeled):**")
        st.dataframe(
            df_unl_active[["person_id", "job_idx", "position"]].head(25),
            use_container_width=True
        )


# -----------------------------
# Page: Labels
# -----------------------------
elif page == "Labels":
    st.markdown(
        "This page shows the distribution of labels in the annotated dataset (ACTIVE jobs) "
        "and in the CSV files (departments and seniority)."
    )
    st.divider()


    # --- First row: Annotated ACTIVE Jobs ---
    st.markdown("### Label Distribution: Annotated ACTIVE Jobs")
    colA, colB = st.columns(2)

    with colA:
        st.markdown("#### Department distribution")
        if df_ann_active.empty:
            st.warning("No annotated ACTIVE jobs available.")
        else:
            dept_counts = df_ann_active["department"].replace("", np.nan).dropna().value_counts()
            fig = plt.figure()
            dept_counts.head(25).plot(kind="bar")
            plt.title("Departments (ACTIVE jobs)")
            plt.ylabel("Number of jobs")
            plt.xlabel("label")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig, clear_figure=True)

    with colB:
        st.markdown("#### Seniority distribution")
        if not df_ann_active.empty:
            sen_counts = df_ann_active["seniority"].replace("", np.nan).dropna().value_counts()
            fig = plt.figure()
            sen_counts.plot(kind="bar")
            plt.title("Seniority (ACTIVE jobs)")
            plt.ylabel("Number of jobs")
            plt.xlabel("label")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig, clear_figure=True)
    

    st.divider()

    # --- Second row: CSV datasets ---
    st.markdown("### Label Distribution: CSV Files")
    colC, colD = st.columns(2)

    with colC:
        st.markdown("#### Department distribution")
        if not df_department.empty:
            dept_counts_csv = df_department['label'].value_counts()
            fig = plt.figure()
            dept_counts_csv.plot(kind="bar")
            plt.title("Departments (CSV)")
            plt.ylabel("Number of jobs")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig, clear_figure=True)

    with colD:
        st.markdown("#### Seniority distribution")
        if not df_seniority.empty:
            sen_counts_csv = df_seniority['label'].value_counts()
            fig = plt.figure()
            sen_counts_csv.plot(kind="bar")
            plt.title("Seniority (CSV)")
            plt.ylabel("Number of jobs")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig, clear_figure=True)

    


# -----------------------------
# Page: Job Title NLP
# -----------------------------
elif page == "Job Title NLP":
    st.markdown(
        " Select the dataset for which you want to explore various analyses of job titles."
        " You can also add custom stopwords to ignore common or irrelevant terms, helping the analysis focus on more meaningful words." )   
    st.divider()
    st.subheader("Job Title NLP Exploration (ACTIVE jobs)")


    st.markdown(
    "Based on the dataset you select, all job titles are analyzed and key statistics such as average and median length (in words and characters) are calculated. " \
    "You can also add custom stopwords to ignore common or irrelevant terms, helping the analysis focus on more meaningful words in job titles."
    )

    if df_ann_active.empty and df_unl_active.empty:
        st.warning("No ACTIVE jobs loaded.")
    else:
        dataset_choice = st.selectbox(
            "Select dataset",
            ["Annotated ACTIVE", "Unlabeled ACTIVE"],
            index=0
        )

        df_target = df_ann_active if dataset_choice == "Annotated ACTIVE" else df_unl_active
        if df_target.empty:
            st.warning("Selected dataset is empty.")
        else:
            stopwords = DEFAULT_STOPWORDS.copy()
            extra_sw = st.text_input("Add stopwords (comma-separated)", value="")
            if extra_sw.strip():
                for w in extra_sw.split(","):
                    stopwords.add(w.strip().lower())
            
            st.divider()
            st.subheader("Analysis")
            st.markdown("Here you can see key statistics of the job titles in the selected dataset.")

            avg_words = df_target["position_norm"].fillna("").apply(lambda x: len(x.split())).mean()
            avg_chars = df_target["position_norm"].fillna("").apply(len).mean()
            med_words = df_target["position_norm"].fillna("").apply(lambda x: len(x.split())).median()
            med_chars = df_target["position_norm"].fillna("").apply(len).median()

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Avg words/title", f"{avg_words:.2f}")
            with m2:
                st.metric("Median words/title", f"{med_words:.0f}")
            with m3:
                st.metric("Avg chars/title", f"{avg_chars:.2f}")
            with m4:
                st.metric("Median chars/title", f"{med_chars:.0f}")

            st.markdown("", unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Title length (words)")
                lengths = df_target["position_norm"].fillna("").apply(lambda x: len(x.split()))
                fig = plt.figure()
                plt.hist(lengths, bins=30)
                plt.title("Distribution of job title length (words)")
                plt.xlabel("Words per title")
                plt.ylabel("Count")
                st.pyplot(fig, clear_figure=True)

            with col2:
                st.markdown("#### Title length (characters)")
                charlens = df_target["position_norm"].fillna("").apply(len)
                fig = plt.figure()
                plt.hist(charlens, bins=30)
                plt.title("Distribution of job title length (characters)")
                plt.xlabel("Characters per title")
                plt.ylabel("Count")
                st.pyplot(fig, clear_figure=True)

            st.divider()

            col3, col4 = st.columns(2)

            with col3:
                st.markdown("#### Top word tokens (normalized titles)")
                top_n = st.slider("Top N tokens", 10, 100, 30)
                common = top_tokens(df_target["position"], stopwords=stopwords, top_n=top_n)
                df_tokens = pd.DataFrame(common, columns=["token", "count"])
                df_tokens.index = df_tokens.index + 1  # Index beginnt jetzt bei 1
                st.dataframe(df_tokens, use_container_width=True)

            with col4:
                st.markdown("#### Top character n-grams (3–5, char_wb)")
                top_n_ng = st.slider("Top N char n-grams", 10, 100, 30)
                common_ng = top_char_ngrams(df_target["position_norm"], ngram_range=(3, 5), top_n=top_n_ng)
                df_ngrams = pd.DataFrame(common_ng, columns=["char_ngram", "count"])
                df_ngrams.index = df_ngrams.index + 1  # Index beginnt jetzt bei 1
                st.dataframe(df_ngrams, use_container_width=True)
            
            st.markdown("Top word tokens (left column): Shows the most frequent words in job titles. You can use the slider to adjust how many of the top words are displayed."
            "Top character n-grams (right column): Shows the most frequent letter sequences (3–5 characters) in job titles. The slider lets you choose how many of the top n-grams to display.")

            

            st.divider()

            st.markdown("#### Top TF–IDF keywords (overall)")
            tfidf_n = st.slider("Top N TF–IDF keywords", 10, 100, 40)
            df_tfidf = top_tfidf_terms(df_target["position_norm"], top_n=tfidf_n, stopwords=list(stopwords))
            df_tfidf.index = df_tfidf.index + 1  # Index beginnt bei 1
            st.dataframe(df_tfidf, use_container_width=True)
            st.markdown("This table displays the most frequent two-word combinations (bigrams) found in the job titles.")

            st.divider()
            
            st.markdown("#### Top bigrams (2-word phrases)")
            bigram_n = st.slider("Top N bigrams", 10, 100, 40)
            df_bi = top_bigrams(df_target["position_norm"], stopwords=stopwords, top_n=bigram_n)
            df_bi.index = df_bi.index + 1  # Start index at 1
            st.dataframe(df_bi, use_container_width=True)
            st.markdown(
                "This table shows the most important words in the job titles, ranked by their TF–IDF score, "
                "which indicates how distinctive a word is across all titles. "
                "Use the slider to select how many top keywords you want to display."
            )



            st.divider()

            st.markdown("#### Abbreviation & C-level frequency (aligned with final RuleLabeler dictionaries)")
            df_abbr = count_abbreviations_in_titles(df_target)
            show_zero = st.checkbox("Show abbreviations with count = 0", value=False)
            if not show_zero:
                df_abbr = df_abbr[df_abbr["count"] > 0]
            st.dataframe(df_abbr, use_container_width=True)
            st.markdown(
                "This table shows the frequency of abbreviations and C-level terms (e.g., CEO, CFO, VP) in the job titles. "
                "Use the checkbox to include abbreviations that do not appear in the dataset."
            )

            st.divider()

            st.markdown("### Keyword search (show example titles)")
            query = st.text_input("Enter a keyword (e.g., manager, sales, engineer)", value="manager").strip().lower()

            if query:
                hits = df_target[df_target["position_norm"].str.contains(re.escape(query), na=False)]
                st.write(f"Found **{len(hits)}** titles containing **'{query}'**.")
                st.dataframe(hits[["position", "position_norm"]].head(50), use_container_width=True)
            
            st.markdown(
                    "This feature allows searching for a specific keyword within the job titles. "
                    "It shows example titles that contain the entered keyword, helping to quickly explore how certain terms are used across the dataset."
                )

            if dataset_choice == "Annotated ACTIVE" and "department" in df_target.columns:
                st.divider()
                st.markdown("### Top TF–IDF keywords per Department (annotated ACTIVE)")
                df_kw_dept = top_keywords_per_label(
                    df_target,
                    text_col="position_norm",
                    label_col="department",
                    top_n=12,
                    stopwords=list(stopwords),
                    min_rows=10
                )
                if df_kw_dept.empty:
                    st.info("Not enough examples per department (need at least 10 titles per class).")
                else:
                    st.dataframe(df_kw_dept, use_container_width=True)
            st.markdown(
                "This table shows the top TF–IDF keywords for each department in the annotated ACTIVE dataset. "
                "It highlights words that are most distinctive for each department, helping to understand typical role-specific vocabulary. "
                "Note: at least 10 titles per department are required to display results."
            )





# -----------------------------
# Page: Career History
# -----------------------------
elif page == "Career History":
    st.markdown(
    "This page explores the career history of annotated profiles, showing the number of jobs per person, active roles, "
    "department changes, and seniority progression over time."
    )
    st.divider()

    st.subheader("Career History EDA (Annotated dataset)")

    if df_ann_jobs.empty:
        st.warning("No annotated profiles available.")
    else:
        jobs_per_person = df_ann_jobs.groupby("person_id").size()

        colA, colB = st.columns(2)
        with colA:
            st.markdown("#### Jobs per profile")
            fig = plt.figure()
            plt.hist(jobs_per_person.values, bins=30)
            plt.title("Distribution of number of jobs per profile")
            plt.xlabel("Jobs per person")
            plt.ylabel("Count")
            st.pyplot(fig, clear_figure=True)

        with colB:
            st.markdown("#### ACTIVE jobs per profile")
            active_per_person = df_ann_jobs[df_ann_jobs["status"] == "ACTIVE"].groupby("person_id").size()
            fig = plt.figure()
            plt.hist(active_per_person.values, bins=20)
            plt.title("Distribution of ACTIVE jobs per profile")
            plt.xlabel("ACTIVE jobs per person")
            plt.ylabel("Count")
            st.pyplot(fig, clear_figure=True)
        st.markdown(
            """
        **Explanation:**  
        - **X-axis:** Shows the number of jobs (or ACTIVE jobs) each profile has.  
        - **Y-axis:** Shows how many profiles have that number of jobs.  

        For example, a bar at X=3 with Y=15 means that 15 profiles have 3 jobs (or 3 ACTIVE jobs) recorded.
        """
        )


        st.divider()

        # Neue Spalten für Department switching und Seniority progression
        colC, colD = st.columns(2)

        with colC:
            st.markdown("#### Department switching")
            dc = compute_department_changes(df_ann_jobs)
            fig = plt.figure()
            plt.hist(dc["dept_changes"], bins=20)
            plt.title("Department changes per profile")
            plt.xlabel("Number of department changes per person")
            plt.ylabel("Number of profiles")
            st.pyplot(fig, clear_figure=True)

        with colD:
            st.markdown("#### Seniority progression")
            si = compute_seniority_increases(df_ann_jobs)
            fig = plt.figure()
            plt.hist(si["sen_increases"], bins=20)
            plt.title("Seniority increases per profile")
            plt.xlabel("Number of seniority increases per person")
            plt.ylabel("Number of profiles")
            st.pyplot(fig, clear_figure=True)
        
        st.markdown(
            """
        **Explanation:**  
        - **X-axis:** Shows the number of changes (either department changes or seniority increases) observed for each profile.  
        - **Y-axis:** Shows how many profiles experienced that number of changes.  

        For example, a bar at X=2 with Y=10 means that 10 profiles had 2 department changes (or 2 seniority increases).
        """
        )
        


# -----------------------------
# Page: Data Quality
# -----------------------------
elif page == "Data Quality":
    st.markdown(
    "This page provides an overview of the dataset's quality, including missing values, duplicate or empty job titles, and the most frequent job titles."
    )
    st.divider()
    st.subheader("Data Quality Checks")

    if df_ann_jobs.empty and df_unl_jobs.empty:
        st.warning("No data loaded.")
    else:
        dataset_choice = st.selectbox(
            "Select dataset",
            ["Annotated", "Unlabeled"],
            index=0
        )
        df_target = df_ann_jobs if dataset_choice == "Annotated" else df_unl_jobs

        if df_target.empty:
            st.warning("Selected dataset is empty.")
        else:
            st.markdown("### Missingness overview")
            st.markdown(
                "This table shows the percentage of missing values in key columns of the selected dataset, helping to identify data quality issues."
            )
            cols = ["position", "status", "department", "seniority", "start_date", "end_date"]
            missing = (df_target[cols].replace("", np.nan).isna().mean() * 100).sort_values(ascending=False)
            st.dataframe(pd.DataFrame({"missing_%": missing.round(2)}), use_container_width=True)
            
            st.divider()

            st.markdown("### Duplicate / empty titles")
            st.markdown(
            "This section shows the proportion of job titles that are empty or duplicated. ")

            empty_titles = (df_target["position_norm"].fillna("") == "").mean() * 100
            dup_rate = df_target["position_norm"].duplicated().mean() * 100
            m1, m2 = st.columns(2)
            with m1:
                st.metric("Empty titles (%)", f"{empty_titles:.2f}%")

            with m2:
                st.metric("Duplicate titles (%)", f"{dup_rate:.2f}%")

            st.divider()


            st.markdown("### Most frequent normalized titles")
            st.markdown("This table shows the job titles that appear most frequently after normalization.")
            top_titles = (
                df_target["position_norm"]
                .replace("", np.nan)
                .dropna()
                .value_counts()
                .head(30)
            )
            df_top_titles = top_titles.rename_axis("title_norm").reset_index(name="count")
            df_top_titles.index = df_top_titles.index + 1  # Index beginnt bei 1
            st.dataframe(df_top_titles, use_container_width=True)

            


# -----------------------------
# Prediction 
# -----------------------------
elif page == "Prediction (Hybrid)":
    st.subheader("Prediction Demo (Hybrid Variante B)")

    try:
        predictor = load_hybrid_model()
        st.success("✅ Hybrid model loaded.")
    except Exception as e:
        st.error("❌ Failed to load hybrid model.")
        st.exception(e)
        st.stop()

    job_title = st.text_input("Job title", "Senior Software Engineer")

    if job_title.strip():
        person_jobs = [{
            "position": job_title,
            "startDate": None,
            "endDate": None,
            "status": "ACTIVE"
        }]

        result = predictor.predict(person_jobs, 0)

        st.subheader("Prediction Result")

        st.metric(
            "Predicted Seniority",
            f"{result['seniority']['label']} ({result['seniority']['confidence']:.2%})"
        )

        st.metric(
            "Predicted Department",
            f"{result['department']['label']} ({result['department']['confidence']:.2%})"
        )

