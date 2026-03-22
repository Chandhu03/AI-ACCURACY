import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

st.set_page_config(page_title="AI Accuracy", layout="wide", page_icon="🤖")

st.markdown("""
<style>
    #MainMenu, footer, header {visibility: hidden;}
    .stApp { background-color: #f5f5f5; color: #1a1a1a; }
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
        padding-top: 2rem;
        min-width: 260px !important;
        max-width: 260px !important;
    }
    [data-testid="collapsedControl"] { display: none !important; }
    [data-testid="stChatMessage"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.5rem;
        color: #1a1a1a;
    }
    .main .block-container { padding: 2rem 3rem; max-width: 900px; }
    .title-area { text-align: center; padding: 3rem 0 2rem 0; }
    .title-area h1 { font-size: 2rem; font-weight: 700; color: #1a1a1a; }
    .title-area p { color: #666; font-size: 0.9rem; margin-top: 0.5rem; }
    .tag {
        display: inline-block;
        background: #f0f0f0;
        border: 1px solid #ddd;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.75rem;
        color: #444;
        margin: 2px;
    }
    hr { border-color: #e0e0e0; }
    .stMarkdown p, .stMarkdown li { color: #1a1a1a !important; }
    h3 { color: #1a1a1a !important; }
    a { color: #1a56db; }
</style>
""", unsafe_allow_html=True)

# ---- File path: same folder as this script ----
FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Stage_2_data.xlsx")

# ---- All provinces for the dropdown ----
PROVINCES = [
    "Alberta (AB)",
    "British Columbia (BC)",
    "Manitoba (MB)",
    "New Brunswick (NB)",
    "Newfoundland and Labrador (NL)",
    "Nova Scotia (NS)",
    "Ontario (ON)",
    "Prince Edward Island (PE)",
    "Quebec (QC)",
    "Saskatchewan (SK)",
    "Northwest Territories (NT)",
    "Nunavut (NU)",
    "Yukon (YT)",
]

# ---- Provinces that have actual data ----
PROVINCES_WITH_DATA = ["Ontario (ON)", "Newfoundland and Labrador (NL)"]

# ---- Government URLs ----
PROVINCE_URLS = {
    "Ontario (ON)": "https://www.ontario.ca",
    "Newfoundland and Labrador (NL)": "https://www.gov.nl.ca",
}

# ---- Province aliases for detecting from query text ----
PROVINCE_ALIASES = {
    "ontario": "Ontario (ON)",
    "on": "Ontario (ON)",
    "newfoundland": "Newfoundland and Labrador (NL)",
    "labrador": "Newfoundland and Labrador (NL)",
    "nl": "Newfoundland and Labrador (NL)",
    "nfl": "Newfoundland and Labrador (NL)",
    "alberta": "Alberta (AB)",
    "ab": "Alberta (AB)",
    "british columbia": "British Columbia (BC)",
    "bc": "British Columbia (BC)",
    "manitoba": "Manitoba (MB)",
    "mb": "Manitoba (MB)",
    "new brunswick": "New Brunswick (NB)",
    "nb": "New Brunswick (NB)",
    "nova scotia": "Nova Scotia (NS)",
    "ns": "Nova Scotia (NS)",
    "prince edward island": "Prince Edward Island (PE)",
    "pei": "Prince Edward Island (PE)",
    "pe": "Prince Edward Island (PE)",
    "quebec": "Quebec (QC)",
    "qc": "Quebec (QC)",
    "saskatchewan": "Saskatchewan (SK)",
    "sk": "Saskatchewan (SK)",
    "northwest territories": "Northwest Territories (NT)",
    "nt": "Northwest Territories (NT)",
    "nwt": "Northwest Territories (NT)",
    "nunavut": "Nunavut (NU)",
    "nu": "Nunavut (NU)",
    "yukon": "Yukon (YT)",
    "yt": "Yukon (YT)",
}


# ---- Load data from all 4 sheets ----
@st.cache_data
def load_data():
    # --- Ontario Govt ---
    df_on_govt = pd.read_excel(FILE_PATH, sheet_name="Ontario - Govt", header=0)
    df_on_govt.columns = df_on_govt.columns.str.strip()
    df_on_govt["Type"] = "Government"
    df_on_govt["Province"] = "Ontario (ON)"

    # --- Ontario NGO ---
    df_on_ngo = pd.read_excel(FILE_PATH, sheet_name="Ontario NGO", header=0)
    df_on_ngo.columns = df_on_ngo.columns.str.strip()
    df_on_ngo["Type"] = "NGO / Private"
    df_on_ngo["Province"] = "Ontario (ON)"

    # --- NFL Govt ---
    df_nl_govt = pd.read_excel(FILE_PATH, sheet_name="NFL - Govt", header=0)
    df_nl_govt.columns = df_nl_govt.columns.str.strip()
    df_nl_govt["Type"] = "Government"
    df_nl_govt["Province"] = "Newfoundland and Labrador (NL)"

    # --- NFL NGO ---
    df_nl_ngo = pd.read_excel(FILE_PATH, sheet_name="NFL - NGO", header=0)
    df_nl_ngo.columns = df_nl_ngo.columns.str.strip()
    df_nl_ngo["Type"] = "NGO / Private"
    df_nl_ngo["Province"] = "Newfoundland and Labrador (NL)"

    # Unify columns
    cols = [
        "Organization / Program", "Primary Sector", "Type", "Province",
        "Website", "Primary Mandate", "Key Services & Functions",
        "Eligibility / Who it's for",
    ]

    frames = []
    for frame in [df_on_govt, df_on_ngo, df_nl_govt, df_nl_ngo]:
        for c in cols:
            if c not in frame.columns:
                frame[c] = None
        frames.append(frame[cols])

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["Organization / Program"])
    combined = combined[combined["Organization / Program"].str.strip() != ""]
    combined = combined[combined["Organization / Program"] != "Organization / Program"]
    # Remove stray rows like backtick
    combined = combined[combined["Organization / Program"].str.len() > 2]
    combined = combined.reset_index(drop=True)
    return combined


df = load_data()


# ---- Build TF-IDF search index ----
@st.cache_data
def build_search_index(_df):
    docs = []
    for _, row in _df.iterrows():
        parts = [
            str(row.get("Organization / Program", "")),
            str(row.get("Primary Sector", "")),
            str(row.get("Primary Mandate", "")),
            str(row.get("Key Services & Functions", "")),
            str(row.get("Eligibility / Who it's for", "")),
        ]
        docs.append(" ".join(p for p in parts if p and p != "nan"))
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b\w+\b",
    )
    X = vectorizer.fit_transform(docs)
    return vectorizer, X


vectorizer, X = build_search_index(df)


# ---- Query expansion ----
QUERY_EXPANSIONS = {
    "healthcare": "healthcare health care medical hospital",
    "wellbeing": "wellbeing well being wellness mental",
    "nonprofit": "nonprofit non profit ngo charity",
    "startup": "startup start up startups innovation",
    "jobs": "jobs employment work career hiring",
    "doctor": "doctor physician medical health care primary",
    "housing": "housing home shelter affordable rent",
    "immigrant": "immigrant newcomer refugee settlement",
    "school": "school education college university student",
}


def expand_query(query):
    q = query.lower()
    for term, expansion in QUERY_EXPANSIONS.items():
        if term in q:
            q = q + " " + expansion
    return q


# ---- Search function ----
def search_orgs(query, province=None, top_n=10):
    expanded = expand_query(query)
    q_vec = vectorizer.transform([expanded])
    similarity = cosine_similarity(q_vec, X).flatten()

    df_results = df.copy()
    df_results["score"] = similarity

    if province:
        df_results = df_results[df_results["Province"] == province]

    df_results = df_results[df_results["score"] > 0.05]
    df_results = df_results.sort_values("score", ascending=False).head(top_n)
    return df_results


# ---- Detect province from query text ----
def detect_province(query):
    q = query.lower()
    # Check longer aliases first to avoid partial matches
    for alias in sorted(PROVINCE_ALIASES.keys(), key=len, reverse=True):
        if alias in q:
            return PROVINCE_ALIASES[alias]
    return None


# ---- Sidebar (same layout as original) ----
with st.sidebar:
    st.markdown("### 🤖 AI Accuracy")
    st.markdown("---")
    st.markdown("""
**About**

AI Accuracy is your go-to research tool for reliable, verified information on government
programs and services across all Canadian provinces and territories. We cut through the noise
so you always get credible, up-to-date data — no guesswork, no misinformation.
    """)
    st.markdown("---")
    st.markdown("**Select Province**")
    selected_province = st.selectbox(
        "",
        options=[""] + PROVINCES,
        format_func=lambda x: "Choose a province..." if x == "" else x,
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**AI Models tracked**")
    for m in ["Gemini", "ChatGPT", "Perplexity", "Claude"]:
        st.markdown(f"<span class='tag'>● {m}</span>", unsafe_allow_html=True)


# ---- Main Chat (same layout as original) ----
if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.markdown("""
    <div class='title-area'>
        <h1>AI Accuracy Research</h1>
        <p>Ask about Canadian government programs by province</p>
        <br/>
        <span class='tag'>💡 "healthcare services in Ontario"</span>
        <span class='tag'>💡 "mental health support in NL"</span>
        <span class='tag'>💡 "employment programs in Newfoundland"</span>
    </div>
    """, unsafe_allow_html=True)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask about Canadian government programs...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Determine which province
    if selected_province:
        target_province = selected_province
    else:
        target_province = detect_province(user_input)

    if not target_province:
        response = (
            "⚠️ Please select a province from the sidebar dropdown, or mention one "
            "in your question (e.g. *Ontario*, *Newfoundland*, *NL*, *BC*)."
        )
    elif target_province not in PROVINCES_WITH_DATA:
        # Province selected but no data available
        prov_name = target_province.split(" (")[0]
        response = (
            f"**Your query:** *{user_input}*\n\n---\n"
            f"\n### 📍 {target_province}\n"
            f"_N/A — Data for {prov_name} is not yet available. "
            f"Currently, only **Ontario (ON)** and **Newfoundland and Labrador (NL)** have data._\n"
        )
    else:
        # Province has data — search
        results = search_orgs(user_input, province=target_province)

        if results.empty:
            response = (
                f"**Your query:** *{user_input}*\n\n---\n"
                f"\n### 📍 {target_province}\n"
                f"_No matching programs found. Try being more specific — "
                f"e.g. *healthcare*, *mental health*, *employment*, *education*._\n"
            )
        else:
            top_sector = results.iloc[0]["Primary Sector"]
            response = f"**Category:** {top_sector}\n\n**Your query:** *{user_input}*\n\n---\n"
            url = PROVINCE_URLS.get(target_province, "")
            response += f"\n### 📍 {target_province}\n"

            for _, row in results.iterrows():
                name = row["Organization / Program"]
                org_type = row["Type"]
                mandate = row.get("Primary Mandate", "")
                services = row.get("Key Services & Functions", "")
                eligibility = row.get("Eligibility / Who it's for", "")
                website = row.get("Website", "")

                type_label = "🏛️ Gov" if org_type == "Government" else "🤝 NGO"
                response += f"\n- **{name}** ({type_label})\n"

                if mandate and str(mandate) != "nan":
                    response += f"  - *Mandate:* {mandate}\n"
                if services and str(services) != "nan":
                    response += f"  - *Services:* {services}\n"
                if eligibility and str(eligibility) != "nan":
                    response += f"  - *Eligibility:* {eligibility}\n"
                if website and str(website) != "nan":
                    w = str(website).strip()
                    if not w.startswith("http"):
                        w = "https://" + w
                    response += f"  - 🔗 [{w}]({w})\n"

            if url:
                prov_name = target_province.split(" (")[0]
                response += (
                    f"\n🔗 For more information, visit the official {prov_name} "
                    f"government site: [{url}]({url})\n"
                )

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
