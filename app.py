import streamlit as st
import pandas as pd
import re
import pdfplumber
import pickle
import os
from PIL import Image

# ---------------- PAGE CONFIG ----------------
st.markdown("""
<div class="hero">
    <h1>📄 AI Resume Screening & ATS Analytics System</h1>
    <p>Smart Resume Parser • ATS Score • Skill Extraction • ML Job Prediction • Power BI Dashboard</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<style>
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(25px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.hero {
    padding: 35px;
    border-radius: 25px;
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #38bdf8;
    animation: fadeInUp 1s ease;
    margin-bottom: 30px;
}

.feature-card {
    background: #1e293b;
    padding: 22px;
    border-radius: 20px;
    border: 1px solid #334155;
    text-align: center;
    animation: fadeInUp 1.2s ease;
    transition: 0.3s;
}

.feature-card:hover {
    transform: translateY(-8px) scale(1.02);
    border-color: #38bdf8;
    box-shadow: 0 0 18px #38bdf8;
}
</style>
""", unsafe_allow_html=True)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {
    background-color: #0f172a;
}
h1, h2, h3 {
    color: #38bdf8;
}
.stMetric {
    background-color: #1e293b;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #334155;
}
.card {
    background-color: #1e293b;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #334155;
    margin-bottom: 15px;
}
.success-box {
    background-color: #064e3b;
    padding: 15px;
    border-radius: 12px;
    color: white;
}
.warning-box {
    background-color: #78350f;
    padding: 15px;
    border-radius: 12px;
    color: white;
}
.danger-box {
    background-color: #7f1d1d;
    padding: 15px;
    border-radius: 12px;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown("""
<div class="hero">
    <h1>📄 AI Resume Screening & ATS Analytics System</h1>
    <p>Smart Resume Parser • ATS Score • Skill Extraction • ML Job Prediction • Power BI Dashboard</p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <h3>🤖 ML Prediction</h3>
        <p>Predicts best job role from resume text.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <h3>📊 ATS Scoring</h3>
        <p>Calculates resume match percentage instantly.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <h3>📈 Dashboard</h3>
        <p>Power BI analytics with clean visuals.</p>
    </div>
    """, unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🚀 Navigation")
st.sidebar.success("Project Running Successfully")
st.sidebar.info("Upload resume PDF and analyze ATS score.")

# ---------------- LOAD DATA ----------------
skills_df = pd.read_csv("dataset/skills_list.csv")
jobs_df = pd.read_csv("dataset/job_roles.csv")

skills_df.columns = skills_df.columns.str.strip()
jobs_df.columns = jobs_df.columns.str.strip()

skills_list = skills_df["Skill Name"].astype(str).tolist()

# Load ML model safely
model = None
if os.path.exists("model.pkl"):
    model = pickle.load(open("model.pkl", "rb"))

# ---------------- FUNCTIONS ----------------
def clean_resume(text):
    text = re.sub(r"http\S+", " ", str(text))
    text = re.sub(r"@\S+", " ", text)
    text = re.sub(r"[^A-Za-z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower()

def extract_skills(resume_text):
    found_skills = []

    for skill in skills_list:
        if skill.lower() in resume_text:
            found_skills.append(skill)

    return list(set(found_skills))

def extract_pdf_text(uploaded_file):
    resume_text = ""

    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                resume_text += text + " "

    return resume_text

# ---------------- FILE UPLOAD ----------------
st.markdown("## 📤 Upload Resume")

uploaded_file = st.file_uploader(
    "Upload resume in PDF format",
    type=["pdf"]
)

if uploaded_file is not None:

    resume_text = extract_pdf_text(uploaded_file)
    cleaned_resume = clean_resume(resume_text)
    detected_skills = extract_skills(cleaned_resume)

    # ---------------- JOB ROLE ----------------
    st.markdown("## 🎯 Select Target Job Role")

    job_titles = jobs_df["Job Title"].tolist()

    selected_role = st.selectbox(
        "Choose Job Role",
        job_titles
    )

    required_skills = jobs_df.loc[
        jobs_df["Job Title"] == selected_role,
        "Required Skills"
    ].iloc[0].split("|")

    # ---------------- MATCHING ----------------
    detected_skills_lower = [
        skill.lower().strip()
        for skill in detected_skills
    ]

    matched_skills = []
    missing_skills = []

    for skill in required_skills:
        cleaned_skill = skill.lower().strip()

        if cleaned_skill in detected_skills_lower:
            matched_skills.append(skill)
        else:
            missing_skills.append(skill)

    ats_score = (
        len(matched_skills) / len(required_skills)
    ) * 100

    # ---------------- ML PREDICTION ----------------
    predicted_role = "Model not trained"

    if model is not None:
        predicted_role = model.predict([cleaned_resume])[0]

    # ---------------- TOP METRICS ----------------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("ATS Score", f"{ats_score:.2f}%")

    with col2:
        st.metric("Matched Skills", len(matched_skills))

    with col3:
        st.metric("Missing Skills", len(missing_skills))

    st.progress(int(ats_score))

    if ats_score >= 80:
        st.markdown(
            "<div class='success-box'>✅ Excellent Resume Match</div>",
            unsafe_allow_html=True
        )
    elif ats_score >= 60:
        st.markdown(
            "<div class='warning-box'>⚠️ Good Resume, Needs Improvement</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div class='danger-box'>❌ Low ATS Score, Improve Resume</div>",
            unsafe_allow_html=True
        )
        st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #020617, #0f172a, #111827);
        color: white;
    }

    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(25px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes glow {
        0% { box-shadow: 0 0 8px #38bdf8; }
        50% { box-shadow: 0 0 22px #38bdf8; }
        100% { box-shadow: 0 0 8px #38bdf8; }
    }

    .hero {
        padding: 35px;
        border-radius: 25px;
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        animation: fadeInUp 1s ease;
        margin-bottom: 30px;
    }

    .hero h1 {
        font-size: 45px;
        color: #38bdf8;
        margin-bottom: 10px;
    }

    .hero p {
        font-size: 18px;
        color: #cbd5e1;
    }

    .feature-card {
        background: rgba(30, 41, 59, 0.9);
        padding: 22px;
        border-radius: 20px;
        border: 1px solid #334155;
        text-align: center;
        animation: fadeInUp 1.2s ease;
        transition: 0.3s;
    }

    .feature-card:hover {
        transform: translateY(-8px) scale(1.02);
        border-color: #38bdf8;
        animation: glow 1.5s infinite;
    }

    .feature-card h3 {
        color: #38bdf8;
        font-size: 22px;
    }

    .feature-card p {
        color: #cbd5e1;
    }

    .stButton button {
        border-radius: 12px;
        background: linear-gradient(135deg, #06b6d4, #2563eb);
        color: white;
        border: none;
        padding: 10px 20px;
        transition: 0.3s;
    }

    .stButton button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 15px #38bdf8;
    }

    [data-testid="stMetricValue"] {
        color: #38bdf8;
    }

    div[data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.95);
        border: 1px solid #334155;
        padding: 18px;
        border-radius: 20px;
        animation: fadeInUp 0.8s ease;
    }
    </style>
    """, unsafe_allow_html=True)

    # ---------------- TABS ----------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "📄 Resume Details",
        "🧠 Skills Analysis",
        "📊 ATS Analysis",
        "📈 Dashboard Gallery"
    ])

    with tab1:
        st.subheader("Original Resume Text")
        st.text_area("Resume Content", resume_text, height=250)

        st.subheader("Predicted Job Role")
        st.success(predicted_role)

    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("✅ Detected Skills")
            st.write(detected_skills)

        with col2:
            st.subheader("🎯 Required Skills")
            st.write(required_skills)

    with tab3:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("✅ Matched Skills")
            st.write(matched_skills)

        with col2:
            st.subheader("❌ Missing Skills")
            st.write(missing_skills)

        st.subheader("📌 Recommended Skills To Learn")
        for skill in missing_skills:
            st.write("✅", skill)

        st.subheader("📄 Resume Suggestions")

        if ats_score < 60:
            st.write("• Add more relevant technical skills.")
            st.write("• Add strong projects related to the selected job role.")
            st.write("• Add certifications.")
            st.write("• Improve keywords based on job description.")
        elif ats_score < 80:
            st.write("• Add advanced tools and technologies.")
            st.write("• Improve ATS keywords.")
            st.write("• Mention measurable project outcomes.")
        else:
            st.write("• Resume looks strong for this job role.")

        # Save ATS result
        ats_result = {
            "Candidate Name": uploaded_file.name,
            "ATS Score": round(ats_score, 2),
            "Predicted Role": predicted_role,
            "Selected Role": selected_role
        }

        ats_df = pd.DataFrame([ats_result])
        ats_df.to_csv("ats_results.csv", index=False)

        st.subheader("🏆 Candidate Ranking")
        st.dataframe(ats_df)

    with tab4:
        st.subheader("📊 Power BI Dashboard Gallery")

        dashboard_files = [
            "dashboard1.png",
            "dashboard2.png",
            "dashboard3.png",
            "dashboard4.png",
            "dashboard5.png",
            "dashboard6.png",
            "dashboard7.png"
        ]

        for file in dashboard_files:
            image_path = os.path.join("images", file)

            if os.path.exists(image_path):
                image = Image.open(image_path)
                st.image(
                    image,
                    caption=file,
                    use_container_width=True
                )
            else:
                st.warning(f"Image not found: {image_path}")

else:
    st.info("Upload a resume PDF to start analysis.")