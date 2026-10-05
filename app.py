import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Cognitive Student Academic Advisor",
    page_icon="🧠",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🧠 Cognitive Student Academic Advisor")

st.write(
    """
    A personalized academic decision-support system using
    Machine Learning, student performance analysis, and
    knowledge-based reasoning.
    """
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv("data/students.csv")

    return df


df = load_data()


# ============================================================
# CREATE PERFORMANCE SCORE
# ============================================================

df["Performance"] = (
    df["Python"]
    + df["DBMS"]
    + df["Mathematics"]
    + df["MachineLearning"]
) / 4


# ============================================================
# MACHINE LEARNING MODEL
# ============================================================

features = [
    "Python",
    "DBMS",
    "Mathematics",
    "MachineLearning",
    "Attendance",
    "StudyHours"
]

X = df[features]
y = df["Performance"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("🎓 Student Profile")

python_mark = st.sidebar.slider(
    "Python Marks",
    0,
    100,
    80
)

dbms_mark = st.sidebar.slider(
    "DBMS Marks",
    0,
    100,
    55
)

math_mark = st.sidebar.slider(
    "Mathematics Marks",
    0,
    100,
    48
)

ml_mark = st.sidebar.slider(
    "Machine Learning Marks",
    0,
    100,
    62
)

attendance = st.sidebar.slider(
    "Attendance (%)",
    0,
    100,
    72
)

study_hours = st.sidebar.slider(
    "Daily Study Hours",
    0,
    12,
    3
)

interest = st.sidebar.selectbox(
    "Area of Interest",
    [
        "Data Science",
        "Artificial Intelligence",
        "Machine Learning",
        "Software Development",
        "Web Development",
        "Cloud Computing"
    ]
)


# ============================================================
# STUDENT DATA
# ============================================================

student_data = pd.DataFrame({
    "Python": [python_mark],
    "DBMS": [dbms_mark],
    "Mathematics": [math_mark],
    "MachineLearning": [ml_mark],
    "Attendance": [attendance],
    "StudyHours": [study_hours]
})


# ============================================================
# PREDICTION
# ============================================================

predicted_score = model.predict(student_data)[0]

predicted_score = max(
    0,
    min(100, predicted_score)
)


# ============================================================
# PERFORMANCE LEVEL
# ============================================================

if predicted_score >= 75:

    performance_level = "Excellent 🟢"

elif predicted_score >= 60:

    performance_level = "Good 🟡"

elif predicted_score >= 50:

    performance_level = "Average 🟠"

else:

    performance_level = "Needs Improvement 🔴"


# ============================================================
# DASHBOARD METRICS
# ============================================================

st.subheader("📊 Academic Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Predicted Performance",
        f"{predicted_score:.1f}%"
    )

with col2:
    st.metric(
        "Attendance",
        f"{attendance}%"
    )

with col3:
    st.metric(
        "Daily Study",
        f"{study_hours} hrs"
    )

with col4:
    st.metric(
        "Performance Level",
        performance_level
    )


# ============================================================
# SUBJECT PERFORMANCE
# ============================================================

st.subheader("📚 Subject Performance")

subject_data = pd.DataFrame({
    "Subject": [
        "Python",
        "DBMS",
        "Mathematics",
        "Machine Learning"
    ],

    "Marks": [
        python_mark,
        dbms_mark,
        math_mark,
        ml_mark
    ]
})

fig = px.bar(
    subject_data,
    x="Subject",
    y="Marks",
    text="Marks",
    range_y=[0, 100],
    title="Subject-wise Academic Performance"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# REASONING ENGINE
# ============================================================

st.subheader("🧠 Cognitive Academic Insights")

marks = {
    "Python": python_mark,
    "DBMS": dbms_mark,
    "Mathematics": math_mark,
    "Machine Learning": ml_mark
}

weak_subjects = []

strong_subjects = []

for subject, mark in marks.items():

    if mark < 60:

        weak_subjects.append(subject)

    elif mark >= 75:

        strong_subjects.append(subject)


# ============================================================
# WEAK SUBJECT ANALYSIS
# ============================================================

if weak_subjects:

    st.warning(
        "⚠️ Weak Subjects: "
        + ", ".join(weak_subjects)
    )

else:

    st.success(
        "✅ No major weak subjects detected."
    )


# ============================================================
# STRONG SUBJECT ANALYSIS
# ============================================================

if strong_subjects:

    st.success(
        "💪 Strong Subjects: "
        + ", ".join(strong_subjects)
    )


# ============================================================
# ATTENDANCE REASONING
# ============================================================

if attendance < 75:

    st.warning(
        "🕐 Attendance Alert: "
        "Your attendance is below 75%. "
        "Improving attendance may help your academic performance."
    )

else:

    st.success(
        "✅ Attendance is at or above 75%."
    )


# ============================================================
# PERSONALIZED RECOMMENDATIONS
# ============================================================

st.subheader("🎯 Personalized Recommendations")

recommendations = []


if math_mark < 60:

    recommendations.append(
        "📐 Strengthen Mathematics fundamentals, "
        "especially statistics, probability and algebra."
    )


if dbms_mark < 60:

    recommendations.append(
        "🗄️ Spend additional study time on DBMS, "
        "SQL queries, normalization and database design."
    )


if python_mark >= 75 and interest == "Data Science":

    recommendations.append(
        "🐍 Your Python foundation is strong. "
        "You can progress toward Pandas, NumPy, "
        "Statistics and Machine Learning."
    )


if python_mark >= 75 and interest == "Artificial Intelligence":

    recommendations.append(
        "🤖 Your Python foundation supports further "
        "learning in Machine Learning and Artificial Intelligence."
    )


if attendance < 75:

    recommendations.append(
        "📅 Improve attendance and maintain consistent "
        "participation in classes."
    )


if study_hours < 3:

    recommendations.append(
        "📖 Consider increasing your daily study time "
        "to at least 3 hours."
    )


if not recommendations:

    recommendations.append(
        "🌟 Your academic profile is relatively balanced. "
        "Continue consistent study and focus on advanced topics."
    )


for recommendation in recommendations:

    st.write(
        f"• {recommendation}"
    )


# ============================================================
# KNOWLEDGE BASE
# ============================================================

knowledge_base = {

    "Data Science": {
        "prerequisites": [
            "Python",
            "Statistics",
            "Mathematics",
            "Machine Learning"
        ]
    },

    "Artificial Intelligence": {
        "prerequisites": [
            "Python",
            "Mathematics",
            "Machine Learning"
        ]
    },

    "Machine Learning": {
        "prerequisites": [
            "Python",
            "Statistics",
            "Mathematics"
        ]
    },

    "Software Development": {
        "prerequisites": [
            "Programming",
            "DBMS",
            "Data Structures"
        ]
    },

    "Web Development": {
        "prerequisites": [
            "HTML",
            "CSS",
            "JavaScript",
            "DBMS"
        ]
    },

    "Cloud Computing": {
        "prerequisites": [
            "Networking",
            "Linux",
            "Programming"
        ]
    }
}


# ============================================================
# CAREER / LEARNING PATH
# ============================================================

st.subheader("🗺️ Personalized Learning Path")

if interest in knowledge_base:

    prerequisites = knowledge_base[interest]["prerequisites"]

    st.write(
        f"Based on your interest in **{interest}**, "
        "the recommended knowledge areas are:"
    )

    for item in prerequisites:

        st.write(
            f"➡️ {item}"
        )


# ============================================================
# PERSONALIZED PATH FOR DATA SCIENCE
# ============================================================

if interest == "Data Science":

    st.markdown(
        """
        ### Recommended Path

        **Python**
        ↓

        **Statistics**
        ↓

        **Mathematics**
        ↓

        **Machine Learning**
        ↓

        **Data Science Projects**
        """
    )


# ============================================================
# PERSONALIZED PATH FOR AI
# ============================================================

elif interest == "Artificial Intelligence":

    st.markdown(
        """
        ### Recommended Path

        **Python**
        ↓

        **Mathematics**
        ↓

        **Machine Learning**
        ↓

        **Deep Learning**
        ↓

        **Artificial Intelligence Projects**
        """
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.subheader("🤖 Machine Learning Model")

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Mean Absolute Error",
        f"{mae:.2f}"
    )

with col2:

    st.metric(
        "R² Score",
        f"{r2:.2f}"
    )


# ============================================================
# NLP-STYLE ACADEMIC ADVISOR
# ============================================================

st.subheader("💬 Ask Your Academic Advisor")

question = st.text_input(
    "Ask a question about your academic performance:"
)


if question:

    q = question.lower()

    if "math" in q:

        answer = (
            "Your Mathematics score is "
            f"{math_mark}%. "
            "Focus on algebra, statistics and probability "
            "before moving to advanced Machine Learning."
        )

    elif "python" in q:

        answer = (
            f"Your Python score is {python_mark}%. "
            "Python is particularly useful for Data Science, "
            "Machine Learning and Artificial Intelligence."
        )

    elif "dbms" in q or "database" in q:

        answer = (
            f"Your DBMS score is {dbms_mark}%. "
            "Focus on SQL, normalization, relationships "
            "and database design."
        )

    elif "attendance" in q:

        answer = (
            f"Your attendance is {attendance}%. "
            "Maintaining regular attendance can support "
            "consistent academic progress."
        )

    elif "career" in q:

        answer = (
            f"Since your interest is {interest}, "
            "you should follow the recommended learning "
            "path shown above."
        )

    elif "weak" in q:

        if weak_subjects:

            answer = (
                "Your weaker subjects are: "
                + ", ".join(weak_subjects)
                + ". Focus additional study time on these areas."
            )

        else:

            answer = (
                "No major weak subjects were detected."
            )

    else:

        answer = (
            "Based on your academic profile, focus on your "
            "weak subjects, maintain attendance and build "
            "skills related to your selected career interest."
        )

    st.info(
        f"🧠 Academic Advisor:\n\n{answer}"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Cognitive Student Academic Advisor | "
    "Machine Learning + Knowledge-Based Reasoning + NLP"
)