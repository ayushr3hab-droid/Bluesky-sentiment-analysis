# MSc Dissertation — Streamlit NLP Research Interface
# Student Discourse Analysis on Bluesky

import streamlit as st
import pandas as pd
import plotly.express as px
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# PAGE CONFIG

st.set_page_config(
    page_title="MSc Dissertation NLP Interface",
    page_icon="🧠",
    layout="wide"
)

# CLASSIFIERS

def classify_theme(text):
    t = str(text).lower()

    if any(w in t for w in ["rent", "flat", "housing", "accommodation", "landlord", "room"]):
        return "Housing"
    elif any(w in t for w in ["loan", "finance", "money", "fees", "cost", "job"]):
        return "Finance"
    elif any(w in t for w in ["friend", "party", "social", "event", "meet"]):
        return "Social"
    elif any(w in t for w in ["exam", "lecture", "study", "university", "assignment"]):
        return "Academic"
    else:
        return "Other"


def classify_content(text):

    t = str(text).lower()

    info_triggers = [
        'due', 'deadline', 'payment', 'pay',
        'announcement', 'update', 'notice',
        'timetable', 'schedule',
        'next month', 'this month', 'on the', 'by'
    ]

    opinion_triggers = [
        'hate', 'love', 'terrible', 'amazing',
        'annoying', 'frustrated', 'worst', 'great'
    ]

    info_score = sum(word in t for word in info_triggers)
    opinion_score = sum(word in t for word in opinion_triggers)

    #  rule
    if info_score >= 1 and opinion_score == 0:
        return "Informational"
    elif opinion_score > info_score:
        return "Opinion-based"
    else:
        return "Informational"


# LOAD DATA 

@st.cache_data
def load_data():

    df = pd.read_csv("leeds_student_posts.csv")

    df = df.fillna("")

    # ensure required columns
    if "text" not in df.columns:
        df["text"] = ""

    if "likeCount" not in df.columns:
        df["likeCount"] = 0

    df["likeCount"] = pd.to_numeric(df["likeCount"], errors="coerce").fillna(0)

    df["timestamp"] = pd.to_datetime(df.get("timestamp", ""), errors="coerce")

    # SENTIMENT (VADER)

    analyzer = SentimentIntensityAnalyzer()

    def get_sentiment(text):
        score = analyzer.polarity_scores(str(text))["compound"]

        if score >= 0.05:
            label = "Positive"
        elif score <= -0.05:
            label = "Negative"
        else:
            label = "Neutral"

        return pd.Series([score, label])

    df[["sentiment_score", "sentiment_label"]] = df["text"].apply(get_sentiment)

    # THEME + CONTENT TYPE 

    df["Theme"] = df["text"].apply(classify_theme)
    df["Content_Type"] = df["text"].apply(classify_content)

    return df


# LOAD DATA

df = load_data()

# SIDEBAR FILTERS

st.sidebar.title("🧠 NLP Interface")

theme_filter = st.sidebar.multiselect(
    "Theme",
    df["Theme"].unique(),
    default=df["Theme"].unique()
)

sentiment_filter = st.sidebar.multiselect(
    "Sentiment",
    df["sentiment_label"].unique(),
    default=df["sentiment_label"].unique()
)

content_filter = st.sidebar.multiselect(
    "Content Type",
    df["Content_Type"].unique(),
    default=df["Content_Type"].unique()
)

# FILTER DATA

filtered = df[
    (df["Theme"].isin(theme_filter)) &
    (df["sentiment_label"].isin(sentiment_filter)) &
    (df["Content_Type"].isin(content_filter))
]

# HEADER

st.title("🧠 MSc Dissertation NLP Interface")

# KPI CARDS

c1, c2, c3, c4 = st.columns(4)

c1.metric("Posts", len(filtered))
c2.metric("Avg Sentiment", round(filtered["sentiment_score"].mean(), 3))
c3.metric("Likes", int(filtered["likeCount"].sum()))
c4.metric("Themes", filtered["Theme"].nunique())

st.divider()

# TABS

tab1, tab2, tab3, tab4 = st.tabs([
    "Sentiment",
    "Themes",
    "Engagement",
    "Live Lab"
])

# SENTIMENT

with tab1:

    fig1 = px.histogram(filtered, x="sentiment_label", color="sentiment_label")
    st.plotly_chart(fig1, use_container_width=True)

# THEMES

with tab2:

    fig2 = px.pie(filtered, names="Theme")
    st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(filtered[["text", "Theme", "Content_Type", "sentiment_label"]])

# ENGAGEMENT

with tab3:

    fig3 = px.bar(
        filtered.groupby("sentiment_label")["likeCount"].mean().reset_index(),
        x="sentiment_label",
        y="likeCount"
    )
    st.plotly_chart(fig3, use_container_width=True)

# LIVE TEST LAB

with tab4:

    user_text = st.text_area("Enter text")

    if user_text:

        analyzer = SentimentIntensityAnalyzer()
        score = analyzer.polarity_scores(user_text)["compound"]

        label = (
            "Positive" if score >= 0.05 else
            "Negative" if score <= -0.05 else
            "Neutral"
        )

        st.metric("Score", round(score, 3))
        st.metric("Sentiment", label)
        st.metric("Theme", classify_theme(user_text))
        st.metric("Content Type", classify_content(user_text))

        st.info("Demonstrates lexical limitation of VADER in mixed-content discourse")

# WORDCLOUD

st.divider()

text = " ".join(filtered["text"].astype(str))
text = re.sub(r"http\S+", "", text)

wc = WordCloud(width=1200, height=400, background_color="white").generate(text)

fig, ax = plt.subplots()
ax.imshow(wc, interpolation="bilinear")
ax.axis("off")

st.pyplot(fig)