import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
from wordcloud import WordCloud
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from datetime import datetime
import pytz

# -------------------- Streamlit Config --------------------
st.set_page_config(page_title="🎿 YouTube Wrapped+", layout="wide", initial_sidebar_state="auto")

# -------------------- Clean, Soft Theme --------------------
st.markdown("""
    <style>
    html, body, [class*="css"] {
        background-color: #f5f7fa !important;
        color: #333333 !important;
        font-family: 'Segoe UI', sans-serif;
    }
    h1, h2, h3, h4, h5, h6, .stMetricLabel, .stMetricValue {
        color: #2f3542 !important;
    }
    .block-container { padding-top: 1rem; }
    .stDataFrame, .stTable {
        background-color: #ffffff !important;
        color: #333333 !important;
        border: 1px solid #dee2e6;
        border-radius: 8px;
    }
    .stApp { background-color: #f5f7fa !important; }
    .stMetric {
        border: 1px solid #ced6e0;
        padding: 10px;
        border-radius: 8px;
        background-color: #ffffff;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }
    .stButton>button {
        color: #ffffff !important;
        border-color: #1e90ff;
        background-color: #1e90ff;
    }
    .stButton>button:hover {
        background-color: #1c7ed6;
        color: #ffffff;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #2f3542;'>🎿 YouTube Wrapped+ 2025</h1>", unsafe_allow_html=True)
st.caption("Crafted by Ohad Farooqui using Streamlit, NLP, ML, and Visual Insight")

# -------------------- File Upload --------------------
st.subheader("📂 Upload Your YouTube Watch History JSON")
uploaded_file = st.file_uploader("Upload your 'watch-history.json' file", type=["json"])

if uploaded_file is not None:
    data = json.load(uploaded_file)
    df = pd.json_normalize(data)

    if 'time' not in df.columns:
        st.error("⛔️ 'time' field not found in uploaded file.")
        st.stop()

    def parse_datetime_with_timezone(ts_str):
        try:
            dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            return dt.astimezone(pytz.timezone("Asia/Kolkata"))
        except:
            return None

    df['time'] = df['time'].apply(parse_datetime_with_timezone)
    df = df.dropna(subset=['time'])

    def extract_channel(subtitles):
        if isinstance(subtitles, list) and len(subtitles) > 0 and 'name' in subtitles[0]:
            return subtitles[0]['name']
        else:
            return 'Unlisted/Unavailable'

    df['channel'] = df['subtitles'].apply(extract_channel)
    df['title'] = df['title'].str.replace("Watched ", "", regex=False).str.strip()
    df['date'] = df['time'].dt.date
    df['year'] = df['time'].dt.year
    df['month'] = df['time'].dt.month_name()
    df['day'] = df['time'].dt.day_name()
    df['hour'] = df['time'].dt.hour

    df['polarity'] = df['title'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['mood'] = df['polarity'].apply(lambda x: 'Positive' if x > 0.1 else 'Negative' if x < -0.1 else 'Neutral')

    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(df['title'])

    kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
    df['topic'] = kmeans.fit_predict(X)
    terms = []
    for i in range(6):
        center = kmeans.cluster_centers_[i]
        top_words = np.argsort(center)[-5:][::-1]
        words = [vectorizer.get_feature_names_out()[j] for j in top_words]
        terms.append(", ".join(words))
    df['topic_label'] = df['topic'].apply(lambda x: terms[x])

    # Sidebar Filters
    st.sidebar.title("🔍 Filters")
    selected_year = st.sidebar.selectbox("Filter by Year", options=["All"] + sorted(df['year'].unique().tolist()))
    selected_channel = st.sidebar.selectbox("Filter by Channel", options=["All"] + sorted(df['channel'].unique().tolist()))

    if selected_year != "All":
        df = df[df['year'] == selected_year]
    if selected_channel != "All":
        df = df[df['channel'] == selected_channel]

    # Stats
    st.subheader("📊 Your Watch Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Videos", len(df))
    c2.metric("Unique Channels", df['channel'].nunique())
    c3.metric("Most Watched Day", df['date'].value_counts().idxmax().strftime("%A, %d %b %Y"))
    c4.metric("Peak Hour", f"{df['hour'].value_counts().idxmax()}:00")

    # Charts
    st.subheader("🎮 Most Watched Channels")
    st.plotly_chart(px.bar(df['channel'].value_counts().head(10).sort_values(), orientation='h', color_discrete_sequence=['#1e90ff']), use_container_width=True)

    st.subheader("🎞️ Most Watched Video Titles")
    st.plotly_chart(px.bar(df['title'].value_counts().head(10).sort_values(), orientation='h', color_discrete_sequence=['#ffa502']), use_container_width=True)

    # Word Cloud
    st.subheader("☁️ Word Cloud of Watched Titles")
    wc = WordCloud(width=800, height=300, background_color='#f5f7fa', colormap='Blues', max_words=100).generate(" ".join(df['title']))
    fig_wc, ax_wc = plt.subplots(figsize=(10, 3))
    ax_wc.imshow(wc, interpolation='bilinear')
    ax_wc.axis('off')
    st.pyplot(fig_wc)

    # Mood Analysis
    st.subheader("💬 Mood Breakdown")
    moods = df['mood'].value_counts()
    fig_mood, ax_mood = plt.subplots(figsize=(3, 3))
    colors = ['#2ed573', '#ffa502', '#70a1ff']
    ax_mood.pie(moods, labels=moods.index, colors=colors, startangle=90)
    ax_mood.set_title("Mood")
    ax_mood.set_ylabel("")
    st.pyplot(fig_mood)

    # Topics
    st.subheader("🧠 Topic Clusters from Video Titles")
    topic_df = df['topic_label'].value_counts().reset_index()
    topic_df.columns = ['Topic Keywords', 'Count']
    st.dataframe(topic_df, use_container_width=True)

    # Heatmap
    st.subheader("🗕️ Activity Heatmap (Days vs Hours)")
    heat_data = df.groupby(['day', 'hour']).size().unstack().reindex([
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    )
    fig_heat, ax_heat = plt.subplots(figsize=(6, 2.5))
    sns.heatmap(heat_data.fillna(0), cmap="crest", ax=ax_heat, linewidths=0.5, linecolor='white', cbar=False)
    ax_heat.set_title("Watch Intensity")
    st.pyplot(fig_heat)

    # Recommendations
    st.subheader("🤖 Suggested Videos Based on Your History")
    count_vect = CountVectorizer(stop_words='english')
    count_matrix = count_vect.fit_transform(df['title'])
    sim_scores = cosine_similarity(count_matrix)
    sim_df = pd.DataFrame(sim_scores, index=df.index, columns=df.index)
    top_title_index = df['title'].value_counts().index[0]
    top_idx = df[df['title'] == top_title_index].index[0]
    similar_scores = sim_df.loc[top_idx].sort_values(ascending=False)
    similar_indices = similar_scores.index[1:6]
    for idx in similar_indices:
        st.markdown(f"- 🔗 {df.loc[idx, 'title']}")

    # Embedded Links
    st.subheader("🔗 Recently Watched Video Links")
    df_links = df[df['title'].str.contains("https://www.youtube.com/watch")].head(10)
    if not df_links.empty:
        for idx, row in df_links.iterrows():
            st.markdown(f"[{row['title']}]({row['title']})")
    else:
        st.info("No direct video links found in watch history titles.")

    # Raw Data & Export
    with st.expander("📄 Raw Watch Data"):
        st.dataframe(df[['time', 'title', 'channel', 'mood', 'topic_label']], use_container_width=True)

    csv = df[['time', 'title', 'channel', 'mood', 'topic_label']].to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "youtube_wrapped_data.csv", "text/csv", key='download-csv')

    # Footer
    st.markdown("""
    ---
    📍 Follow me on [Instagram](https://www.instagram.com/ohadfarooqui) | [LinkedIn](https://www.linkedin.com/in/ohadfarooqui/)  
    💡 Project by Ohad Farooqui — Made with Streamlit, Python, and 💙
    """)

else:
    st.info("👆 Upload your watch-history.json to get started!")
