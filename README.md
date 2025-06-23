
# 🎧 YouTube Wrapped+ (2025 Edition)

**YouTube Wrapped+** is a beautiful, data-driven web app built with Streamlit that visualizes your YouTube watch history — offering mood insights, channel breakdowns, word clouds, topic detection, and smart recommendations.

🚀 Crafted with love using **Python**, **NLP**, **Machine Learning**, and **Streamlit**.

---

## 📸 Live Demo

> **Coming Soon:** [Streamlit Cloud Link Here](https://share.streamlit.io/yourusername/youtube-wrapped-streamlit)

---

## 📂 What You Need

1. Your `watch-history.json` from [Google Takeout](https://takeout.google.com/).
2. Python 3.8 or above.
3. A few Python packages (listed below).

---

## 🧠 Features

- 📊 Total stats: most-watched day, channels, and time slots  
- 🧠 NLP-based **mood detection** using `TextBlob`  
- 🔍 **Topic detection** using TF-IDF + KMeans Clustering  
- 🌈 Beautiful **word clouds**  
- 📅 Heatmaps for daily + hourly activity  
- 🤖 **Smart recommendations** using cosine similarity  
- 🔗 Embedded YouTube video links  
- 📁 Downloadable CSV of your data  
- 🔌 Dynamic filtering by **year** and **channel**

---

## 📦 Installation

Clone the repo:

```bash
git clone https://github.com/yourusername/youtube-wrapped-streamlit.git
cd youtube-wrapped-streamlit
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run streamlit_app.py
```

---

## 📝 How To Export YouTube Watch History

1. Go to [Google Takeout](https://takeout.google.com/).
2. Deselect all > Select only **YouTube and YouTube Music**.
3. Choose **JSON format**, and download.
4. Extract the zip and find the `watch-history.json` file.
5. Upload it in the Streamlit app.

---

## 📸 Screenshots

| Watch Summary | Word Cloud | Mood Pie |
|---------------|------------|----------|
| ![Summary](assets/summary.png) | ![WordCloud](assets/wordcloud.png) | ![Mood](assets/mood.png) |

---

## 💻 Built With

- [Streamlit](https://streamlit.io/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [scikit-learn](https://scikit-learn.org/)
- [TextBlob](https://textblob.readthedocs.io/)
- [Plotly](https://plotly.com/python/)
- [WordCloud](https://github.com/amueller/word_cloud)

---

## 🙋‍♂️ About Me

**Ohad Farooqui**  
📍 Data Scientist | Python Enthusiast | Visual Thinker  
📷 [Instagram](https://www.instagram.com/ohadfarooqui)  
🔗 [LinkedIn](https://www.linkedin.com/in/ohadfarooqui)

---

## 📃 License

MIT License. Free to use, improve, and share.

---

## 💡 Bonus Ideas

- Deploy on **Streamlit Cloud** for public sharing  
- Add **search by keyword or title**  
- Integrate with **Spotify/Netflix Wrapped-style animation**

---

> Made with 💙 by Ohad Farooqui
