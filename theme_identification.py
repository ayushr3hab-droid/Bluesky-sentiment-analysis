# IMPORT LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re
import nltk
import ssl

# FIX SSL ISSUE 
try:
    _create_unverified_https_context = ssl._create_unverified_context
    ssl._create_default_https_context = _create_unverified_https_context
except:
    pass

# DOWNLOAD VADER 
nltk.download('vader_lexicon')

from nltk.sentiment import SentimentIntensityAnalyzer

# LOAD DATA
df = pd.read_csv("leeds_student_posts.csv")

print(" Data Loaded Successfully")
print(df.columns)

#  CLEAN DATA
df = df.dropna(subset=['text'])
df = df[df['text'].str.strip() != ""]

print(f"\n Cleaned Data: {len(df)} rows")

# SENTIMENT ANALYSIS (VADER)
sia = SentimentIntensityAnalyzer()

df['sentiment'] = df['text'].apply(lambda x: sia.polarity_scores(str(x))['compound'])

def label_sentiment(score):
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

df['sentiment_label'] = df['sentiment'].apply(label_sentiment)

print("\n Sentiment Counts:")
print(df['sentiment_label'].value_counts())

#  SENTIMENT GRAPH
plt.figure(figsize=(8,5))
df['sentiment_label'].value_counts().plot(kind='bar')
plt.title("Sentiment Distribution (VADER)")
plt.savefig("sentiment_distribution.png")
print(" Sentiment graph saved")

#  TIME ANALYSIS
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df = df.dropna(subset=['timestamp'])

monthly = df.groupby(df['timestamp'].dt.month)['sentiment'].mean()

plt.figure(figsize=(8,5))
monthly.plot(marker='o')
plt.title("Sentiment Over Time")
plt.savefig("sentiment_time.png")
print(" Time graph saved")

#  TOP WORDS (THEMES)
text_data = " ".join(df['text'].astype(str))

words = re.findall(r'\b\w+\b', text_data.lower())

stopwords = set([
    'the','and','is','in','to','of','a','for','on','it','this','that',
    'i','you','www','com','https','bsky'
])

filtered_words = [w for w in words if w not in stopwords]

top_words = Counter(filtered_words).most_common(20)

print("\n Top 20 Words:")
for w,c in top_words:
    print(w, c)

# THEMES CLASSIFICATION
def detect_theme(text):
    text = text.lower()

    if "exam" in text or "study" in text:
        return "Academics"
    elif "rent" in text or "house" in text:
        return "Housing"
    elif "job" in text or "money" in text:
        return "Finance"
    elif "party" in text or "friend" in text:
        return "Social"
    else:
        return "Other"

df['theme'] = df['text'].apply(detect_theme)

print("\n Theme Counts:")
print(df['theme'].value_counts())

#  SENTIMENT BY THEME
plt.figure(figsize=(8,5))
df.groupby('theme')['sentiment'].mean().plot(kind='bar')
plt.title("Sentiment by Theme")
plt.savefig("theme_sentiment.png")
print(" Theme graph saved")

# ENGAGEMENT ANALYSIS
if 'likeCount' in df.columns:
    print("\n Avg Likes by Sentiment:")
    print(df.groupby('sentiment_label')['likeCount'].mean())

#  SAVE FINAL DATA
df.to_csv("cleaned_final.csv", index=False)

print("\n ALL ANALYSIS COMPLETE")