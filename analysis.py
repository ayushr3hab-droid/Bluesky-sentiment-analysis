# -------------------------------
# 0️⃣ IMPORT LIBRARIES
# -------------------------------
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
from collections import Counter
import re

# -------------------------------
# 1️⃣ LOAD DATA
# -------------------------------
df = pd.read_csv("leeds_student_posts.csv")

print("✅ Data Loaded Successfully")
print("\nColumns:")
print(df.columns)

# -------------------------------
# 2️⃣ CLEAN DATA
# -------------------------------
# Remove empty text
df = df.dropna(subset=['text'])
df = df[df['text'].str.strip() != ""]

print(f"\n✅ Cleaned Data: {len(df)} rows remaining")

# -------------------------------
# 3️⃣ COMPUTE SENTIMENT (if missing)
# -------------------------------
if 'sentiment' not in df.columns or df['sentiment'].isnull().all():
    print("⚡ Computing sentiment using TextBlob...")
    df['sentiment'] = df['text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
else:
    df['sentiment'] = df['sentiment'].astype(float)

print("\n✅ Sentiment column ready")
print(df['sentiment'].describe())

# -------------------------------
# 4️⃣ PLOT SENTIMENT DISTRIBUTION
# -------------------------------
plt.figure(figsize=(8,5))
df['sentiment'].hist(bins=20, color='skyblue', edgecolor='black')
plt.title("Sentiment Distribution of Leeds Student Posts")
plt.xlabel("Sentiment Score")
plt.ylabel("Number of Posts")
plt.grid(False)
plt.show()

# -------------------------------
# 5️⃣ TIME ANALYSIS
# -------------------------------
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df = df.dropna(subset=['timestamp'])

plt.figure(figsize=(8,5))
df.groupby(df['timestamp'].dt.month)['sentiment'].mean().plot(marker='o')
plt.title("Average Sentiment by Month")
plt.xlabel("Month")
plt.ylabel("Average Sentiment")
plt.grid(True)
plt.show()

# -------------------------------
# 6️⃣ MOST COMMON WORDS (TOPICS)
# -------------------------------
text_data = " ".join(df['text'].astype(str))

words = re.findall(r'\b\w+\b', text_data.lower())

# Remove common stopwords
stopwords = set([
    'the','and','is','in','to','of','a','for','on','it','this','that','i','you'
])

filtered_words = [word for word in words if word not in stopwords]

common_words = Counter(filtered_words).most_common(20)

print("\n🔥 Top 20 Common Words in Student Posts:")
for word, count in common_words:
    print(word, ":", count)

# -------------------------------
# 7️⃣ SENTIMENT VS ENGAGEMENT
# -------------------------------
if 'likeCount' in df.columns:
    print("\n📊 Average Likes per Sentiment:")
    print(df.groupby('sentiment')['likeCount'].mean())

# -------------------------------
# 8️⃣ SAVE CLEANED DATA
# -------------------------------
df.to_csv("cleaned_analysis.csv", index=False)
print("\n✅ Cleaned data saved as cleaned_analysis.csv")