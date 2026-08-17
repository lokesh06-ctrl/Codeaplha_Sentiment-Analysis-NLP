import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os
import re


# ==========================================
# 1. Create output folder
# ==========================================

os.makedirs("output", exist_ok=True)


# ==========================================
# 2. Load reviews
# ==========================================

input_file = "data/test_reviews.csv"

df = pd.read_csv(input_file)

print("\nReviews loaded:")
print(df)


# ==========================================
# 3. Sentiment classification
# ==========================================

def classify_sentiment(text):

    text = str(text).lower().strip()

    neutral_phrases = [
        "okay",
        "ok",
        "average",
        "nothing special",
        "not bad",
        "normal",
        "fair",
        "acceptable",
        "ordinary"
    ]

    for phrase in neutral_phrases:
        if phrase in text:
            return "Neutral"

    polarity = TextBlob(text).sentiment.polarity

    if polarity > 0.05:
        return "Positive"

    elif polarity < -0.05:
        return "Negative"

    else:
        return "Neutral"

# ==========================================
# 4. Polarity
# ==========================================

def get_polarity(text):

    return TextBlob(str(text)).sentiment.polarity


# ==========================================
# 5. Subjectivity
# ==========================================

def get_subjectivity(text):

    return TextBlob(str(text)).sentiment.subjectivity


# ==========================================
# 6. NRC-style emotion lexicon
# ==========================================

emotion_lexicon = {

    "joy": {
        "love", "happy", "excellent", "amazing",
        "wonderful", "great", "awesome", "enjoy",
        "enjoyed", "best", "perfect"
    },

    "sadness": {
        "sad", "disappointed", "disappointing",
        "unhappy", "upset", "cry", "bad"
    },

    "anger": {
        "hate", "angry", "terrible", "worst",
        "annoying", "annoyed", "furious",
        "awful"
    },

    "fear": {
        "fear", "afraid", "scared", "danger",
        "dangerous", "worry", "worried"
    },

    "surprise": {
        "surprise", "surprised", "unexpected",
        "shocked", "shock", "wow"
    },

    "trust": {
        "trust", "reliable", "safe", "quality",
        "honest", "recommend"
    },

    "disgust": {
        "disgusting", "disgust", "gross",
        "horrible", "awful"
    },

    "anticipation": {
        "expect", "excited", "waiting",
        "hope", "hopefully", "looking"
    }
}


# ==========================================
# 7. Detect emotion
# ==========================================

def detect_emotion(text):

    text = str(text).lower()

    words = re.findall(r"\b[a-zA-Z]+\b", text)

    emotion_scores = {}

    for emotion, keywords in emotion_lexicon.items():

        score = 0

        for word in words:

            if word in keywords:
                score += 1

        emotion_scores[emotion] = score


    # Find highest scoring emotion

    best_emotion = max(
        emotion_scores,
        key=emotion_scores.get
    )

    highest_score = emotion_scores[best_emotion]


    if highest_score == 0:
        return "Neutral"


    return best_emotion


# ==========================================
# 8. Apply analysis
# ==========================================

df["Sentiment"] = df["Review"].apply(
    classify_sentiment
)

df["Polarity"] = df["Review"].apply(
    get_polarity
)

df["Subjectivity"] = df["Review"].apply(
    get_subjectivity
)

df["Emotion"] = df["Review"].apply(
    detect_emotion
)


# ==========================================
# 9. Display results
# ==========================================

print("\n===================================")
print("SENTIMENT ANALYSIS RESULTS")
print("===================================")

print(
    df[
        [
            "Review",
            "Sentiment",
            "Emotion",
            "Polarity",
            "Subjectivity"
        ]
    ].to_string(index=False)
)


# ==========================================
# 10. Save results
# ==========================================

output_file = "output/analyzed_data.csv"

df.to_csv(
    output_file,
    index=False,
    encoding="utf-8"
)

print("\nAnalysis saved to:")
print(output_file)


# ==========================================
# 11. Sentiment chart
# ==========================================

plt.figure(figsize=(7, 5))

df["Sentiment"].value_counts().plot(
    kind="bar"
)

plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")

plt.tight_layout()

plt.savefig(
    "output/sentiment_chart.png"
)

plt.close()


# ==========================================
# 12. Emotion chart
# ==========================================

plt.figure(figsize=(8, 5))

df["Emotion"].value_counts().plot(
    kind="bar"
)

plt.title("Emotion Distribution")
plt.xlabel("Emotion")
plt.ylabel("Number of Reviews")

plt.tight_layout()

plt.savefig(
    "output/emotion_chart.png"
)

plt.close()


# ==========================================
# 13. Word cloud
# ==========================================

all_reviews = " ".join(
    df["Review"].astype(str)
)

wordcloud = WordCloud(
    width=1000,
    height=500,
    background_color="white"
).generate(all_reviews)


plt.figure(figsize=(12, 6))

plt.imshow(
    wordcloud,
    interpolation="bilinear"
)

plt.axis("off")

plt.tight_layout()

plt.savefig(
    "output/wordcloud.png"
)

plt.close()


# ==========================================
# 14. Final message
# ==========================================

print("\n===================================")
print("ANALYSIS COMPLETED SUCCESSFULLY!")
print("===================================")

print("\nGenerated files:")

print("1. output/analyzed_data.csv")
print("2. output/sentiment_chart.png")
print("3. output/emotion_chart.png")
print("4. output/wordcloud.png")