# 📊 Sentiment Analysis NLP

A Python-based Natural Language Processing (NLP) project for analyzing customer reviews and classifying them as **Positive, Negative, or Neutral**.

The project also detects emotions, calculates polarity and subjectivity, generates visualizations, and provides an interactive Streamlit dashboard.

---

## 🚀 Features

- 📝 Review data processing
- 😊 Positive sentiment detection
- 😐 Neutral sentiment detection
- 😞 Negative sentiment detection
- ❤️ Emotion detection using an emotion lexicon
- 📈 Polarity analysis
- 🎯 Subjectivity analysis
- 📊 Sentiment distribution charts
- ❤️ Emotion distribution charts
- ☁️ Word cloud generation
- 🔎 Review search and filtering
- 📥 Download analyzed review data
- 🌐 Interactive Streamlit dashboard
- 📁 CSV input and output

---

## 🧠 NLP Techniques

This project uses several NLP techniques:

### 1. Sentiment Analysis

Reviews are classified into:

- **Positive**
- **Negative**
- **Neutral**

TextBlob is used to calculate the sentiment polarity.

### 2. Emotion Detection

The project uses an emotion lexicon to identify emotions such as:

- Joy
- Sadness
- Anger
- Fear
- Surprise
- Trust
- Disgust
- Anticipation

### 3. Polarity

Polarity represents the emotional direction of the review.

Range:

```text
-1 → Very Negative
 0 → Neutral
+1 → Very Positive