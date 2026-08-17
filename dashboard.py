import streamlit as st
import pandas as pd
from textblob import TextBlob
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Sentiment Analysis NLP",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("📊 Sentiment Analysis NLP Dashboard")

st.write(
    "Analyze customer reviews using Natural Language Processing "
    "(NLP), sentiment classification, and emotion detection."
)

st.divider()


# =========================================================
# EMOTION LEXICON
# =========================================================

emotion_lexicon = {

    "Joy": {
        "love", "happy", "excellent", "amazing",
        "wonderful", "great", "awesome", "enjoy",
        "enjoyed", "best", "perfect"
    },

    "Sadness": {
        "sad", "disappointed", "disappointing",
        "unhappy", "upset", "cry", "bad"
    },

    "Anger": {
        "hate", "angry", "terrible", "worst",
        "annoying", "annoyed", "furious", "awful"
    },

    "Fear": {
        "fear", "afraid", "scared", "danger",
        "dangerous", "worry", "worried"
    },

    "Surprise": {
        "surprise", "surprised", "unexpected",
        "shocked", "shock", "wow"
    },

    "Trust": {
        "trust", "reliable", "safe", "quality",
        "honest", "recommend"
    },

    "Disgust": {
        "disgusting", "disgust", "gross",
        "horrible", "awful"
    },

    "Anticipation": {
        "expect", "excited", "waiting",
        "hope", "hopefully", "looking"
    }
}


# =========================================================
# SENTIMENT
# =========================================================

def analyze_sentiment(text):

    text = str(text).lower().strip()

    # Common neutral expressions
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
            polarity = TextBlob(text).sentiment.polarity
            return "Neutral", polarity

    polarity = TextBlob(text).sentiment.polarity

    if polarity > 0.05:
        sentiment = "Positive"

    elif polarity < -0.05:
        sentiment = "Negative"

    else:
        sentiment = "Neutral"

    return sentiment, polarity


# =========================================================
# EMOTION
# =========================================================

def detect_emotion(text):

    text = str(text).lower()

    words = re.findall(
        r"\b[a-zA-Z]+\b",
        text
    )

    scores = {}

    for emotion, keywords in emotion_lexicon.items():

        score = sum(
            1 for word in words
            if word in keywords
        )

        scores[emotion] = score

    best_emotion = max(
        scores,
        key=scores.get
    )

    if scores[best_emotion] == 0:
        return "Neutral"

    return best_emotion


# =========================================================
# ANALYZE DATA
# =========================================================

def analyze_dataframe(df):

    if "Review" not in df.columns:

        st.error(
            "CSV must contain a column named 'Review'."
        )

        return None

    df = df.copy()

    df["Sentiment"] = df["Review"].apply(
        lambda x: analyze_sentiment(x)[0]
    )

    df["Polarity"] = df["Review"].apply(
        lambda x: analyze_sentiment(x)[1]
    )

    df["Subjectivity"] = df["Review"].apply(
        lambda x: TextBlob(
            str(x)
        ).sentiment.subjectivity
    )

    df["Emotion"] = df["Review"].apply(
        detect_emotion
    )

    return df


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("⚙️ Data Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload Review CSV",
    type=["csv"]
)


# =========================================================
# MAIN APP
# =========================================================

if uploaded_file is None:

    st.info(
        "👈 Upload a CSV file from the sidebar to begin."
    )

    st.markdown(
        """
        ### CSV Format

        Your CSV must contain a column called:

        **Review**

        Example:

        ```text
        Review
        I love this product!
        The product is terrible.
        The product is okay.
        ```
        """
    )

else:

    # -----------------------------------------------------
    # LOAD DATA
    # -----------------------------------------------------

    df = pd.read_csv(uploaded_file)

    st.success(
        f"Successfully loaded {len(df)} reviews."
    )

    analyzed_df = analyze_dataframe(df)


    if analyzed_df is not None:

        # =================================================
        # METRICS
        # =================================================

        total = len(analyzed_df)

        positive = (
            analyzed_df["Sentiment"] == "Positive"
        ).sum()

        negative = (
            analyzed_df["Sentiment"] == "Negative"
        ).sum()

        neutral = (
            analyzed_df["Sentiment"] == "Neutral"
        ).sum()


        positive_pct = (
            positive / total * 100
            if total > 0 else 0
        )

        negative_pct = (
            negative / total * 100
            if total > 0 else 0
        )

        neutral_pct = (
            neutral / total * 100
            if total > 0 else 0
        )


        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "📋 Total Reviews",
            total
        )

        col2.metric(
            "😊 Positive",
            f"{positive} ({positive_pct:.1f}%)"
        )

        col3.metric(
            "😞 Negative",
            f"{negative} ({negative_pct:.1f}%)"
        )

        col4.metric(
            "😐 Neutral",
            f"{neutral} ({neutral_pct:.1f}%)"
        )


        st.divider()


        # =================================================
        # SENTIMENT ANALYSIS
        # =================================================

        st.header("📊 Sentiment Analysis")


        col1, col2 = st.columns(2)


        with col1:

            sentiment_counts = (
                analyzed_df["Sentiment"]
                .value_counts()
            )

            st.subheader(
                "Sentiment Distribution"
            )

            st.bar_chart(
                sentiment_counts
            )


        with col2:

            st.subheader(
                "Sentiment Percentage"
            )

            sentiment_percentage = (
                analyzed_df["Sentiment"]
                .value_counts(normalize=True)
                * 100
            )

            st.bar_chart(
                sentiment_percentage
            )


        # =================================================
        # EMOTION ANALYSIS
        # =================================================

        st.header("❤️ Emotion Analysis")

        emotion_counts = (
            analyzed_df["Emotion"]
            .value_counts()
        )

        st.bar_chart(
            emotion_counts
        )


        # =================================================
        # WORD CLOUD
        # =================================================

        st.header("☁️ Review Word Cloud")

        all_reviews = " ".join(
            analyzed_df["Review"].astype(str)
        )

        if all_reviews.strip():

            wordcloud = WordCloud(
                width=1000,
                height=500,
                background_color="white"
            ).generate(
                all_reviews
            )

            fig, ax = plt.subplots(
                figsize=(12, 5)
            )

            ax.imshow(
                wordcloud,
                interpolation="bilinear"
            )

            ax.axis("off")

            st.pyplot(fig)


        # =================================================
        # FILTERS
        # =================================================

        st.header("🔎 Review Explorer")

        col1, col2 = st.columns(2)


        with col1:

            sentiment_filter = st.selectbox(
                "Filter by Sentiment",
                [
                    "All",
                    "Positive",
                    "Negative",
                    "Neutral"
                ]
            )


        with col2:

            search_text = st.text_input(
                "Search Reviews"
            )


        filtered_df = analyzed_df.copy()


        if sentiment_filter != "All":

            filtered_df = filtered_df[
                filtered_df["Sentiment"]
                == sentiment_filter
            ]


        if search_text:

            filtered_df = filtered_df[
                filtered_df["Review"]
                .astype(str)
                .str.contains(
                    search_text,
                    case=False,
                    na=False
                )
            ]


        st.write(
            f"Showing {len(filtered_df)} reviews"
        )


        st.dataframe(
            filtered_df,
            use_container_width=True
        )


        # =================================================
        # POSITIVE REVIEWS
        # =================================================

        st.header("😊 Positive Reviews")

        positive_reviews = analyzed_df[
            analyzed_df["Sentiment"] == "Positive"
        ]

        if len(positive_reviews) > 0:

            st.dataframe(
                positive_reviews[
                    [
                        "Review",
                        "Emotion",
                        "Polarity"
                    ]
                ],
                use_container_width=True
            )

        else:

            st.info(
                "No positive reviews found."
            )


        # =================================================
        # NEGATIVE REVIEWS
        # =================================================

        st.header("😞 Negative Reviews")

        negative_reviews = analyzed_df[
            analyzed_df["Sentiment"] == "Negative"
        ]

        if len(negative_reviews) > 0:

            st.dataframe(
                negative_reviews[
                    [
                        "Review",
                        "Emotion",
                        "Polarity"
                    ]
                ],
                use_container_width=True
            )

        else:

            st.info(
                "No negative reviews found."
            )


        # =================================================
        # DOWNLOAD
        # =================================================

        st.header("📥 Export Results")

        csv_data = analyzed_df.to_csv(
            index=False
        ).encode("utf-8")


        st.download_button(
            label="⬇️ Download Analyzed Data",
            data=csv_data,
            file_name="analyzed_reviews.csv",
            mime="text/csv"
        )


        # =================================================
        # FOOTER
        # =================================================

        st.divider()

        st.caption(
            "Sentiment Analysis NLP | "
            "Built with Python, Pandas, TextBlob, "
            "Streamlit and NLP emotion lexicons."
        )