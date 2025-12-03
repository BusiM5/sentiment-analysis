import streamlit as st
from textblob import TextBlob
import pandas as pd

# Title
st.title("Sentiment Dashboard")
st.write("Enter some text below and see the sentiment analysis!")

# Text input
user_input = st.text_area("Type your text here:")

if user_input:
    # Analyze sentiment
    blob = TextBlob(user_input)
    polarity = blob.sentiment.polarity

    # Determine sentiment
    if polarity > 0:
        sentiment = "Positive"
    elif polarity < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    # Show results
    st.write(f"**Sentiment:** {sentiment}")
    st.write(f"**Polarity score:** {polarity}")

    # Optional: show a simple chart
    df = pd.DataFrame({"Polarity": [polarity]})
    st.bar_chart(df)
