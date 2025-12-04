import streamlit as st
from textblob import TextBlob
import pandas as pd
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Sentiment Dashboard",
    page_icon="😊",
    layout="wide"
)

# Title and description
st.title("📊 Sentiment Dashboard")
st.write("""
Analyze the sentiment of your text!  
Type or paste multiple sentences, and see the overall sentiment distribution.
""")

# Text input
user_input = st.text_area("Type your sentences here (separate by period):")

if user_input:
    # Split input into sentences
    sentences = [s.strip() for s in user_input.split('.') if s.strip()]
    
    results = []
    for sentence in sentences:
        blob = TextBlob(sentence)
        polarity = blob.sentiment.polarity
        if polarity > 0:
            sentiment = "Positive"
        elif polarity < 0:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        results.append({"Sentence": sentence, "Sentiment": sentiment, "Polarity": polarity})
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Display table
    st.subheader("Sentences and Sentiment")
    st.dataframe(df, use_container_width=True)
    
    # Calculate sentiment counts
    sentiment_counts = df['Sentiment'].value_counts()
    
    # Plot pie chart
    st.subheader("Sentiment Distribution")
    fig, ax = plt.subplots()
    ax.pie(
        sentiment_counts, 
        labels=sentiment_counts.index, 
        autopct='%1.1f%%', 
        startangle=90, 
        colors=['#66b3ff','#ff9999','#99ff99']
    )
    ax.axis('equal')  # Equal aspect ratio ensures pie chart is circular
    st.pyplot(fig)
    
    # Summary
    st.subheader("Summary")
    st.write(f"Total sentences: {len(sentences)}")
    st.write(f"Positive: {sentiment_counts.get('Positive', 0)}")
    st.write(f"Negative: {sentiment_counts.get('Negative', 0)}")
    st.write(f"Neutral: {sentiment_counts.get('Neutral', 0)}")

import pandas as pd
import streamlit as st

st.title("Sentiment Analysis Dashboard")

# Load predictions CSV
df = pd.read_csv("data/sentiment_predictions_with_confidence.csv")

# Show predictions table
st.subheader("Predictions Table")
st.dataframe(df[['Text', 'Expected Sentiment', 'Predicted Sentiment', 'Confidence']])

# Sentiment distribution chart
st.subheader("Sentiment Distribution")
st.bar_chart(df['Predicted Sentiment'].value_counts())



# File upload
uploaded_file = st.file_uploader("Or upload a text file (.txt or .csv)", type=["txt", "csv"])

if uploaded_file is not None:
    # If CSV, read with pandas
    if uploaded_file.name.endswith(".csv"):
        df_file = pd.read_csv(uploaded_file)
        # Assume the first column has text
        text_data = df_file.iloc[:, 0].tolist()
        user_input = ". ".join(text_data)
    else:  # txt file
        text_data = uploaded_file.read().decode("utf-8")
        user_input = text_data

