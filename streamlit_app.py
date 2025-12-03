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

# Streamlit UI Enhancements
st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")

# Sidebar Navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Upload & Analyze", "Visualizations", "Metrics"])

# Header Styling
st.markdown("""
    <style>
        .main-title {font-size:40px; color:#4CAF50; font-weight:bold; text-align:center;}
        .sub-title {font-size:20px; color:#555; text-align:center;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">Sentiment Analysis Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Analyze text sentiment, visualize results, and evaluate accuracy</p>', unsafe_allow_html=True)


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
# Global DataFrame variable
df = None

if section == "Upload & Analyze":
    st.header("Step 1: Upload Your Dataset")
    uploaded_file = st.file_uploader("Upload a CSV file with 'Text' and 'Expected Sentiment' columns", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
        st.write("Preview of your data:")
        st.dataframe(df.head())

        if st.button("Run Sentiment Analysis", help="Click to analyze sentiments using Hugging Face API"):
            predictions = []
            scores = []

            with st.spinner("Analyzing sentiments... This may take a few seconds"):
                for text in df["Text"]:
                    result = query(text)
                    label = result[0][0]["label"]
                    score = result[0][0]["score"]
                    predictions.append(label)
                    scores.append(score)

            df["Predicted Sentiment"] = predictions
            df["Confidence Score"] = scores

            st.success("Analysis complete!")
            st.subheader("Results Preview")
            st.dataframe(df.head())

            st.download_button("Download Results as CSV", df.to_csv(index=False), "sentiment_results.csv", "text/csv")

elif section == "Visualizations":
    st.header("Step 2: Visualize Sentiment Data")
    if df is None:
        st.warning("Please upload and analyze data first in the 'Upload & Analyze' section.")
    else:
        st.subheader("Sentiment Distribution")
        fig = px.bar(df["Predicted Sentiment"].value_counts().reset_index(),
                     x="index", y="Predicted Sentiment", color="index",
                     labels={"index": "Sentiment", "Predicted Sentiment": "Count"},
                     title="Distribution of Predicted Sentiments")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Confidence Score Histogram")
        fig2 = px.histogram(df, x="Confidence Score", nbins=20,
                            title="Confidence Score Distribution")
        st.plotly_chart(fig2, use_container_width=True)

elif section == "Metrics":
    st.header("Step 3: Accuracy Metrics & Confusion Matrix")
    if df is None:
        st.warning("Please upload and analyze data first in the 'Upload & Analyze' section.")
    else:
        y_true = df["Expected Sentiment"]
        y_pred = df["Predicted Sentiment"]

        st.subheader("Classification Report")
        report = classification_report(y_true, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred, labels=["Positive", "Negative", "Neutral"])
        cm_df = pd.DataFrame(cm, index=["Actual Positive", "Actual Negative", "Actual Neutral"],
                             columns=["Pred Positive", "Pred Negative", "Pred Neutral"])
        st.dataframe(cm_df)

        fig3 = px.imshow(cm, text_auto=True,
                         labels=dict(x="Predicted", y="Actual", color="Count"),
                         x=["Positive", "Negative", "Neutral"],
                         y=["Positive", "Negative", "Neutral"],
                         title="Confusion Matrix Heatmap")
        st.plotly_chart(fig3, use_container_width=True)

st.sidebar.info("Replace 'YOUR_HF_TOKEN' with your Hugging Face API token before running.")

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
