
# Updated Streamlit app with improved UI design

import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from sklearn.metrics import confusion_matrix, classification_report

# Hugging Face API details
API_URL = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment"
headers = {"Authorization": f"Bearer YOUR_HF_TOKEN"}  # Replace with your Hugging Face token

# Function to call Hugging Face API
def query(text):
    response = requests.post(API_URL, headers=headers, json={"inputs": text})
    return response.json()

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
