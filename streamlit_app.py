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

# Sidebar for input mode
st.sidebar.header("Input Method")
input_mode = st.sidebar.radio("Choose input", ["Type Text", "Upload CSV File"])

# --- Helper function to analyze sentences ---
def analyze_sentences(sentences):
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
    return pd.DataFrame(results)

df = None  # Will hold results

# === TYPE TEXT MODE ===
if input_mode == "Type Text":
    user_input = st.text_area("Type your sentences here (separate by period):")

    if user_input:
        sentences = [s.strip() for s in user_input.split('.') if s.strip()]
        if sentences:
            df = analyze_sentences(sentences)

# === UPLOAD CSV MODE ===
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV file (first column = text)", type="csv")
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            # Take first column as text
            sentences = data.iloc[:, 0].dropna().astype(str).tolist()
            if sentences:
                df = analyze_sentences(sentences)
                st.success(f"Successfully loaded {len(sentences)} sentences from CSV!")
            else:
                st.error("No text found in the uploaded CSV.")
        except Exception as e:
            st.error(f"Error reading file: {e}")

# === SHOW RESULTS IF WE HAVE DATA ===
if df is not None and len(df) > 0:
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
        colors=['#99ff99', '#ff9999', '#66b3ff']  # Green=Pos, Red=Neg, Blue=Neu
    )
    ax.axis('equal')
    st.pyplot(fig)
    
    # Summary
    st.subheader("Summary")
    st.write(f"Total sentences: {len(df)}")
    st.write(f"Positive: {sentiment_counts.get('Positive', 0)}")
    st.write(f"Negative: {sentiment_counts.get('Negative', 0)}")
    st.write(f"Neutral: {sentiment_counts.get('Neutral', 0)}")

    # === EXPORT OPTIONS ===
    st.subheader("Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.download_button(
            label="Download as CSV",
            data=df.to_csv(index=False),
            file_name="sentiment_results.csv",
            mime="text/csv"
        )
    
    with col2:
        st.download_button(
            label="Download as JSON",
            data=df.to_json(orient="records"),
            file_name="sentiment_results.json",
            mime="application/json"
        )
    
    with col3:
        if st.button("Generate PDF Report"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "Code Crusaders - Sentiment Analysis Report", ln=1, align="C")
            pdf.ln(10)
            pdf.set_font("Arial", size=12)
            for _, row in df.iterrows():
                pdf.cell(0, 8, f"{row['Sentiment']}: {row['Sentence'][:100]}", ln=1)
            buffer = BytesIO()
            pdf.output(buffer)
            b64 = base64.b64encode(buffer.getvalue()).decode()
            st.markdown(
                f'<a href="data:application/pdf;base64,{b64}" download="CodeCrusaders_Report.pdf">Download PDF Report</a>',
                unsafe_allow_html=True
            )

    st.success("All features complete: Text input • CSV upload • CSV/JSON/PDF export • Visualization")

else:
    st.info("Enter text or upload a CSV file to see results.")
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






