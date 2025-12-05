# app.py
"""
Code Crusaders – CAPACITI Week 5 AI in Action
Ready-to-run Streamlit app with:
 - TextBlob and Hugging Face API sentiment engines
 - Multi-class labels: Positive / Neutral / Negative
 - Confidence scores
 - Keyword extraction (TF-IDF + noun-phrases fallback)
 - Batch upload (CSV) and batching for API requests
 - CSV / JSON / PDF export
 - Simple explainability: keywords + per-keyword polarity (TextBlob)
 - Optional manual-label evaluation (confusion matrix + classification report)
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import BytesIO
from fpdf import FPDF
from textblob import TextBlob
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
import base64
import math
import json
import time

st.set_page_config(page_title="Code Crusaders - CAPACITI", layout="wide")
st.title("Code Crusaders – 📊 Sentiment Dashboard")
st.markdown("**Multi-class • Confidence • Keyword Highlighting • Batch Upload • CSV/JSON/PDF Export**")

# ---------- Sidebar: settings ----------
st.sidebar.header("Settings")

engine = st.sidebar.selectbox("Sentiment Engine", ["TextBlob (local)", "Hugging Face API"])
hf_api_key = st.sidebar.text_input("Hugging Face API Key (if using HF)", type="password")
hf_model = st.sidebar.text_input("Hugging Face Model (optional)", value="cardiffnlp/twitter-roberta-base-sentiment")
batch_size = st.sidebar.slider("API batch size (for HF)", 1, 16, 8)
max_chars = st.sidebar.number_input("Max chars to analyze per item", 50, 20000, 2000)
st.sidebar.markdown("---")
st.sidebar.markdown("**Note:** If you don't provide an HF key or choose TextBlob, the app will use TextBlob locally.")

# ---------- Utility functions ----------
LABEL_MAP = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}

def analyze_textblob(text):
    """Return (label, confidence, extra) using TextBlob polarity"""
    blob = TextBlob(text)
    pol = blob.sentiment.polarity  # -1..1
    if pol > 0.1:
        label = "Positive"
    elif pol < -0.1:
        label = "Negative"
    else:
        label = "Neutral"
    confidence = round(min(1.0, abs(pol)), 3)
    return label, confidence, {"polarity": pol}

def call_hf_batch(texts, model, api_key, timeout=20):
    """
    Call HF Inference API with a batch of texts.
    Returns a list of responses (parallel to texts), or raises an exception.
    """
    if not api_key:
        raise ValueError("Hugging Face API key not provided.")
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"inputs": texts}
    # For the HF inference API, sending a list triggers batch inference for supported models.
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"HF API error {r.status_code}: {r.text}")
    return r.json()

def analyze_hf_safe(texts, model, api_key, batch_size=8):
    """
    Accepts list of texts; returns list of tuples (label, confidence, raw)
    Will batch the texts into requests of size batch_size.
    """
    results = []
    n = len(texts)
    for i in range(0, n, batch_size):
        chunk = texts[i:i+batch_size]
        try:
            resp = call_hf_batch(chunk, model, api_key)
            # resp is typically a list of lists: each element is list of dicts with label and score
            # If model returns single dict for single input, handle both cases.
            if isinstance(resp, dict):
                # single response for entire chunk? attempt to normalize
                resp = [resp] * len(chunk)
            # If resp length equals len(chunk) and each item is list -> good
            if len(resp) != len(chunk):
                # fallback: try to interpret as single output per input
                pass
            for item in resp:
                # item expected to be a list of label/score dicts
                if isinstance(item, list):
                    best = max(item, key=lambda x: x.get("score", 0.0))
                    label = LABEL_MAP.get(best.get("label", ""), best.get("label", "Unknown"))
                    conf = round(best.get("score", 0.0), 3)
                    results.append((label, conf, item))
                else:
                    # unexpected shape
                    results.append(("Error", 0.0, item))
        except Exception as e:
            # if HF fails, append error tuples for chunk
            for _ in chunk:
                results.append(("Error", 0.0, str(e)))
            # optionally brief wait to avoid hammering
            time.sleep(1)
    return results

def extract_keywords_tfidf(corpus, top_n=5):
    """Return list-of-lists of top_n keywords per document using TF-IDF"""
    try:
        # use simple token pattern to avoid stop words; let Tfidf do the heavy lifting
        vect = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=1000)
        X = vect.fit_transform(corpus)
        feature_names = np.array(vect.get_feature_names_out())
        keywords_per_doc = []
        for row in X:
            if isinstance(row, np.ndarray):
                data = row
            else:
                data = row.toarray().ravel()
            top_indices = data.argsort()[::-1][:top_n]
            kws = feature_names[top_indices]
            # keep non-empty keywords
            kws = [k for k in kws if len(k.strip())>0]
            keywords_per_doc.append(kws)
        return keywords_per_doc
    except Exception:
        # fallback: noun phrases from TextBlob
        kws = []
        for doc in corpus:
            blob = TextBlob(doc)
            np_list = list(dict.fromkeys([p.lower() for p in blob.noun_phrases]))[:top_n]
            kws.append(np_list)
        return kws

def make_pdf_from_df(df, title="Report"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, title, ln=1, align="C")
    pdf.ln(6)
    pdf.set_font("Arial", size=11)
    
    for idx, row in df.iterrows():
        text = str(row.get("Text", ""))[:800]
        sent = row.get("Sentiment", "")
        conf = row.get("Confidence", "")
        line = f"{idx+1}. [{sent} | {conf}] {text}"
        # FIX: Clean Unicode characters before writing
        line = line.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"').replace("–", "-").replace("—", "-")
        pdf.multi_cell(0, 6, line)
        pdf.ln(1)
    
    # FIX: Use dest='S' to return string, then encode
    pdf_str = pdf.output(dest='S')  # Returns string
    buffer = BytesIO()
    buffer.write(pdf_str.encode('latin-1', 'replace'))
    buffer.seek(0)
    return buffer

def hf_available():
    return engine.startswith("Hugging Face") and hf_api_key.strip() != ""

# ---------- UI: Input mode ----------
mode = st.radio("Input", ["Single Text", "Upload CSV", "Manual Labels (eval)"], index=0, horizontal=True)

results = []  # will be list of dicts

if mode == "Single Text":
    st.subheader("Single Text Analyzer")
    text = st.text_area("Enter text", height=200, value="I love this product. It is fantastic and useful.")
    analyze_button = st.button("Analyze Text")
    if analyze_button:
        txt = text.strip()[:max_chars]
        if not txt:
            st.error("Please enter some text.")
        else:
            if engine == "TextBlob (local)":
                label, conf, extra = analyze_textblob(txt)
            else:
                if not hf_available():
                    st.warning("Hugging Face selected but API key is missing. Falling back to TextBlob.")
                    label, conf, extra = analyze_textblob(txt)
                else:
                    out = analyze_hf_safe([txt], model=hf_model, api_key=hf_api_key, batch_size=batch_size)[0]
                    label, conf, extra = out
            # keywords
            kws = extract_keywords_tfidf([txt], top_n=6)[0]
            # per-keyword polarity via TextBlob (simple explainability)
            kw_explain = []
            for kw in kws:
                pol = TextBlob(kw).sentiment.polarity
                polarity_label = "pos" if pol>0.0 else ("neg" if pol<0.0 else "neu")
                kw_explain.append({"keyword": kw, "polarity": round(pol,3), "label": polarity_label})
            results.append({"Text": txt, "Sentiment": label, "Confidence": conf, "Keywords": ", ".join(kws), "Keyword_Explain": kw_explain})
elif mode == "Upload CSV":
    st.subheader("Batch CSV Upload")
    st.markdown("Upload a CSV where the first column contains text (or a column named `text` / `Text`).")
    uploaded = st.file_uploader("Upload CSV file", type=["csv"])
    run_batch = st.button("Analyze File")
    if uploaded:
        try:
            df_in = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Unable to read uploaded CSV: {e}")
            df_in = None
        if df_in is not None:
            # determine text column
            text_col = None
            candidates = [c for c in df_in.columns if c.lower() in ("text", "content", "message")]
            if candidates:
                text_col = candidates[0]
            else:
                text_col = df_in.columns[0]
            st.info(f"Using column `{text_col}` for text.")
            st.dataframe(df_in.head(10))
            if run_batch:
                texts = df_in[text_col].astype(str).fillna("").tolist()
                # trim and filter empty or short
                texts_proc = [t.strip()[:max_chars] for t in texts]
                # call engine
                if engine == "TextBlob (local)":
                    for t in texts_proc:
                        if len(t) < 1:
                            results.append({"Text": t, "Sentiment": "Empty", "Confidence": 0.0, "Keywords": "", "Keyword_Explain": []})
                            continue
                        lbl, conf, extra = analyze_textblob(t)
                        kws = extract_keywords_tfidf([t], top_n=6)[0]
                        kw_explain = [{"keyword": kw, "polarity": round(TextBlob(kw).sentiment.polarity,3),
                                       "label": ("pos" if TextBlob(kw).sentiment.polarity>0 else ("neg" if TextBlob(kw).sentiment.polarity<0 else "neu"))}
                                      for kw in kws]
                        results.append({"Text": t, "Sentiment": lbl, "Confidence": conf, "Keywords": ", ".join(kws), "Keyword_Explain": kw_explain})
                else:
                    if not hf_available():
                        st.warning("Hugging Face selected but API key is missing. Falling back to TextBlob.")
                        for t in texts_proc:
                            lbl, conf, extra = analyze_textblob(t)
                            kws = extract_keywords_tfidf([t], top_n=6)[0]
                            kw_explain = [{"keyword": kw, "polarity": round(TextBlob(kw).sentiment.polarity,3),
                                           "label": ("pos" if TextBlob(kw).sentiment.polarity>0 else ("neg" if TextBlob(kw).sentiment.polarity<0 else "neu"))}
                                          for kw in kws]
                            results.append({"Text": t, "Sentiment": lbl, "Confidence": conf, "Keywords": ", ".join(kws), "Keyword_Explain": kw_explain})
                    else:
                        # batch call
                        st.info("Calling Hugging Face API — this may take a moment depending on file size.")
                        chunks = [texts_proc[i:i+batch_size] for i in range(0, len(texts_proc), batch_size)]
                        idx = 0
                        for chunk in chunks:
                            with st.spinner(f"Processing batch {idx+1}/{len(chunks)}..."):
                                out = analyze_hf_safe(chunk, model=hf_model, api_key=hf_api_key, batch_size=batch_size)
                                for (lbl, conf, raw), t in zip(out, chunk):
                                    kws = extract_keywords_tfidf([t], top_n=6)[0]
                                    kw_explain = [{"keyword": kw, "polarity": round(TextBlob(kw).sentiment.polarity,3),
                                                   "label": ("pos" if TextBlob(kw).sentiment.polarity>0 else ("neg" if TextBlob(kw).sentiment.polarity<0 else "neu"))}
                                                  for kw in kws]
                                    results.append({"Text": t, "Sentiment": lbl, "Confidence": conf, "Keywords": ", ".join(kws), "Keyword_Explain": kw_explain})
                                idx += 1
                                time.sleep(0.2)
                st.success(f"Processed {len(results)} items.")
elif mode == "Manual Labels (eval)":
    st.subheader("Upload dataset with ground-truth labels for evaluation")
    st.markdown("CSV must have columns: `text` and `label` (labels are: Positive, Neutral, Negative). This will run predictions and show a confusion matrix & classification report.")
    eval_file = st.file_uploader("Upload labeled CSV", type=["csv"])
    run_eval = st.button("Run Evaluation")
    if eval_file:
        try:
            df_eval = pd.read_csv(eval_file)
        except Exception as e:
            st.error(f"Unable to read file: {e}")
            df_eval = None
        if df_eval is not None:
            if 'text' not in [c.lower() for c in df_eval.columns] or 'label' not in [c.lower() for c in df_eval.columns]:
                st.warning("CSV must contain columns named 'text' and 'label' (case-insensitive). Try renaming them.")
            else:
                # map correct columns
                text_col = [c for c in df_eval.columns if c.lower()=='text'][0]
                label_col = [c for c in df_eval.columns if c.lower()=='label'][0]
                texts = df_eval[text_col].astype(str).tolist()
                gold = df_eval[label_col].astype(str).tolist()
                preds = []
                confs = []
                if run_eval:
                    st.info("Running predictions...")
                    if engine == "TextBlob (local)" or not hf_available():
                        for t in texts:
                            lbl, conf, extra = analyze_textblob(t[:max_chars])
                            preds.append(lbl); confs.append(conf)
                    else:
                        out = analyze_hf_safe([t[:max_chars] for t in texts], model=hf_model, api_key=hf_api_key, batch_size=batch_size)
                        for lbl, conf, raw in out:
                            preds.append(lbl); confs.append(conf)
                    # results
                    eval_df = pd.DataFrame({"Text": texts, "Gold": gold, "Pred": preds, "Confidence": confs})
                    st.dataframe(eval_df.head(20))
                    # classification report
                    report = classification_report(gold, preds, labels=["Positive","Neutral","Negative"], output_dict=True, zero_division=0)
                    st.subheader("Classification Report")
                    st.json(report)
                    # confusion matrix
                    cm = confusion_matrix(gold, preds, labels=["Positive","Neutral","Negative"])
                    cm_df = pd.DataFrame(cm, index=["Positive","Neutral","Negative"], columns=["Positive","Neutral","Negative"])
                    st.subheader("Confusion Matrix")
                    st.dataframe(cm_df)

# ---------- Display results if any ----------
if results:
    df = pd.DataFrame(results)
    # ensure columns exist
    if "Keywords" not in df.columns:
        df["Keywords"] = ""
    if "Confidence" not in df.columns:
        df["Confidence"] = 0.0

    st.markdown("### Results")
    st.dataframe(df[["Text","Sentiment","Confidence","Keywords"]].head(200))

    # Visuals
    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(df, names="Sentiment", title="Sentiment Distribution")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = px.box(df, x="Sentiment", y="Confidence", title="Confidence by Sentiment")
        st.plotly_chart(fig2, use_container_width=True)

    # Comparison: show top keywords per sentiment
    st.markdown("### Top Keywords by Sentiment (TF-IDF aggregated)")
    try:
        # group texts by sentiment and compute TF-IDF top keywords
        groups = {}
        for s in df["Sentiment"].unique():
            texts_s = df[df["Sentiment"]==s]["Text"].astype(str).tolist()
            if len(texts_s) < 1:
                continue
            kws = extract_keywords_tfidf(texts_s, top_n=8)
            # flatten and count
            flat = [k for doc in kws for k in doc]
            freq = pd.Series(flat).value_counts().head(12)
            groups[s] = freq
        # display
        for s, freq in groups.items():
            st.write(f"**{s}**")
            st.bar_chart(freq)
    except Exception as e:
        st.warning(f"Keyword aggregation failed: {e}")

    # Explanation: allow user to expand rows and view keyword explainability
    st.markdown("### Explanation (keyword-level)")
    for idx, row in df.head(50).iterrows():
        with st.expander(f"Item {idx+1}: {row['Sentiment']} | {row['Confidence']}"):
            st.write(row["Text"])
            st.write("**Keywords:**", row["Keywords"])
            kwex = row.get("Keyword_Explain", [])
            if isinstance(kwex, (str,)):
                try:
                    kwex = json.loads(kwex)
                except Exception:
                    kwex = []
            if kwex:
                ex_df = pd.DataFrame(kwex)
                st.dataframe(ex_df)
            else:
                st.write("No keyword explanations available.")

    # Export buttons
    st.markdown("### Export Results")
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv_bytes, file_name="results.csv", mime="text/csv")
    st.download_button("Download JSON", df.to_json(orient="records").encode('utf-8'), file_name="results.json", mime="application/json")

    # PDF download — FIXED
    pdf_buf = make_pdf_from_df(df[["Text","Sentiment","Confidence"]], title="Code Crusaders - CAPACITI Report")
    pdf_bytes = pdf_buf.getvalue()
    st.download_button(
         "Download PDF Report",
         data=pdf_bytes,
         file_name="CodeCrusaders_Report.pdf",
         mime="application/pdf"
)

st.markdown("---")
st.markdown("**Notes & limitations**")
st.markdown("""
- TextBlob is a lightweight local analyser: quick but less accurate for many domain-specific texts.
- Hugging Face models often provide much better performance; you must provide an API key and select a model (default used in the app).
- Keyword extraction uses TF-IDF which is simple and explainable but not state-of-the-art (KeyBERT / spaCy / RAKE can be added).
- Explainability here is a simple heuristic: keywords + TextBlob polarity per keyword — adequate for highlighting drivers but not a token-level attribution from the HF model.
- For production: add rate limiting, retries with exponential backoff, caching of API calls, and a robust explainability package (LIME/SHAP) if you need token-level attributions.
""")





