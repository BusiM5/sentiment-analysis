
mkdir -p docs
nano docs/accuracy_report.md


# Accuracy Report — Sentiment Analysis

## Dataset Overview
- **Total texts:** 110
- **Class distribution:**  
  - Positive: 40  
  - Negative: 40  
  - Neutral: 30  

This dataset includes customer-style reviews and statements, balanced across Positive and Negative classes, with a moderate Neutral portion.



## Metrics (Lexicon Baseline)
- **Accuracy:** 82.7%
- **Macro Precision:** 0.838  
- **Macro Recall:** 0.831  
- **Macro F1-score:** 0.826  
- **Weighted Precision:** 0.855  
- **Weighted Recall:** 0.827  
- **Weighted F1-score:** 0.834  



### Per-Class Performance
| Class     | Precision | Recall | F1  |
|-----------|-----------|--------|-----|
| Positive  | 0.865     | 0.800  | 0.831 |
| Neutral   | 0.650     | 0.867  | 0.743 |
| Negative  | 1.000     | 0.825  | 0.904 |



### Confusion Matrix
|           | Pred_Pos | Pred_Neu | Pred_Neg |
|-----------|----------|----------|----------|
| True_Pos  | 32       | 8        | 0        |
| True_Neu  | 4        | 26       | 0        |
| True_Neg  | 1        | 6        | 33       |



## Discussion of API Limitations

Commercial and open-source sentiment APIs differ in label sets, confidence scoring, and domain coverage—factors that affect accuracy on datasets like customer reviews. **AWS Comprehend** outputs one of four sentiments (*Positive, Negative, Neutral, Mixed*) together with per-class scores, which is convenient for thresholding and batch workflows; however, mapping `Mixed → Neutral` for three-class evaluations can blur genuine mixed polarity and inflate Neutral counts if texts truly contain both praise and criticism.

Open models like **CardiffNLP’s twitter-roberta-base-sentiment** provide robust three-way classification with well-calibrated probabilities, but they are trained predominantly on **tweets**; domain shift from social media to product reviews can introduce misclassifications, especially for longer, formal, or multi-clause inputs.

Across APIs, **sarcasm, irony, and subtle negation** remain hard problems. Positive-looking words used sarcastically (e.g., “Great, it broke on day one”) often fool classifiers, and negation like “not terrible” can flip polarity in ways that require context modeling. Even with modern transformers, variability and uncertainty can arise from decoding settings, prompt phrasing (for LLM-based APIs), and training data biases; recent analyses highlight inconsistent outputs on the same input when conditions change, underscoring the need for transparent confidence thresholds and deterministic evaluation pipelines.

Comparative research on **cloud NLP services** finds performance differences by task and dataset. For example, studies comparing AWS Comprehend to Azure and other vendors report varying strengths—some services excel at detecting negatives, others show stronger macro-F1—emphasizing that **no single API dominates across domains** and that longitudinal changes in vendor models can shift outcomes over time. This motivates maintaining a **repeatable evaluation harness** (confusion matrix + macro/weighted metrics) and documenting thresholds, mappings, and preprocessing for fair comparisons.

Add accuracy report with metrics and discussion
