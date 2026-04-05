# 🚀 Influencer Integrity Detector

## 🧠 Problem Statement
Brands lose millions due to influencer inflation where fake engagement (bots) inflates metrics.
Our goal is to detect bots, compute real engagement, and provide a reliable influencer trust score.

---

## 🔥 Key Features

### 👤 User-Level Detection
- Bot probability prediction
- Behavioral + linguistic + stylometry features
- SHAP explainability

### ⭐ Influencer-Level Metrics
- Authentic Engagement Rate (AER)
- Fake Engagement %
- True Reach
- Integrity Score
- Adjusted Cost Per Engagement (CPE)

### 🕸️ Graph-Based Detection
- Bot community detection using Louvain clustering
- Co-comment + timing + similarity edges

### 🧠 Advanced ML
- XGBoost + Probability Calibration
- SBERT semantic similarity
- Stylometry analysis
- Cross-post behavior modeling

---

## 🏗️ Folder Structure

```
project/
│
├── api/
│   ├── app.py
│   ├── pipeline.py
│   ├── model_loader.py
│   ├── scoring.py
│
├── feature_extraction/
│   ├── linguistic.py
│   ├── behavioural.py
│   ├── user.py
│   ├── merge.py
│   ├── build_features.py
│
│
├── evaluation_metrics/
│   ├── scoring.py
│   ├── influencer_scoring.py
│   ├── graph_detection.py
│   ├── explain.py
│
├── model_training/
│   ├── xg_boost.py
│
├── model/
│   ├── shap_explainer.pkl
│   ├── xgb_model.pkl
│
├── config/
│   ├── features.py
│
├── data/
│   ├── generated/
│   ├── input/
│
├── pipeline_runner.py
│
├── test/
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_pipeline.py
│   ├── test_model.py
│   ├── test_graph.py
│   ├── test_robustness.py
│   ├── test_stress.py
|   ├── pytest.ini
│
├── streamlit_app.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

Additional:
```bash
pip install sentence-transformers python-louvain shap
```

---

## ▶️ How to Run

### 1. Train Model
```bash
python models/xg_boost.py
```

### 2. Start API
```bash
uvicorn api.app:app --reload
```

### 3. Run Streamlit Dashboard
```bash
streamlit run streamlit_app.py
```

---

## 📊 API Usage

### Endpoint:
```
POST /predict
```

### Input:
- users
- posts
- comments

### Output:
- user-level predictions
- influencer-level metrics
- detected bot communities

---

## 📈 Model Evaluation

We evaluate using:
- AUC-ROC
- Precision / Recall / F1
- Precision@K
- Fake Engagement Detection Rate

---

## 🧠 Explainability

- SHAP global feature importance
- Per-user explanation
- Feature contribution insights

---

## 🚀 Innovation Highlights

- Semantic detection using SBERT
- Cross-post behavior tracking
- Stylometry-based bot detection
- Graph-based community detection
- Calibrated probabilities for real-world reliability

---

## 🏆 Business Impact

- Detect fake engagement accurately
- Estimate true influencer reach
- Optimize marketing spend
- Reduce fraud risk

---

## 🔮 Future Improvements

- Real-time streaming detection
- Larger-scale graph processing (Spark)
- Advanced deep learning models

---

Built as a high-performance AI system for hackathon evaluation.
