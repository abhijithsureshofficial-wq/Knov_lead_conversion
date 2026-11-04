# 🎓 Lead Conversion Prediction & Sales Pipeline Analytics
**An End-to-End Machine Learning & Business Intelligence Project for EdTech CRM Optimization**

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Wrangling-red?logo=pandas)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-green?logo=xgboost)

## 📖 Executive Summary
In high-volume EdTech sales environments, sales teams often suffer from "lead fatigue" and inefficient follow-up routing. This project analyzes **27,375+ real-world CRM leads** to identify the drivers of conversion and build a machine learning classification system capable of predicting lead success. 

By addressing a severe **31.4:1 class imbalance** (only 3.09% baseline conversion rate), this project demonstrates advanced data wrangling, cost-sensitive learning, target leakage detection, and actionable Business Intelligence generation.

## 📂 Dataset Overview
- **Source:** Real-world EdTech CRM (PII masked for privacy compliance).
- **Volume:** 27,375 records spanning multiple marketing campaigns.
- **Features:** 15 attributes including acquisition source, campaign details, executive assignment, and interaction timestamps.
- **Target Variable:** `lead_stage` (Binarized to `is_converted`: 1 = CONVERTED, 0 = LOST/Prospect/Follow Up).

## 🔒 Data Privacy & Ethics
- The source dataset contains contact fields (name, phone, email). Although these were synthetically altered in the original brief, **the raw file is intentionally excluded** from this repository as a privacy best-practice.
- All PII columns are **dropped during Phase 3** before any analysis or modeling.
- The published `cleaned_leads.csv` contains **no personally identifiable information**.
- Notebook previews redact PII fields to keep outputs safe for public viewing.

## 🏗️ Project Architecture
The project is structured in a professional, phase-gated Jupyter Notebook:
1. **Phase 1-2:** Business Framing, Data Ingestion & Profiling.
2. **Phase 3:** PII Handling, Dead-Column Dropping & Missing Data Strategy.
3. **Phase 4:** Feature Engineering (Speed-to-Lead metrics, temporal extraction).
4. **Phase 5:** Exploratory Data Analysis (BI Dashboards & Statistical Filtering).
5. **Phase 6-7:** Predictive Modeling with Stratified Splitting & Cost-Sensitive Learning.
6. **Phase 8:** Model Interpretability & Deployment Simulation.

## 🧠 Machine Learning Approach
Due to the extreme class imbalance (97% negative class), standard Accuracy is a misleading metric. A naive model guessing "Not Converted" would achieve 97% accuracy but provide zero business value.
- **Metrics Used:** PR-AUC (Precision-Recall Area Under Curve) and F1-Score.
- **Algorithms Tested:** Logistic Regression, Random Forest, and XGBoost.
- **Imbalance Handling:** Applied `scale_pos_weight` in XGBoost and `class_weight='balanced'` in tree-based models to mathematically penalize False Negatives (missing a hot lead).

## ⚠️ Critical Business Insight: Target Leakage Detection
During model evaluation, XGBoost achieved a near-perfect ROC-AUC (0.999). Feature importance analysis revealed that post-interaction variables (e.g., `lead_tag = 'not-interested'`) were driving predictions. 

**Professional Takeaway:** These tags are applied *after* the first sales call. Therefore, this model serves as a **Post-Touch Lead Scoring System** (prioritizing follow-ups) rather than a Pre-Touch Routing System. Documenting this distinction is critical for real-world MLOps deployment.

## 📊 Key Business Recommendations
1. **Eradicate Ghost Leads:** 24.5% of generated leads were never contacted. Automated SMS triggers are recommended for leads untouched past 2 hours.
2. **Speed-to-Lead Optimization:** EDA confirms a correlation between rapid first-response times and conversion probability.
3. **Channel ROI Reallocation:** Shift ad spend toward high-intent `source_types` identified in the BI analysis, pausing low-yield campaigns.

## 🚀 How to Use the Prediction System
The notebook includes an inference function that dynamically handles one-hot encoding alignment for new, unseen leads:

```python
from notebook import predict_lead_conversion

new_lead = {
    'lead_status': 'PENDING WITH EXECUTIVE',
    'campaign_name': 'Data Analytics Knovista',
    'source_type': 'INSTAGRAM',
    'ever_called': 0,
    'response_time_hours': -1
    # ... other features
}
```
predict_lead_conversion(new_lead)
### Output: 🎯 Conversion Probability: 68.4% | 📊 Recommended Action Tier: 🟡 WARM LEAD

## 📁 Repository Structure

```text
Knov_Lead_conversion/
├── Knovista_Lead_Conversion_Prediction.ipynb   # End-to-end notebook with rendered outputs
├── README.md                                   # Project documentation
├── requirements.txt                            # Python dependencies
├── .gitignore                                  # Excludes raw PII dataset and temp files
└── outputs/
    ├── data/
    │   └── cleaned_leads.csv                   # PII-free, analysis-ready dataset
    ├── models/
    │   ├── xgboost_lead_model.joblib           # Champion trained XGBoost model
    │   └── feature_columns.joblib              # Feature column order for inference
    ├── metrics/
    │   └── model_comparison.csv                # Evaluation metrics for all models
    └── figures/
        └── 01-07_*.png                         # High-res exported business charts
```

## 🔄 How to Reproduce This Project

### Prerequisites
Python 3.9+
Google Colab (recommended) or Jupyter Notebook
Dependencies: pip install -r requirements.txt

### Data Availability & Privacy
The raw dataset (Unified_Leads.csv) contains contact fields and is intentionally excluded from this repository (see Data Privacy & Ethics). Instead, the cleaned, PII-free dataset (outputs/data/cleaned_leads.csv) is published for transparency.
Because of this, there are two ways to engage with the project:

### Path 1 — Full End-to-End Run (requires raw data access)
For the project owner or anyone granted access to the original CRM export:
1. Place Unified_Leads.csv inside the Lead_Conversion/ folder (or update file_path in Phase 2).
2. Open Knovista_Lead_Conversion_Prediction.ipynb in Colab or Jupyter.
3. If using Colab, mount your Drive when prompted.
4. Run all cells top-to-bottom. Every artifact (model, metrics, figures) regenerates into outputs/.

### Path 2 — Verify & Re-run Analysis (published cleaned data only)
For reviewers using only this public repository, Phases 2–4 depend on the private raw file. However, you can load the published cleaned dataset and reproduce everything from Phase 5 (EDA) onward:

```python
import pandas as pd

# Load the published PII-free cleaned dataset
df_clean = pd.read_csv('outputs/data/cleaned_leads.csv')

# Sanity check
print(df_clean.shape)
print(df_clean['is_converted'].value_counts(normalize=True))
```
From here, you can re-run the EDA, model training (Phases 6–7), and the prediction system (Phase 8).

#### 📖 Note: The notebook ships with all outputs rendered, so you can also read the complete analysis end-to-end on GitHub without running a single cell.

## 🛠️ Tech Stack
Languages: Python 3
Data Manipulation: Pandas, NumPy
Visualization: Matplotlib, Seaborn
Machine Learning: Scikit-Learn, XGBoost
Environment: Google Colab / Jupyter Notebook


#### *Built as a professional portfolio project demonstrating end-to-end Data Science capabilities, from raw CRM data extraction to business strategy formulation.*
