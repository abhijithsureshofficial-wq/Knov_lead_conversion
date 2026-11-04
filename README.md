# 🎓 Lead Conversion Prediction & Sales Pipeline Analytics
**An End-to-End Machine Learning & Business Intelligence Project for EdTech CRM Optimization**

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Wrangling-red?logo=pandas)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-green?logo=xgboost)

## 📖 Executive Summary
In high-volume EdTech sales environments, sales teams often suffer from "lead fatigue" and inefficient follow-up routing. This project analyzes **27,375+ real-world CRM leads** to identify the drivers of conversion and build a machine learning classification system capable of predicting lead success. 

By addressing a severe **31.4:1 class imbalance** (only 3.09% baseline conversion rate), this project demonstrates advanced data wrangling, cost-sensitive learning, and actionable Business Intelligence generation.

## 📂 Dataset Overview
- **Source:** Real-world EdTech CRM (PII masked for privacy compliance).
- **Volume:** 27,375 records spanning multiple marketing campaigns.
- **Features:** 15 attributes including acquisition source, campaign details, executive assignment, and interaction timestamps.
- **Target Variable:** `lead_stage` (Binarized to `is_converted`: 1 = CONVERTED, 0 = LOST/Prospect).

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
