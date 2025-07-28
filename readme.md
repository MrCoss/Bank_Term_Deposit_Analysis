# Bank Term Deposit Subscription Prediction
This project presents an end-to-end machine learning pipeline to predict whether a bank client will subscribe to a term deposit, based on historical marketing data. The solution includes comprehensive data cleaning, class imbalance handling with SMOTE, model training (Random Forest, XGBoost, Logistic Regression, Gradient Boosting), hyperparameter tuning, and export of a deployment-ready model with a detailed PDF report.

---

## Project Structure

```text
Bank_Term_Deposit_Analysis/
│
├── datasets/
│   └── bankmarketing.csv                  # Raw dataset
│
├── output/
│   ├── cleaned_dataset.csv                # Preprocessed dataset
│   ├── scaled_encoded_dataset.csv         # After encoding and scaling
│   ├── classification_report.txt          # Evaluation metrics
│   ├── models/
│   │   └── best_model_xgboost.pkl         # Final exported model
│   ├── plots/
│   │   ├── target_distribution.png
│   │   ├── missing_values.png
│   │   ├── scaled_numeric_distributions.png
│   │   ├── feature_importance.png
│   │   ├── confusion_matrix.png
│   │   ├── optimized_rf_confusion_matrix.png
│   │   ├── logisticregression_confusion_matrix.png
│   │   ├── gradientboosting_confusion_matrix.png
│   │   └── xgboost_confusion_matrix.png
│   └── Term_Deposit_Analysis_Report.pdf   # Final PDF report
│
├── Bank_Term_Deposit_Analysis.ipynb       # Full pipeline notebook
├── requirements.txt                       # Python dependencies
├── .gitignore
└── README.md                              # This file
````

---

## Key Highlights

* Cleaned & deduplicated dataset with robust preprocessing
* Auto-detected CSV delimiters and string normalization
* Label encoding and feature scaling
* Class imbalance resolved using **SMOTE**
* Hyperparameter tuning with `GridSearchCV`
* Visualizations with **Rose Pine** custom theme
* Comparison of models: Random Forest, Logistic Regression, Gradient Boosting, XGBoost
* Final model exported and ready for deployment
* PDF report with all metrics, visuals, and summary

---

## Sample Visualizations

### Target Distribution

![Target Distribution](output/plots/target_distribution.png)

### Feature Importance

![Feature Importance](output/plots/feature_importance.png)

### Optimized Random Forest Confusion Matrix

![Optimized RF](output/plots/optimized_rf_confusion_matrix.png)

---

## Model Evaluation Summary

| Model                              | Accuracy | Weighted F1 | Notes                           |
| ---------------------------------- | -------- | ----------- | ------------------------------- |
| Random Forest (Raw)                | \~92%    | \~0.89      | High precision, poor recall     |
| Random Forest (SMOTE + GridSearch) | \~95%    | \~0.94      | Balanced & optimized            |
| Gradient Boosting                  | \~93%    | \~0.93      | Strong generalization           |
| **XGBoost**                        | \~94.7%  | **0.9505**  | Best performer overall        |
| Logistic Regression                | \~85%    | \~0.87      | Interpretable, but lower recall |

---

## Final Deployed Model

```bash
output/models/best_model_xgboost.pkl
```

The best performing model (**XGBoost**) is exported and ready for real-world deployment in marketing automation systems, CRM tools, or customer segmentation pipelines.

---

## Tech Stack

* **Language:** Python 3.10+
* **Libraries:** pandas, numpy, seaborn, matplotlib
* **ML Tools:** scikit-learn, XGBoost, imbalanced-learn
* **Report:** PDF generation via ReportLab

---

## Setup Instructions

```bash
# Clone the repository
git clone https://github.com/yourusername/Bank_Term_Deposit_Analysis.git
cd Bank_Term_Deposit_Analysis

# Install dependencies
pip install -r requirements.txt

# Launch the Jupyter notebook
jupyter notebook Bank_Term_Deposit_Analysis.ipynb
```

---

## Future Improvements

* Cross-validation strategies (StratifiedKFold, RepeatedKFold)
* Experiment tracking with MLflow or Weights & Biases
* Model interpretability with SHAP
* Deployment via Flask/FastAPI or Streamlit
* CI/CD and Docker containerization

---

## Author

**Costas Pinto**
MCA | AI-ML Enthusiast | Data Analytics Intern  
[Skilled Mentor Internship | July–August 2025]

---

## Report Output

Full PDF report with all plots and summaries:
[`output/Term_Deposit_Analysis_Report.pdf`](output/Term_Deposit_Analysis_Report.pdf)

---
