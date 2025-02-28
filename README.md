# Data Tools Final

## Diabetes Risk Prediction and Health Analytics

### Project Description
This project addresses diabetes risk prediction by using patient data consisting of demographic details, clinical measurements, and lifestyle factors to build an effective predictive model. Instead of assuming a complex architecture from the start, we will systematically evaluate different approaches, starting with traditional machine learning models before considering deep learning techniques.  

Given that our data is tabular, we will focus on **Random Forests and Feed-Forward Neural Networks (FNNs)** as our primary modeling approaches, ensuring that we use a method that balances both performance and interpretability. The goal is to develop a solution that is both **accurate** and **transparent**, helping healthcare providers understand the key risk factors driving predictions.  

---

## Goals

- **Baseline Model Development:** Implement and compare traditional models such as **logistic regression, random forests, and feed-forward neural networks (FNNs)** to establish baseline performance.  
- **Exploration of Model Effectiveness:** Determine whether more complex models provide meaningful benefits beyond traditional methods.  
- **Feature Interaction Analysis:** Identify key contributing factors in diabetes risk prediction through **feature importance analysis** (e.g., SHAP, permutation-based explanations).  
- **Interpretability and Transparency:** Ensure all models provide clear insights into their predictions so that medical professionals can make informed decisions.  
- **Visualization Tools:** Develop interactive dashboards to help users explore patient risk profiles and understand model outputs effectively.  

---

## Data Collection

- **Source:** We plan to use the [Diabetes Prediction in America Dataset](https://www.kaggle.com/datasets/ashaychoudhary/diabetes-prediction-in-america-dataset), a synthetically generated patient record dataset (HIPAA-compliant).  
- **Data Augmentation:** We may integrate additional publicly available diabetes datasets to improve model generalization.  

---

## Data Modeling

We will adopt a **progressive modeling strategy** that first establishes baseline performance before determining whether additional complexity is justified.  

### Baseline Models

- **Logistic Regression, Random Forests, and Feed-Forward Neural Networks (FNNs)** to establish initial benchmarks.  
- Evaluate these models on key performance metrics to determine whether deep learning techniques provide a significant advantage.  

### Feature Interaction & Interpretability

- **Feature Importance Analysis:** Use **SHAP values, feature permutation importance, or correlation-based methods** to analyze what influences predictions.  
- **Embedding Analysis:** If using deep learning, explore how the model organizes different patient risk profiles through **Principal Component Analysis (PCA)**.  

---

## Data Visualization

- **Model Interpretability:** Use SHAP values, feature importance scores, and permutation-based methods to provide insights into how the model makes decisions.  
- **Embedding Space Analysis:** If using deep learning, apply PCA or t-SNE to visualize patient feature embeddings and risk patterns.  
- **Interactive Dashboards:** Develop interactive visualizations using **Plotly or Dash**, allowing users to input patient attributes and observe the model’s predictions.  

---

## Test Plan

- Use a **train/validate/test split** to train and fine-tune models while preventing overfitting.  
- **Evaluation metrics:**  
  - **Predictive Accuracy:** Compare performance across models.  
  - **Precision, Recall, F1-Score:** Assess how well the model balances false positives and false negatives.  
  - **ROC-AUC Score:** Evaluate classification ability, especially for probability-based models.  

---
