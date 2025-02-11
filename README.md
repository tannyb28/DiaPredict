# Data Tools Final
*Diabetes risk prediction and health analytics*

## Project Description
This project addresses diabetes risk prediction by using patient data consisting of demographic details, clinical measurements, and lifestyle factors to build a sophisticated deep learning model. Utilizing a TabTransformer architecture, our approach will capture the complex, non-linear interactions among various features, facilitating early detection of diabetes risk. The model is designed to deliver predictions that are not only highly accurate but also interpretable, enabling healthcare providers to understand the key factors driving risk.

## Goals
- **Model Development:** Build and optimize a TabTransformer-based model that surpasses traditional methods in predictive performance, measured by metrics like accuracy, precision, recall, and ROC-AUC.
- **Feature Interaction Analysis:** Implement advanced feature embeddings and attention mechanisms to identify and quantify the influence of individual risk factors.
- **Interpretability:** Ensure the model's decisions are transparent by applying Shapley Addictive Explanations (SHAP), thus providing actionable insights for clinical decision-making.
- **Visualization Tools:** Develop interactive dashboards for visualizing patient risk profiles and model outputs, aiding healthcare providers in understanding and utilizing the predictions effectively.

## Data Collection
- **Source:** We plan to use the [Diabetes Prediction in America Dataset](https://www.kaggle.com/datasets/ashaychoudhary/diabetes-prediction-in-america-dataset) from Kaggle which consists of synthetically generated patient records (HIPAA-compliant) in order to build out an effective model. We may augment the data with other publically available diabetes datasets.

## Data Modeling
We plan to start with Feature Extraction:
- **Data Structuring:** First organize patient records into a structured format suitable for the model, ensuring a consistent set of features for each record.
- **Enhanced Embeddings:** Develop feature embeddings that integrate the various demographic, clinical, and lifestyle data into a unified latent space, enabling the model to learn based on various non-linear features.

Then the specifics of the model:
- **Model Architecture:** Implement a TabTransformer architecture that utilizes self-attention mechanisms to capture relationships within the tabular data.
- **Supervised Learning:** Train the model on the structured patient data to predict diabetes risk as a binary classification task.

## Data Visualization
- **Embedding Space Analysis:** Use techniques like Principal Component Analysis (PCA) to visualize the various embeddings we created, also allowing us to identify the patterns related to various risk profiles.
- **Model Interpretability:** Use tools like SHAP (SHapley Additive exPlanations) to visualize the impact individual features made on the model's predictions, promoting transparency of how the model makes decisions.
- **Interactive Dashboards:** Develop interactive visualizations using platforms like Plotly to allow users to input various patient profiles and see the model outcomes.

## Test Plan
- Use a standard train/validate/test split in order to train the model, fine-tune hyper parameters, and evaluate the model without overfitting
- **Evaluation metrics:**
  - **Predictive Accuracy:** Evaluate how accurate predictions are with accuracy, precision, recall, and F1 score.
  - **Classification Ability:** Calculate ROC-AUC score to determine how well the model is able to differentiate between patients with and without diabetes.
