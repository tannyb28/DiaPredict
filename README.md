# DiaPredict: Predicting Diabetes Risk from Health Indicators

## Project Summary
DiaPredict is a machine learning project developed as part of a Data Science Tools and Applications course. The objective is to predict the likelihood of diabetes in individuals based on a set of health indicators from survey data. The project covers the full data science lifecycle including data collection, preprocessing, visualization, feature engineering, and model training/evaluation.

## Dataset Overview
We initially worked with an older dataset for the midterm phase of the project. For the final report, we switched to a newer and more comprehensive dataset: `diabetes_binary_health_indicators_BRFSS2015.csv`. This dataset was sourced from the Behavioral Risk Factor Surveillance System (BRFSS) and includes numerous binary and categorical health indicators for diabetes prediction.

The change was motivated by the availability of more balanced and richer features in the updated dataset, improving the reliability of our models.

## Project Goals
- Build a predictive model to estimate the risk of diabetes based on health-related survey data.
- Explore which health indicators are most predictive of diabetes.
- Practice end-to-end data science workflows and tools in a reproducible environment.

## Data Collection
The dataset is publicly available and was obtained in CSV format. It includes over 250,000 records and dozens of features related to demographics, physical activity, BMI, smoking, alcohol consumption, and more.

## Data Cleaning and Preprocessing
- Handled missing values and type conversions.
- Applied one-hot encoding to categorical features as needed.
- Investigated multicollinearity using Variance Inflation Factor (VIF).
- Performed feature selection using statistical methods such as chi-squared and ANOVA (f_classif).

## Exploratory Data Analysis (EDA)
We performed a detailed exploratory data analysis to understand distributions, correlations, and feature importance:

- **Pandas Profiling** provided a quick summary of data statistics.
- **Correlation matrix** and pairwise plots highlighted feature relationships.

Key Visualizations:
- ![RF Feature Importances](assets/rf_feature_importances.png)
- ![Learning curve](assets/learning_curve.png)
- ![ROC Curve](assets/roc_curve.png)
- ![PCA Graph](assets/pca_train.png)

## Feature Engineering
We used statistical feature selection techniques including:
- SelectKBest with `chi2` and `f_classif` scores
- PCA for dimensionality reduction and visualization

## Modeling
We trained and evaluated several classification models:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree Classifier
- Random Forest Classifier (with hyperparameter tuning via GridSearchCV)
- XGBoost Classifier

Evaluation metrics:
- Accuracy
- Confusion Matrix
- Classification Report (Precision, Recall, F1-score)
- ROC Curve and AUC

## Results Summary
Random Forest and XGBoost performed best in terms of accuracy and generalization. Feature importance plots indicated that BMI, age, and physical activity were among the top predictive variables.

## How to Reproduce
> Note: A Makefile and GitHub Actions test workflow will be added to automate setup and testing.

## Future Work and Limitations
- Improve hyperparameter tuning and cross-validation
- Implement more interactive visualizations using Plotly or Dash
- Further balance the dataset or experiment with resampling methods like SMOTE

## Video Presentation
*Link to YouTube presentation will be added here once available.*
