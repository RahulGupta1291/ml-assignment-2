# Assignment 2 — AIMLZG565 — Machine Learning

**WILP Division**
**M. Tech. in Artificial Intelligence & Machine Learning**
**I Semester 2026-27**

**Student Name:** RAHUL GUPTA

**BITS ID:** 2026AJ05022

**Semester:** 1st (2026-27)

---

## Multi-Model Classification and App Deployment

### Problem statement
In this assignment, we have to implement the following:
- Implement multiple classification models
- Build an interactive Streamlit web application to demonstrate your models
- Deploy the app on Streamlit Community Cloud (FREE)
- Using the models, we are trying to predict whether a patient is likely to develop any heart disease based on the data from 14 different features.

---

## Dataset Used
Heart Disease Prediction Dataset 2026
The original dataset contains 100,000 patient records synthesised to reflect real-world cardiovascular health data for the United States adult population (ages 29–80). Every feature distribution, comorbidity prevalence, and clinical measurement has been carefully calibrated against peer-reviewed epidemiological studies, making it suitable for rigorous machine learning research, academic coursework, and clinical risk modelling. For training our models, we used a subset of 2,000 records and 10 unique test sets of 200 records each.

The target variable heart_disease (binary: 0 = No, 1 = Yes) achieves a 46.1% positive rate, providing a near-balanced classification challenge without artificial oversampling.

https://www.kaggle.com/datasets/mohankrishnathalla/heart-disease-prediction-dataset-2026

---

## Github Repository Link

Link: https://github.com/RahulGupta1291/ml-assignment-2

---

## Models Used

**Evaluation Metrics and Results for each model:**

| ML Model Name | Accuracy | AUC Score | Precision | Recall | F1 Score | MCC Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic_Regression | 0.71 | 0.80 | 0.70 | 0.73 | 0.72 | 0.42 |
| Decision_Tree | 0.54 | 0.54 | 0.55 | 0.48 | 0.51 | 0.08 |
| KNN | 0.65 | 0.72 | 0.68 | 0.55 | 0.61 | 0.30 |
| Naive_Bayes_Gaussian | 0.69 | 0.78 | 0.72 | 0.63 | 0.67 | 0.38 |
| Support_Vector_Machine | 0.69 | 0.77 | 0.69 | 0.69 | 0.69 | 0.38 |

**Observations for each model:**

| ML Model Name | Observation about model performance | 
| :--- | :---: |
| Logistic_Regression | Optimises linear log-odds to convert clinical feature weights into well-calibrated probabilities, successfully balancing precision and recall for peak performance | 
| Decision_Tree | Exhibits severe instability and overfitting by building rigid, localised rules that fail to generalise effectively on unseen test data |
| KNN | Suffers from the "curse of dimensionality" as sparse, one-hot encoded variables distort geometric distance calculations and obscure patient similarities |
| Naive_Bayes_Gaussian | Assumes absolute feature independence, ignoring clinical correlations to yield high precision but a high miss rate (low recall) |
| Support_Vector_Machine | Establishes a geometric distance-based hyperplane that scales balanced metrics symmetrically, though it remains less effective than a probabilistic approach |

---
