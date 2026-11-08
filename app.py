import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ---- Page Configuration ----
st.set_page_config(
    page_title="Heart Disease Classification Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Dictionary Mapping Display Names to Exact .joblib Filenames ----
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors (KNN)": "knn.joblib",
    "Naive Bayes (Gaussian)": "naive_bayes_gaussian.joblib",
    "Support Vector Machine (SVM)": "support_vector_machine.joblib"
}

# ---- Cached Data & Model Loaders ----
@st.cache_resource
def load_model(model_filename):
    model_path = os.path.join("model", model_filename)
    return joblib.load(model_path)

@st.cache_data
def load_test_data(set_number):
    csv_path = os.path.join("test_sets", f"heart_disease_test_set_{set_number}.csv")
    return pd.read_csv(csv_path)

# ---- Sidebar Configuration ----
st.sidebar.header("Configuration")

selected_model_name = st.sidebar.selectbox(
    "Select Classification Model",
    list(MODEL_FILES.keys())
)

selected_dataset_num = st.sidebar.selectbox(
    "Select Test Dataset",
    options=list(range(1, 11)),
    format_func=lambda x: f"Test Dataset {x}"
)

# ---- Main Dashboard Header & Dataset Description ----
st.title("Heart Disease Classification Dashboard")

st.info(
    "**Dataset Overview:** This dashboard evaluates the Classification models trained on the "
    "*Heart Disease Prediction Dataset 2026*. Each unique test dataset contains 200 balanced, unseen patient records "
    "(100 healthy, 100 diagnosed with heart disease) across clinical and lifestyle indicators."
)


# ---- Define updated list of categorical variables  ----
continuous_vars = [
    "age", "bmi", "stress_level",
    "resting_bp", "cholesterol", "st_depression",
    "avg_resting_hr"
]

categorical_vars = [
    "sex", "smoking_status", "alcohol_intake",
    "physical_activity", "diabetes", "hypertension",
    "family_history", "previous_heart_event"
]

# ---- Function to encode the datasets ----
def encode_data(data):

    # ---- Drop the unnecesary columns ----
    data_new=data.drop(columns=["region","income_level","st_slope","sleep_hours", "diet_quality","max_heart_rate","hrv_score","fasting_blood_sugar","resting_ecg","exercise_angina"])
    data_encoded = pd.get_dummies(data_new, columns=categorical_vars, drop_first=True)

    # ---- Convert any remaining boolean True/False outputs into clean 1s and 0s ----
    dummy_cols = [col for col in data_encoded.columns if col not in data.columns or col in categorical_vars]
    data_encoded[dummy_cols] = data_encoded[dummy_cols].astype(int)

    return data_encoded


MODEL_SUMMARY = {
    "Logistic Regression": "Logistic Regression excels by modeling the log-odds of the target variable as a linear combination of predictors, making it highly effective for multi-feature clinical datasets. By converting feature weights into well-calibrated probabilities, it balances precision and recall to achieve the highest overall predictive performance.",
    "Decision Tree": "A standalone Decision Tree is highly unstable and prone to overfitting, building rigid boundaries tailored exclusively to training data idiosyncrasies. When exposed to unseen test data, these fragile decision pathways fail to generalize, resulting in erratic, low-utility classifications.",
    "K-Nearest Neighbors (KNN)": "The KNN algorithm degrades due to the \"curse of dimensionality,\" as one-hot encoding 14 categorical features expands the dataset into a sparse, high-dimensional space. This spatial expansion distorts geometric distance metrics, preventing the model from accurately identifying true patient similarities.",
    "Naive Bayes (Gaussian)": "The Naive Bayes classifier assumes strict feature independence, ignoring natural clinical correlations among variables like age, blood pressure, and cholesterol. This structural limitation causes the model to adopt an overly conservative classification threshold, yielding high precision but a high miss rate (low recall).",
    "Support Vector Machine (SVM)": "Prior standardization via StandardScaler allowed the SVM to establish a clean geometric hyperplane to separate the patient cohorts. By optimizing for spatial distance, the model scales balanced metrics symmetrically, though it remains less effective than the probabilistic framework of Logistic Regression."
}


# ---- Load Selected Assets & Generate Predictions ----

model = load_model(MODEL_FILES[selected_model_name])
df_test = load_test_data(selected_dataset_num)


# ---- Separate your features (X) and target variable (y) ----
test_encoded = encode_data(df_test)
X_test = test_encoded.drop(columns=['heart_disease'])
y_true = test_encoded['heart_disease']

# ---- Initialize the StandardScaler ----
scaler = StandardScaler()

# ---- Fit the scaler on the training data and transform both sets ----
X_test_scaled = X_test.copy()

# ---- Fit & transform the continuous columns on Train; transform only on Test ----
X_test_scaled[continuous_vars] = scaler.fit_transform(X_test[continuous_vars])

# ---- Generate predictions and probabilities ----
y_pred = model.predict(X_test_scaled)

if hasattr(model, "predict_proba"):
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    auc_val = roc_auc_score(y_true, y_prob)
else:
    auc_val = roc_auc_score(y_true, y_pred)

# ---- Calculate all 6 performance metrics ----
acc_val = accuracy_score(y_true, y_pred)
prec_val = precision_score(y_true, y_pred, zero_division=0)
rec_val = recall_score(y_true, y_pred, zero_division=0)
f1_val = f1_score(y_true, y_pred, zero_division=0)
mcc_val = matthews_corrcoef(y_true, y_pred)

# ---- Dynamic Performance Summary Paragraph ----
st.markdown(
    f"**Model Performance Summary:** Evaluating **{selected_model_name}** on **Test Dataset {selected_dataset_num}** "
    f"yields an overall accuracy of **{acc_val:.2%}** and an AUC of **{auc_val:.4f}**. "
    f"The model achieves a recall of **{rec_val:.2%}** (identifying true positive heart disease cases) "
    f"with a precision of **{prec_val:.2%}**, resulting in a balanced MCC score of **{mcc_val:.4f}**."
    
    f"\n\n{MODEL_SUMMARY[selected_model_name]}"
)

st.divider()

# ---- Evaluation Metrics Grid ----
st.subheader(f"Evaluation Metrics for: {selected_model_name}")

col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", f"{acc_val:.4f}")
col2.metric("AUC Score", f"{auc_val:.4f}")
col3.metric("Precision", f"{prec_val:.4f}")

col4, col5, col6 = st.columns(3)
col4.metric("Recall", f"{rec_val:.4f}")
col5.metric("F1 Score", f"{f1_val:.4f}")
col6.metric("MCC Score", f"{mcc_val:.4f}")

st.divider()

# ---- Visual Confusion Matrix Plot ----
st.subheader("Confusion Matrix")

# ---- Constrain the chart width using columns so it doesn't stretch across the entire wide screen ----
plot_col, _ = st.columns([1, 2])

with plot_col:
    fig, ax = plt.subplots(figsize=(4, 4))

    # ---- Ensure clean white plot background and remove gridlines ----
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)

    ax.grid(False)
    ax.set_title(selected_model_name, fontsize=12, fontweight="bold", pad=10)

    st.pyplot(fig, use_container_width=False)
    plt.close(fig)
