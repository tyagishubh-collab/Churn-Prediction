import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score
)

import pickle
import streamlit as st
import feature_engine
import shap

st.set_option('deprecation.showPyplotGlobalUse', False)

st.header("Customer Churn Prediction")

st.sidebar.title("Customer Churn Prediction")
st.sidebar.markdown("Explanation with SHAP")

select_display = st.sidebar.radio(
    "Select Analysis",
    options=["Data", "Model Results", "Feature Importance", "Individual Results", "What IF Analysis"]
)


# -------------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("Churn.csv", na_values=" ")
    return data

data = load_data()
data = data.dropna()

selected_features = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'TechSupport',
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges'
]

# -------------------------------------------------------------------
# DISPLAY DATA TAB
# -------------------------------------------------------------------
if select_display == "Data":
    n_rows = st.slider("Select No. of Rows to Display",
                       min_value=10, max_value=len(data), value=10, step=10)
    st.dataframe(data[selected_features + ['Churn']].head(n_rows))


# -------------------------------------------------------------------
# LOAD MODEL + ENCODER
# -------------------------------------------------------------------
@st.cache_resource
def load_label_encoder():
    with open("encoder.pkl", 'rb') as file:
        return pickle.load(file)

@st.cache_resource
def load_model():
    with open("model.pkl", 'rb') as file:
        return pickle.load(file)

label_encoder = load_label_encoder()
model = load_model()

features = data[selected_features]
target = data['Churn'].map({'No': 0, 'Yes': 1})

# Encode features
features = label_encoder.transform(features)


# -------------------------------------------------------------------
# MODEL RESULTS TAB
# -------------------------------------------------------------------
if select_display == "Model Results":

    metric = st.selectbox("Select the Metric",
                          ["Confusion Matrix", "ROC-AUC", "Precision-Recall"])

    predictions = model.predict_proba(features)[:, 1]

    # ----------------------------------------------------------
    # CONFUSION MATRIX
    # ----------------------------------------------------------
    if metric == "Confusion Matrix":

        threshold = st.slider("Select Threshold", 0.0, 1.0, 0.5, 0.1)
        pred_labels = (predictions > threshold).astype(int)

        col1, col2 = st.columns((2, 1))

        with col1:
            fig, ax = plt.subplots(figsize=(3, 3))
            conf_mat = confusion_matrix(target, pred_labels)

            sns.heatmap(conf_mat, cmap='Blues', annot=True, fmt='d',
                        xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'], ax=ax)

            plt.xlabel("Predicted")
            plt.ylabel("True")
            st.pyplot(fig)

        with col2:
            st.markdown(f"**F1 Score:** {round(f1_score(target, pred_labels), 3)}")
            st.markdown(f"**Precision:** {round(precision_score(target, pred_labels), 3)}")
            st.markdown(f"**Recall:** {round(recall_score(target, pred_labels), 3)}")

    # ----------------------------------------------------------
    # ROC CURVE
    # ----------------------------------------------------------
    if metric == "ROC-AUC":
        fpr, tpr, _ = roc_curve(target, predictions)
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.plot(fpr, tpr, label="ROC Curve")
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
        st.pyplot(fig)

    # ----------------------------------------------------------
    # PRECISION-RECALL CURVE
    # ----------------------------------------------------------
    if metric == "Precision-Recall":
        precision, recall, _ = precision_recall_curve(target, predictions)
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.plot(recall, precision, label="Precision-Recall Curve")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend()
        st.pyplot(fig)


# -------------------------------------------------------------------
# FEATURE IMPORTANCE TAB
# -------------------------------------------------------------------
@st.cache_resource
def load_SHAP_VALUES():
    with open('shap_values.pkl', 'rb') as file:
        return pickle.load(file)

if select_display == "Feature Importance":
    shap_values = load_SHAP_VALUES()
    st.markdown("**Feature Importance with SHAP**")
    shap.summary_plot(shap_values.values[:, :, 1],
                      plot_type='bar', feature_names=selected_features)
    st.pyplot()


# -------------------------------------------------------------------
# INDIVIDUAL RESULTS TAB
# -------------------------------------------------------------------
if select_display == "Individual Results":

    customer_id = st.selectbox("Select a Customer ID", data.customerID.values)
    index = data[data.customerID == customer_id].index.values[0]

    churn_prob = model.predict_proba(features[index].reshape(1, -1))[:, 1].item()

    if st.button("Submit"):

        st.write("Customer Info")
        st.write(data.iloc[index][selected_features].to_frame().T)

        col1, col2 = st.columns((3, 1))

        with col1:
            st.write("Waterfall Chart")
            shap_values = load_SHAP_VALUES()
            shap.plots.waterfall(shap_values[index][:, 1], max_display=15)
            st.pyplot()

        with col2:
            st.markdown(f"**Actual Churn:** {data.iloc[index]['Churn']}")
            st.markdown(f"**Predicted Churn Probability:** {round(churn_prob, 3)}")


# -------------------------------------------------------------------
# WHAT IF ANALYSIS TAB
# -------------------------------------------------------------------
if select_display == "What IF Analysis":

    st.subheader("Select Feature Values")

    input_data = pd.DataFrame(columns=selected_features)

    for feat in selected_features:
        if (data[feat].dtype == 'O') or (len(data[feat].unique()) < 10):
            input_data.loc[0, feat] = st.selectbox(feat, data[feat].unique())
        else:
            input_data.loc[0, feat] = st.number_input(feat, min_value=0)

    if st.button("Submit"):
        st.markdown("**Selected Customer Info**")
        st.write(input_data)

        encoded = label_encoder.transform(input_data)
        pred = model.predict_proba(encoded)[:, 1].item()
        st.write(f"Predicted Churn Probability: {round(pred, 3)}")

        col1, col2 = st.columns((1, 1.2))

        explainer = shap.TreeExplainer(model)
        shap_vals = explainer(encoded)

        with col1:
            st.markdown("**Variable Importance**")
            shap.summary_plot(shap_vals.values[:, :, 1],
                              plot_type='bar', feature_names=selected_features)
            st.pyplot()

        with col2:
            st.markdown("**Waterfall Chart**")
            shap.plots.waterfall(shap_vals[0][:, 1], max_display=15)
            st.pyplot()
