import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Flower Classifier", page_icon="🌸")
st.title("🌸 Synthetic Flower Species Classifier")
st.write("SVM with tuned RBF kernel")

model = joblib.load("svm_flower_model.joblib")

sepal_length = st.number_input("Sepal length (cm)", min_value=4.0, max_value=8.0, value=5.8, step=0.01)
sepal_width = st.number_input("Sepal width (cm)", min_value=2.0, max_value=4.5, value=3.0, step=0.01)
petal_length = st.number_input("Petal length (cm)", min_value=1.0, max_value=7.0, value=4.0, step=0.01)
petal_width = st.number_input("Petal width (cm)", min_value=0.1, max_value=2.8, value=1.3, step=0.01)

input_df = pd.DataFrame([{
    "sepal_length_cm": sepal_length,
    "sepal_width_cm": sepal_width,
    "petal_length_cm": petal_length,
    "petal_width_cm": petal_width
}])

if st.button("Predict species"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted species: {prediction}")
