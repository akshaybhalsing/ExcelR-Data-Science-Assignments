import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Load Dataset
df = pd.read_csv("diabetes.csv")

# Features and Target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Train Model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

st.title("Diabetes Prediction App")

st.write("Enter Patient Details")

preg = st.number_input("Pregnancies", 0, 20, 1)
glu = st.number_input("Glucose", 0, 200, 120)
bp = st.number_input("Blood Pressure", 0, 150, 70)
skin = st.number_input("Skin Thickness", 0, 100, 20)
insulin = st.number_input("Insulin", 0, 900, 80)
bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.number_input("Age", 1, 100, 30)

if st.button("Predict"):
    prediction = model.predict([[preg, glu, bp, skin, insulin, bmi, dpf, age]])

    if prediction[0] == 1:
        st.success("Diabetic")
    else:
        st.success("Not Diabetic")