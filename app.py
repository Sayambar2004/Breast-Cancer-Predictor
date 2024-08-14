import streamlit as st
import numpy as np
import pickle

# Load the model
model = pickle.load(open('BCPnew.pkl', 'rb'))

# Title
st.title("Breast Cancer Prediction App")

# Description
st.write("""
This application uses a Deep Learning Model to predict the likelihood of breast cancer being malignant or benign based on various features.
The model is trained on the Breast Cancer Wisconsin (Diagnostic) Dataset available in Kaggle.
""")

# Sidebar for user input
st.sidebar.header("Input Features")

def user_input_features():
    texture_mean = st.sidebar.slider('Texture Mean', 0.0, 50.0, 10.0)
    area_mean = st.sidebar.slider('Area Mean', 100.0, 3000.0, 500.0)
    smoothness_mean = st.sidebar.slider('Smoothness Mean', 0.0, 0.2, 0.1)
    compactness_mean = st.sidebar.slider('Compactness Mean', 0.0, 1.0, 0.2)
    concavity_mean = st.sidebar.slider('Concavity Mean', 0.0, 1.0, 0.2)
    symmetry_mean = st.sidebar.slider('Symmetry Mean', 0.0, 0.4, 0.2)
    fractal_dimension_mean = st.sidebar.slider('Fractal Dimension Mean', 0.0, 0.1, 0.05)
    texture_se = st.sidebar.slider('Texture SE', 0.0, 5.0, 1.0)
    area_se = st.sidebar.slider('Area SE', 0.0, 100.0, 20.0)
    smoothness_se = st.sidebar.slider('Smoothness SE', 0.0, 0.1, 0.02)
    compactness_se = st.sidebar.slider('Compactness SE', 0.0, 0.3, 0.05)
    concavity_se = st.sidebar.slider('Concavity SE', 0.0, 0.4, 0.05)
    symmetry_se = st.sidebar.slider('Symmetry SE', 0.0, 0.1, 0.02)
    fractal_dimension_se = st.sidebar.slider('Fractal Dimension SE', 0.0, 0.05, 0.01)
    texture_worst = st.sidebar.slider('Texture Worst', 0.0, 60.0, 25.0)
    area_worst = st.sidebar.slider('Area Worst', 0.0, 3000.0, 1000.0)
    smoothness_worst = st.sidebar.slider('Smoothness Worst', 0.0, 0.25, 0.1)
    compactness_worst = st.sidebar.slider('Compactness Worst', 0.0, 1.5, 0.3)
    concavity_worst = st.sidebar.slider('Concavity Worst', 0.0, 2.0, 0.5)
    symmetry_worst = st.sidebar.slider('Symmetry Worst', 0.0, 0.6, 0.3)
    fractal_dimension_worst = st.sidebar.slider('Fractal Dimension Worst', 0.0, 0.1, 0.03)

    data = {
        'texture_mean': texture_mean,
        'area_mean': area_mean,
        'smoothness_mean': smoothness_mean,
        'compactness_mean': compactness_mean,
        'concavity_mean': concavity_mean,
        'symmetry_mean': symmetry_mean,
        'fractal_dimension_mean': fractal_dimension_mean,
        'texture_se': texture_se,
        'area_se': area_se,
        'smoothness_se': smoothness_se,
        'compactness_se': compactness_se,
        'concavity_se': concavity_se,
        'symmetry_se': symmetry_se,
        'fractal_dimension_se': fractal_dimension_se,
        'texture_worst': texture_worst,
        'area_worst': area_worst,
        'smoothness_worst': smoothness_worst,
        'compactness_worst': compactness_worst,
        'concavity_worst': concavity_worst,
        'symmetry_worst': symmetry_worst,
        'fractal_dimension_worst': fractal_dimension_worst,
    }

    features = np.array([list(data.values())])
    return features

# Get user input
input_df = user_input_features()

# Prediction
if st.button('Predict'):
    prediction = model.predict(input_df)
    if prediction == 1:
        st.write("The model predicts **Malignant** breast cancer.")
    else:
        st.write("The model predicts **Benign** breast cancer.")

# Display feature importance
st.header("Model Information")
st.write("""
The model was trained on a dataset of 569 entries with 21 input features. The model uses a 3 layered Artificial Neural Network giving a very impressive test accuracy of 0.98.
The Dataset was highly pre-processed before implementing the Model. The model also uses various methods to control overfitting and Vanishing Gradient Problem.
""")


# Section for the Creator's information
st.markdown("---")
st.markdown("""
### Creator Information

**Name:** Sayambar Roy Chowdhury

**Note:** This model is solely made for experimental purpose.

**LinkedIn:** https://www.linkedin.com/in/sayambar-roy-chowdhury-731b0a282/

**GitHub:** https://github.com/Sayambar2004
""")

