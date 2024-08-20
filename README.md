# Breast Cancer Predictor

## Overview

This repository contains the code and resources for a **Breast Cancer Prediction Model** built using the Wisconsin Breast Cancer Dataset available on Kaggle. The project leverages an Artificial Neural Network (ANN) to predict the likelihood of breast cancer based on various input features. An interactive web interface has been developed using Streamlit to visualize the predictions.

## Features

- **Highly Preprocessed Data:** Only the most relevant 21 features were retained after thorough preprocessing, feature importance analysis, and correlation checks.
- **Artificial Neural Network (ANN):** The model consists of:
  - 1 Input Layer
  - 3 Hidden Layers
  - 1 Output Layer
- **Model Performance:**
  - Total Params: 7,301
  - Trainable Params: 2,433
  - Optimizer Params: 4,868
  - Test Accuracy: 98%
- **Optimization Techniques:**
  - Fine-tuned using various optimizers, activation functions, and architectures.
  - Employed early stopping and dropout layers to prevent overfitting and vanishing gradient issues.

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Sayambar2004/Breast-Cancer-Predictor.git
   cd Breast-Cancer-Predictor
   ```

2. **Install the Required Packages:**
   Ensure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit App:**
   ```bash
   streamlit run app.py
   ```

## Usage

- The Streamlit app will provide an interactive interface for users to input the required features and see the model's prediction.
- The model is trained on the Wisconsin Breast Cancer Dataset, making it highly accurate (98% test accuracy).
- While the model is not intended for end-user applications due to the number of input features required, the app serves as a powerful visualization tool for educational and experimental purposes.

## Dataset

The dataset used in this project is the Wisconsin Breast Cancer Dataset, available on [Kaggle](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data).

## Model Summary

- **Input Layer:** 21 features
- **Hidden Layers:** 3 fully connected layers
- **Output Layer:** Binary classification (0: Benign, 1: Malignant)
- **Optimizer:** Fine-tuned using various configurations
- **Activation Functions:** Experimented with ReLU, sigmoid, and others
- **Regularization:** Early stopping and dropout layers

## Repository Structure

```
Breast-Cancer-Predictor/
│
├── app.py                     # Streamlit app script
├── model/                     # Directory containing the trained model
│   └── BCPnew.pkl             # Trained ANN model (Pickle file)
├── data/                      # Directory containing the dataset
│   └── breast_cancer_data.csv # Preprocessed dataset
├── requirements.txt           # Required Python packages
└── README.md                  # Project documentation
```

## Live Demo

Check out the live demo of the app here: [Breast Cancer Predictor Web App](https://breast-cancer-predictor-sayambarroychowdhury.streamlit.app/)

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to check out the [issues page](https://github.com/Sayambar2004/Breast-Cancer-Predictor/issues).

## License

This project is open-source and available under the [MIT License](LICENSE).

## Contact

**Sayambar Roy Chowdhury**  
[LinkedIn](https://www.linkedin.com/in/sayambar-roy-chowdhury/) | [GitHub](https://github.com/Sayambar2004)

