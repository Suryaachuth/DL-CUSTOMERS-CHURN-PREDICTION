# DL-CUSTOMERS-CHURN-PREDICTION
# Overview
This project focuses on predicting customer churn using an Artificial Neural Network (ANN) model. Customer churn is a critical problem for businesses, and predicting it accurately can help in retaining customers and making informed business decisions.
## Background
Customer churn is the phenomenon where customers discontinue their relationship with a company or service. It can be costly for businesses as it impacts revenue and customer loyalty. This project aims to build an ANN model to predict customer churn based on historical customer data.

## Requirements
- Python 3.7 (or higher)
- TensorFlow 2.0 (or higher)
- Keras 2.6 (or higher)
- Pandas
- NumPy
- Scikit-learn

## Installation
To install the required dependencies, run the following command:

```bash
pip install tensorflow keras pandas numpy scikit-learn
Usage
To use the churn prediction model, follow these steps:

Prepare your dataset in a CSV format with features and the target variable (churn label).
Run the script to train the ANN model on your dataset.
Use the trained model to predict customer churn for new data.
Dataset
The dataset used for this project contains historical customer data, including various features such as age, gender, usage patterns, customer complaints, etc. The target variable is the "churn" label, indicating whether a customer churned (1) or not (0). The dataset was preprocessed to handle missing values and convert categorical variables to numerical representations.

Model Architecture
The ANN model architecture used in this project consists of three dense layers with ReLU activation functions. The number of units in each layer is as follows:

Input layer: 19 units (corresponding to the number of features)
Hidden layer 1: 20 units with ReLU activation
Hidden layer 2: 15 units with ReLU activation
Hidden layer 3: 20 units with ReLU activation
Output layer: 1 unit with Sigmoid activation (for binary classification)
The model also includes a dropout layer with a dropout rate of 0.3 for regularization to reduce overfitting.

Training
The ANN model is trained using the Adam optimizer with a binary cross-entropy loss function. The batch size is set to 32, and the model is trained for 200 epochs.

Evaluation
The model's performance is evaluated on a separate test dataset using the following metrics:

Precision
Recall
F1-score
Accuracy
Results
The model achieved the following performance metrics on the test dataset:

Precision for class 0 (negative class): 0.79
Precision for class 1 (positive class): 0.69
Recall for class 0 (negative class): 0.92
Recall for class 1 (positive class): 0.43
F1-score for class 0 (negative class): 0.85
F1-score for class 1 (positive class): 0.53
Accuracy: 0.77
Contributing
Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please create a new issue or submit a pull request.







