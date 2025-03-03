# CSE150A_Proj
# AI Agent Overview

Our AI agent is designed to analyze user-uploaded data, train a model specific to that data, and provide meaningful insights. It operates under the PEAS framework as follows:

Performance Measure: The effectiveness of the model is measured based on accuracy, precision, recall, and other relevant evaluation metrics.

Environment: The agent operates in a data-driven world where users provide datasets, and the AI processes them to generate predictive insights.

Actuators: The agent modifies model parameters, optimizes training, and outputs predictions or insights.

Sensors: It ingests structured and unstructured data, processes it, and refines its model based on training results.

# Agent Type

The agent is a goal-based learning system, meaning it seeks to improve its predictive capabilities iteratively. It fits within probabilistic modeling by utilizing statistical techniques such as Bayesian inference, regression models, and neural networks (depending on the dataset).

# Data Exploration and Preprocessing

The dataset used is the Telecom Customer Churn Dataset which is a collection of data on 7,043 customers. Our goal is to predict if a customer is likely to churn (leave the service) using data about demographic, usage and billing. Various key factors play into churn prediction: demographic information (gender, partner status, dependents and senior citizen status), customer information (contract length, payment method, and tenure), service usage (internet service usage, phone service usage and streaming TV status) and billing information (monthly accumulated charges and total charges paid). In order to better visualize the data, we plot the churn distribution in our dataset (using the code in dataExploration.py):

![Churn Distribution](churnDistribution.jpg)


In order to work with the data we do thing major things: load and preprocess the data, feature engineering and training and evaluation. Firstly, we preprocess the data by converting the TotalCharges variable to a numerical format, drop missing variables and convert categorical variables into numerical values. For feature engineering we disregard customerID since it is not relevant for our churn predictions. We then split the data into features (denoted X) and target (denoted Y). At the training and evaluation stage we split the data into 80% training and 20% testing. 

Our model has dependencies on a few python libraries and preprocessing steps in the code. We use pandas, numpy, sklearn.model_selection.train_test_split (to split the data into training and testing) and imblearn.over_sampling.SMOTE (to balance the dataset by oversampling the smaller class). 

# Agent Setup & Probabilistic Modeling

Our AI system follows these steps:

Data Ingestion: The system accepts user-uploaded datasets in various formats.

Data Preprocessing: Missing values are handled, and categorical variables are encoded.

Model Training: A machine learning model is trained on the dataset.

Evaluation: The trained model is evaluated based on relevant performance metrics.

Deployment: The model is stored and can be used for future predictions.

# Model Training

To train the first model:

Load the dataset.

Preprocess the data (handle missing values, encoding, scaling, etc.).

Split the data into training and testing sets.

Select and train a machine learning model (e.g., logistic regression, decision tree, neural network).

Optimize hyperparameters for better performance.

# Repository and Code Links

All code and Jupyter notebooks have been uploaded and are accessible at:
[GitHub Repository](https://github.com/pranshug2704/CSE150A_Proj)

# Possible Improvements

Feature Engineering: Add more relevant features to improve predictive accuracy.

Hyperparameter Tuning: Experiment with different model parameters.

Model Selection: Try alternative models like ensemble methods or deep learning.

Data Augmentation: Increase dataset size through synthetic data generation or augmentation.
