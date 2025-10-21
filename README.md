# Credit-Card-Fraud-Detection

This repository contains a Jupyter notebook for detecting credit card fraud using logistic regression in Python. The project addresses class imbalance by undersampling non-fraudulent transactions, trains the model on balanced data, and evaluates performance with accuracy scores on train and test sets.

## Overview
- **Project_10_Credit_Card_Fraud_Detection.ipynb**: Loads the dataset, explores data (e.g., class distribution), separates fraudulent and legitimate transactions, undersamples legitimate ones (to 492 samples matching fraud), splits data, trains a logistic regression model, and computes accuracy (~94%).
- Libraries: NumPy, Pandas, scikit-learn (train_test_split, LogisticRegression, accuracy_score).
- Dataset: Anonymized credit card transactions with PCA-transformed features (V1-V28), Time, Amount, and Class (0=legitimate, 1=fraudulent).

Key Steps:
- Data Loading and Exploration: Checks for missing values, class distribution (highly imbalanced: ~99.8% legitimate).
- Balancing: Randomly samples 492 legitimate transactions to match fraud count.
- Model: Logistic Regression (default parameters).
- Evaluation: Accuracy on train (~94%) and test (~94%) data.

## Prerequisites
- Python 3.x
- Libraries: `numpy`, `pandas`, `scikit-learn`
