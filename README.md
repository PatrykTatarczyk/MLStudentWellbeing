# 🧠 Student Well-being Classifier: Predicting High-Stress Levels

## 📌 Project Overview
This project focuses on building and evaluating machine learning classification models to predict high perceived stress levels among students. Rather than relying on a single algorithm, this script acts as an automated evaluation pipeline, comparing four distinct ML models. It systematically tests how various hyperparameters impact model accuracy and visualizes the trade-off between training and testing performance to identify overfitting.

## 🎯 Analytical Objectives
* Predict whether a student falls into the "High Perceived Stress" category based on behavioral and academic indicators.
* Automate the hyperparameter tuning process across multiple algorithms.
* Generate visual reports (Accuracy vs. Hyperparameter values) to make data-driven decisions on model selection.

## 🛠️ Tech Stack & Libraries
* **Language:** Python
* **Data Handling:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn` (StandardScaler, train_test_split, metrics)
* **Algorithms:** K-Nearest Neighbors (KNN), Random Forest, Support Vector Machine (SVM), Decision Tree
* **Data Visualization:** `matplotlib`

## ⚙️ Methodology & Hyperparameter Tuning

### 1. Data Preprocessing
* **Feature Selection:** Removed direct identifiers and leak variables (like direct anxiety/depression scores) to ensure the model learns from underlying behavioral patterns, not direct psychological labels.
* **Scaling:** Applied `StandardScaler` to normalize feature distributions, which is strictly required for distance-based models like KNN and SVM.

### 2. Model Evaluation Pipeline
The script systematically evaluates the following models and hyperparameters:
* **K-Nearest Neighbors (KNN):** Tuned the number of neighbors (`n_neighbors`), distance metrics (`euclidean`, `manhattan`, `chebyshev`), and custom weight functions (uniform, distance, $1/d^2$, $1/\sqrt{d}$).
* **Random Forest:** Evaluated ensemble robustness by tuning the number of trees (`n_estimators`), max tree depth (`max_depth`), and minimum samples per leaf (`min_samples_leaf`).
* **Support Vector Machine (SVM):** Tested various geometric boundaries by changing the `kernel` (linear, rbf, poly, sigmoid), regularization parameter (`C`), and polynomial degrees.
* **Decision Trees:** Analyzed split criteria (`gini` vs `entropy`), depth limits, and node split
