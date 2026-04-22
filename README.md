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
* **Decision Trees:** Analyzed split criteria (`gini` vs `entropy`), depth limits, and node split conditions.

### 3. Visual Reporting
For every hyperparameter tested, the script automatically plots and saves a `.png` chart comparing Training Accuracy vs. Testing Accuracy. This allows for immediate visual identification of the model's "sweet spot" before overfitting occurs.

## 📂 Project Structure
* `Raw_Data.csv` - The dataset containing student well-being indicators.
* `model_evaluation.py` - The main automated ML pipeline script.
* `*.png` files - Automatically generated charts showing hyperparameter impact (e.g., `rf_max_depth.png`, `svm_kernel.png`).

## 💡 How to Run
1. Ensure the dataset (`Raw_Data.csv`) is in the same directory as the script.
2. Install required packages: `pip install pandas numpy scikit-learn matplotlib`.
3. Run the script: `python model_evaluation.py`. The console will output live accuracy logs, and the charts will be saved directly to the folder.
