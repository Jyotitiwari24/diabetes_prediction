# Diabetes Prediction ML Model 🩺

A machine learning project that predicts **whether a patient has diabetes** using clinical features. Built with **Python, Pandas, and Scikit-learn**. The model uses **Logistic Regression** and provides accuracy metrics for training and testing datasets.

---

## Features
- Predicts diabetes based on features like age, BMI, glucose, blood pressure, insulin, etc.
- Data preprocessing: handles missing values, feature scaling, and encoding.
- Logistic Regression model with **train/test split evaluation**.
- Outputs model accuracy and can predict new patient data.
- Easy to extend with other ML models like Random Forest or XGBoost.

---

## Dataset
- Based on **Kaggle Pima Indians Diabetes Dataset** (or any CSV dataset you used).  
- CSV contains features such as:

Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome


- `Outcome` = 1 indicates diabetes, 0 indicates no diabetes.

---

## Project Structure
diabetes-prediction/
├── data/
│ └── train.csv # Dataset CSV
├── diabetes_prediction.py # Main ML script
├── model/
│ └── model.pkl # Saved trained model (optional)
├── requirements.txt # Dependencies
└── README.md # Project description


---

## Installation & Usage

### 1. Clone Repo
```bash
git clone https://github.com/yourusername/diabetes-prediction.git
cd diabetes-prediction
2. Install Dependencies
pip install -r requirements.txt
3. Run Prediction Script
python diabetes_prediction.py
Outputs model accuracy for training and test data.

Can test new patient data using model.predict().

How it Works
Load CSV dataset into Pandas DataFrame.

Handle missing values and preprocess features.

Split data into X (features) and Y (labels).

Train Logistic Regression model.

Evaluate accuracy on training and test sets.

Predict new data by transforming it with preprocessing steps.

Dependencies
Python 3.8+

pandas

numpy

scikit-learn
