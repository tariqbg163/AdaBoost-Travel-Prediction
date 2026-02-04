# AdaBoost-Travel-Prediction

## Overview
This project demonstrates an **end-to-end machine learning pipeline** for predicting whether a customer will take a product (`ProdTaken`) using **AdaBoost** and **Random Forest** classifiers. The project includes **data cleaning, feature engineering, model training, hyperparameter tuning, cross-validation, and prediction workflow**.

The dataset is based on travel customer information, and the goal is to predict customer behavior for targeted marketing.

---

## Project Structure

AdaBoost-Travel-Prediction/
│
├── Dataset/
│ └── Travel.csv # CSV dataset
│
├── notebooks/
│ └── AdaBoost_Classifier.ipynb # Jupyter notebook demonstrating workflow
│
├── src/
│ ├── data_preprocessing.py # Load, clean, fill missing values, encode & scale features
│ ├── feature_engineering.py # Create derived features (e.g., TotalVisiting)
│ ├── model_training.py # Train models, hyperparameter tuning, cross-validation
│ └── predict.py # Make predictions on new customer data
│
├── .gitignore # Ignore venv, checkpoints, dataset (optional)
├── requirements.txt # Python packages for the project
├── README.md # Project description & instructions
├── venv/ # Virtual environment (ignored)
└── LICENSE # Optional license


---

## Dataset
- **CustomerID**: Unique identifier  
- **Age**: Age of the customer  
- **MaritalStatus**: Married / Unmarried  
- **Gender**: Male / Female  
- **DurationOfPitch**: Duration of the sales pitch  
- **TypeofContact**: How customer was contacted  
- **PreferredPropertyStar**: Customer preference  
- **NumberOfTrips**: Number of trips taken  
- **NumberOfChildrenVisiting** + **NumberOfPersonVisiting** → Combined into `TotalVisiting`  
- **MonthlyIncome**: Income of customer  
- **ProdTaken**: Target variable (0 = No, 1 = Yes)  

> **Note:** Dataset is located in `Dataset/Travel.csv`. You may use a sample if sensitive.

---

## Features & Preprocessing
- Handle missing values (median for numeric, mode for categorical)  
- Standardize categorical values (e.g., `Gender`, `MaritalStatus`)  
- Feature engineering: combine `NumberOfChildrenVisiting` + `NumberOfPersonVisiting` → `TotalVisiting`  
- Encode categorical features using **OneHotEncoder**  
- Scale numerical features using **StandardScaler**  

---

## Models
- Logistic Regression  
- Decision Tree Classifier  
- Random Forest Classifier  
- AdaBoost Classifier  
- Gradient Boosting Classifier  

**Hyperparameter tuning** is done using `RandomizedSearchCV` for Random Forest and AdaBoost.

---

## Model Performance

| Model | Training Accuracy | Testing Accuracy |
|-------|-----------------|----------------|
| Random Forest | 0.999 | 0.928 |
| AdaBoost      | 0.847 | 0.836 |

- **Random Forest**: highest accuracy, slightly overfits  
- **AdaBoost**: more balanced performance, easier to interpret  

---


