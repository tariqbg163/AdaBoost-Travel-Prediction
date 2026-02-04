# Holiday Package Purchase Prediction (Random Forest \& AdaBoost)

## Overview
This project demonstrates an **end-to-end machine learning pipeline** for predicting whether a customer will take a product (`ProdTaken`) using **AdaBoost** and **Random Forest** classifiers. The project includes **data cleaning, feature engineering, model training, hyperparameter tuning, cross-validation, and prediction workflow**.

The dataset is based on travel customer information, and the goal is to predict customer behavior for targeted marketing.

---

## Project Structure

Dataset/

Travel.csv → The dataset used for training and testing the model.

notebooks/

AdaBoost_Classifier.ipynb → Jupyter notebook that shows your full workflow (EDA, preprocessing, model training, evaluation).

src/ → All your Python scripts for modular code

data_preprocessing.py → Functions for loading data, cleaning it, handling missing values, encoding & scaling features.

feature_engineering.py → Functions for creating new features like TotalVisiting.

model_training.py → Functions for training models, tuning hyperparameters, cross-validation, and evaluation.

predict.py → Functions to make predictions on new customer data using trained models.

.gitignore

Tells Git which files/folders to ignore (like venv/, __pycache__/, .ipynb_checkpoints/).

requirements.txt

Lists all Python packages your project depends on. Anyone can install them with pip install -r requirements.txt.

README.md

Project documentation with instructions, explanations, and usage examples.

venv/

Your virtual environment (ignored by Git; don’t push this folder).

LICENSE


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


