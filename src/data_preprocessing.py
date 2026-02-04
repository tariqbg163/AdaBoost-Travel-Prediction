# src/data_preprocessing.py
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

def load_data(path):
    """Load CSV dataset."""
    path = "Dataset\Travel.csv"
    return pd.read_csv(path)

def clean_data(df):
    """Clean categorical features and standardize values."""
    # Standardize MaritalStatus
    df['MaritalStatus'] = df['MaritalStatus'].replace('Single', 'Unmarried')

    # Standardize Gender
    df["Gender"] = df["Gender"].replace({
        "Fe Male": "Female",
        "M": "Male",
        "F": "Female"
    })

    # Drop irrelevant column
    if 'CustomerID' in df.columns:
        df.drop(columns=['CustomerID'], inplace=True)
    
    return df

def handle_missing_values(df):
    """Fill missing values for numeric and categorical columns."""
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['TypeofContact'].fillna(df['TypeofContact'].mode()[0], inplace=True)
    df['DurationOfPitch'].fillna(df['DurationOfPitch'].median(), inplace=True)
    df['NumberOfFollowups'].fillna(df['NumberOfFollowups'].mode()[0], inplace=True)
    df['PreferredPropertyStar'].fillna(df['PreferredPropertyStar'].mode()[0], inplace=True)
    df['NumberOfTrips'].fillna(df['NumberOfTrips'].median(), inplace=True)
    df['NumberOfChildrenVisiting'].fillna(df['NumberOfChildrenVisiting'].mode()[0], inplace=True)
    df['MonthlyIncome'].fillna(df['MonthlyIncome'].median(), inplace=True)
    
    return df

def preprocess_features(df, target='ProdTaken'):
    """Split dataset into features X and target y, encode categorical and scale numerical."""
    X = df.drop(columns=[target])
    y = df[target]

    # Separate categorical and numerical features
    cat_features = X.select_dtypes(include='O').columns
    num_features = X.select_dtypes(exclude='O').columns

    numeric_transformer = StandardScaler()
    ohe_transformer = OneHotEncoder(drop='first')

    preprocessor = ColumnTransformer([
        ("OneHotEncoder", ohe_transformer, cat_features),
        ("StandardScaler", numeric_transformer, num_features)
    ])

    X_transformed = preprocessor.fit_transform(X)
    return X_transformed, y, preprocessor

