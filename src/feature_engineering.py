# src/feature_engineering.py
import pandas as pd

def create_features(df):
    """Create new features and drop old ones."""
    if 'NumberOfChildrenVisiting' in df.columns and 'NumberOfPersonVisiting' in df.columns:
        df['TotalVisiting'] = df['NumberOfChildrenVisiting'] + df['NumberOfPersonVisiting']
        df.drop(columns=['NumberOfChildrenVisiting', 'NumberOfPersonVisiting'], inplace=True)
    return df
