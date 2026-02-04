# src/predict.py
import numpy as np

def predict_model(model, preprocessor, new_data):
    """
    Make predictions on new raw input data.
    
    Parameters:
    - model: trained model (e.g., AdaBoostClassifier or RandomForestClassifier)
    - preprocessor: ColumnTransformer used to preprocess training data
    - new_data: pandas DataFrame with same columns as training features (before preprocessing)
    
    Returns:
    - predictions: numpy array of predicted labels
    """
    # Preprocess new data
    X_new = preprocessor.transform(new_data)
    
    # Make predictions
    predictions = model.predict(X_new)
    
    # Optional: probability predictions
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_new)
        return predictions, probabilities
    
    return predictions
