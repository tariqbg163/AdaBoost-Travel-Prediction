# src/model_training.py
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, cross_validate, RandomizedSearchCV

def train_models(X_train, y_train, X_test, y_test):
    """Train multiple models and print performance."""
    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        "Gradient Boost": GradientBoostingClassifier()
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        print(f"{name} Accuracy: {acc:.4f}")
    return results

def tune_hyperparameters(X_train, y_train):
    """Perform RandomizedSearchCV for RF and AB."""
    rf_params = {
        "max_depth": [5, 8, 10, 15, None],
        "max_features": [5, 7, 8, "auto"],
        "min_samples_split": [2, 8, 15, 20],
        "n_estimators": [100, 200, 500, 1000]
    }
    ab_params = {
        "n_estimators": [50, 60, 70, 80, 90],
        "algorithm": ["SAMME", "SAMME.R"]
    }

    randomcv_models = [
        ("RF", RandomForestClassifier(), rf_params),
        ("AB", AdaBoostClassifier(), ab_params)
    ]

    best_models = {}
    for name, model, params in randomcv_models:
        search = RandomizedSearchCV(estimator=model, param_distributions=params,
                                    n_iter=30, cv=3, verbose=0, n_jobs=-1)
        search.fit(X_train, y_train)
        best_models[name] = search.best_estimator_
        print(f"Best {name} Params: {search.best_params_}")

    return best_models

def cross_validate_models(models, X_train, y_train):
    scoring = ['accuracy', 'precision_weighted', 'recall_weighted']
    for name, model in models.items():
        results = cross_validate(model, X_train, y_train, cv=5, scoring=scoring)
        print(f"Model: {name}")
        print(f"Accuracy: {results['test_accuracy'].mean():.4f}")
        print(f"Precision: {results['test_precision_weighted'].mean():.4f}")
        print(f"Recall: {results['test_recall_weighted'].mean():.4f}")
        print("-"*50)
