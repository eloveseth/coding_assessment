import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.model_selection import GridSearchCV, train_test_split
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import xgboost as xgb

PARAM_GRID = [{
    'model__max_depth': [4, 6],
    # 'model__min_child_weight': [4, 6],
}]


def train_model(df: pd.DataFrame):
    """Train the model using xgboost and a custom cost function.

    Args:
        df: processed dataset used to train model

    Returns:
        y_pred: model predictions
        y_pred_proba: model prediction probabilities
    """
    np.random.seed(42)

    x_train, x_test, y_train, y_test = create_train_test_split(df)

    clf = create_pipeline()

    grid_search = GridSearchCV(clf, PARAM_GRID, cv=10, verbose=1, n_jobs=-1)
    grid_search.fit(x_train, y_train)

    clf = grid_search.best_estimator_
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    y_pred_proba = clf.predict_proba(x_test)[:, 1]

    return y_pred, y_pred_proba


def create_pipeline():
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    smote = SMOTE(random_state=500, sampling_strategy=.78)
    scaler = MinMaxScaler()
    model = xgb.XGBClassifier(
        objective=custom_se
    )

    clf = Pipeline([
        ('imputer', imp),
        ('smote', smote),
        ('standardize', scaler),
        ('model', model)
    ])

    return clf


def create_train_test_split(df: pd.DataFrame):
    # split data into training and test
    x = df.drop("hired", axis=1).values
    y = df.hired.values

    x_train, x_test, y_train, y_test = train_test_split(x, y)

    return x_train, x_test, y_train, y_test


def custom_se(y_pred, y_true):
    """Custom squared error objective, using a simplified version of MSE.

    Args:
        y_pred: predictions
        y_true: ground truth values

    Returns:
        grad:
        hess:
    """

    grad = gradient_se(y_pred, y_true)
    hess = hessian_se(y_pred, y_true)

    return grad, hess


def gradient_se(y_pred, y_true):
    """Helper function for custom_se. Computes gradient squared error.

    Args:
        y_pred: predictions
        y_true: ground truth values

    Returns:
        grad:
    """
    grad = 2*(y_pred - y_true)

    return grad


def hessian_se(y_pred, y_true):
    """Helper function for custom_se. Computes hessien.

    Args:
        y_pred: predictions
        y_true: ground truth values

    Returns:
        hess:
    """
    hess = 0*y_true + 2

    return hess



if __name__ == "__main__":
    train_model()
