import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

PARAM_GRID = [
    {
        #"smote__sampling_strategy": [.7, .74, .78, .8],
        "model__C": [1e8, 0.1, 1.0, 10.0, 100.0],
    }
]

def train_model(config: DictConfig):
    """Function to train the model"""

    print(f"Train modeling using {input_path}")
    print(f"Model used: {config.model.name}")
    print(f"Save the output to {output_path}")

    x_train, x_test, y_train, y_test = split_data(df)
    clf = logistic_regression(x_train, x_test, y_train, y_test)


def logistic_regression(X_train, X_test, y_train, y_test):
    clf = create_pipeline('simple')

    grid_search = GridSearchCV(clf, PARAM_GRID, cv=10, verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    predict_and_score(clf, X_test, y_test)

    return clf

def create_pipeline(imputer: str):
    if imputer == 'simple':
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    elif imputer == 'iterative':
        imp = IterativeImputer(max_iter=20, random_state=42)

    smote = SMOTE(random_state=500, sampling_strategy=.78)
    scaler = MinMaxScaler()
    model = LogisticRegression(solver='liblinear',
                               max_iter=100,
                               fit_intercept=True,
                               C=1e8)

    clf = Pipeline([('smote', smote),
                    ('imputer', imp),
                    ('standardize', scaler),
                    ('model', model)])

    return clf


def split_data(df):
    np.random.seed(42)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    return x_train, x_test, y_train, y_test


def predict_and_score(clf, X_test, y_test):
    # predictions and score model
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]

    roc_auc = metrics.roc_auc_score(y_test, y_pred_proba) * 100
    accuracy = metrics.accuracy_score(y_test, y_pred) * 100
    print('Training AUC: %.4f %%' % roc_auc)
    print('Training accuracy: %.4f %%' % accuracy)

    print('Confusion matrix:\n', metrics.confusion_matrix(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred))

    # create ROC curve
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label="AUC=" + str(roc_auc))
    plt.legend(loc=4)
    plt.show()


if __name__ == "__main__":
    train_model()
