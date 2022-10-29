import pandas as pd
from sklearn import metrics


def evaluate_model(y_test, y_pred, y_pred_proba, clf, feature_names):
    """Evaluate model performance.

    Args:
        y_test:
        y_pred:
        y_pred_proba:
        clf:
        feature_names:

    Returns:
        roc_auc
        accuracy
        features_ranked
    """
    roc_auc = round(metrics.roc_auc_score(y_test, y_pred_proba) * 100, 4)
    accuracy = round(metrics.accuracy_score(y_test, y_pred) * 100, 4)
    print(f'Training AUC: {roc_auc}X')
    print(f'Training accuracy: {accuracy}%')

    print(f'Confusion matrix: {metrics.confusion_matrix(y_test, y_pred)}')
    print(metrics.classification_report(y_test, y_pred))

    features_ranked = rank_features(clf, feature_names)

    return roc_auc, accuracy, features_ranked


def rank_features(clf, feature_names):
    """Helper function for evaluate_model. Rank features by feature importance.

    Args:
        clf:
        feature_names:

    Returns:
        features_ranked
    """
    features_ranked = pd.DataFrame()
    import pdb; pdb.set_trace()
    features_ranked['features'] = feature_names[1:]
    features_ranked['importance'] = clf.named_steps['model'].feature_importances_

    features_ranked.sort_values(by='importance', ascending=False, inplace=True)
    print(features_ranked.head(n=10))

    return features_ranked


if __name__ == '__main__':
    evaluate_model()