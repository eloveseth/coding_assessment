import pandas as pd

def evaluate_model(clf, x_test, y_test):
    roc_auc = metrics.roc_auc_score(y_test, y_pred_proba) * 100
    accuracy = metrics.accuracy_score(y_test, y_pred) * 100
    print(f'Training AUC: %.4f %%{roc_auc}')
    print(f'Training accuracy: %.4f %%{accuracy}')

    print(f'Confusion matrix: {metrics.confusion_matrix(y_test, y_pred)}')
    print(metrics.classification_report(y_test, y_pred))

    # create ROC curve
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label="AUC=" + str(roc_auc))
    plt.legend(loc=4)
    plt.show()


def rank_features(clf: , x_train: ):
    features_ranked = pd.DataFrame()
    features_ranked['features'] = x_train.columns
    features_ranked['importance'] = clf.feature_importances_

    print(features_ranked.sort_values(by='importance', ascending=False, inplace=True))

    return features_ranked


if __name__ == '__main__':
    evaluate_model()