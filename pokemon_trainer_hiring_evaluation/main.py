from src.process import process_data
from src.train_model import train_model
from src.evaluate import evaluate_model


def main():
    """
    Main method to run model.

    Args:

    Returns:
    """
    df = process_data()
    y_test, y_pred, y_pred_proba, clf, x_train = train_model(df)
    roc_auc, accuracy, features_ranked = evaluate_model(y_test, y_pred, y_pred_proba, clf, df.columns)


if __name__ == '__main__':
    main()
