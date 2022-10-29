from src.process import process_data
from src.train_model import train_model


def main():
    """
    Main method to run model.

    Args:

    Returns:
    """
    df = process_data()
    train_model(df)




if __name__ == '__main__':
    main()
