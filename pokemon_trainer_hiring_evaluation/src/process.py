import pandas as pd
import numpy as np

from utils import read_csv_to_dataframe


def process_data():
    """Function to process raw data.

    Args:

    Returns:
        processed_data
    """

    # load data
    pokemon = read_csv_to_dataframe("./data/ds_pokemon_names.csv")
    trainer = read_csv_to_dataframe("./data/ds_pokemon_trainer_application_data.csv")

    # clean data

    # merge data


def clean_pokemon(df: pd.DataFrame) -> df.Dataframe:

def clean_trainer(df: pd.DataFrame) -> df.Dataframe:



if __name__ == '__main__':
    process_data()
