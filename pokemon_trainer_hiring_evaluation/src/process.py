import pandas as pd
from typing import List

from utils import read_csv_to_dataframe, extract_pay_type, extract_pay_amt

PAY_RATE = {'RecentTrainingExperience1PayRateEnd': ['pay_type_1', 'pay_amt_1'],
            'RecentTrainingExperience2PayRateEnd': ['pay_type_2', 'pay_amt_2'],
            'RecentTrainingExperience3PayRateEnd': ['pay_type_3', 'pay_amt_3'],
            'RecentTrainingExperience4PayRateEnd': ['pay_type_4', 'pay_amt_4']}

FEATURES = [

]
def process_data() -> pd.Dataframe:
    """Function to process raw trainer data.

    Args:

    Returns:
        processed_data
    """
    # load data
    trainer = read_csv_to_dataframe("./data/ds_pokemon_trainer_application_data.csv")

    # feature selection and feature engineering
    trainer_processed = select_features(trainer, FEATURES)

    trainer_pay_rates = clean_trainer_pay_rates(trainer)

    trainer_processed = pd.merge(trainer_pay_rates,
                                 on=)

    return trainer_processed


def clean_trainer_pay_rates(trainer: pd.DataFrame) -> df.Dataframe:
    """Helper function for process_data. Extracts pay rates and pay types from trainer set. Converts all pay to
    similar scale for inclusion in model.

    Args:
        trainer: raw unprocessed dataset

    Returns:
        processed_data
    """
    trainer_rates = trainer[PAY_RATE.keys()]

    for key, _ in PAY_RATE.items():
        trainer_rates[key] = trainer_rates[key].apply(str)

    for key, value in PAY_RATE.items():
        trainer_rates[value[0]] = trainer_rates[key].apply(lambda row: extract_pay_type(pay_rate=row))

    for key, value in PAY_RATE.items():
        trainer_rates[value[1]] = trainer_rates[key].apply(lambda row: extract_pay_amt(pay_rate=row))
        trainer_rates[value[1]] = pd.to_numeric(trainer_rates[value[1]], errors='coerce')

    return trainer_rates


def select_features(trainer: pd.Dataframe, cols: List[str]) -> df.DataFrame:
    """Helper function for process_data. Selects list of features from raw trainer dataset.

    Args:
        trainer: raw unprocessed dataset

    Returns:
        processed_data
    """
    trainer_cleaned = trainer[cols]

    return trainer_cleaned
