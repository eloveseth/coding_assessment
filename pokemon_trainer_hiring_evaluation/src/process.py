import pandas as pd

from .utils import extract_pay_type, extract_pay_amt, read_csv_to_df, write_df_to_csv

PAY_RATE = {
    'RecentTrainingExperience1PayRateEnd': ['pay_type_1', 'pay_amt_1'],
    'RecentTrainingExperience2PayRateEnd': ['pay_type_2', 'pay_amt_2'],
    'RecentTrainingExperience3PayRateEnd': ['pay_type_3', 'pay_amt_3'],
    'RecentTrainingExperience4PayRateEnd': ['pay_type_4', 'pay_amt_4']
}

FEATURES = [
    'hired',
    'WorkingForJobAppliedFor',
    'GymCertified',
]


def process_data() -> pd.DataFrame:
    """Function to process raw trainer data.

    Args:

    Returns:
        processed_data
    """
    # load data
    trainer = read_csv_to_df("./data/raw/ds_pokemon_trainer_application_data.csv")

    # feature selection and feature engineering
    trainer_processed = trainer[FEATURES]
    trainer_processed = trainer_processed.dropna(subset=['hired'])

    trainer_pay_rates = clean_trainer_pay_rates(trainer)
    trainer_processed = trainer_processed.join(trainer_pay_rates)

    trainer_processed = trainer_processed.join(pd.get_dummies(trainer['GymBadge4Pokemon'], prefix='gym_badge_'))
    trainer_processed = trainer_processed.join(pd.get_dummies(trainer['CurrentlyTrainingPokemon'], prefix='current_'))
    trainer_processed = trainer_processed.join(pd.get_dummies(trainer['LastPerformaceRating'], prefix='certified_'))

    write_df_to_csv(trainer_processed, './data/processed/trainer_processed.csv')
    import pdb; pdb.set_trace()

    return trainer_processed


def clean_trainer_pay_rates(trainer: pd.DataFrame) -> pd.DataFrame:
    """Helper function for process_data. Extracts pay rates and pay types from trainer set. Converts all pay to
    similar scale for inclusion in model.

    Args:
        trainer: raw unprocessed dataset

    Returns:
        processed_data
    """
    trainer_rates = trainer[PAY_RATE.keys()]
    pd.set_option('mode.chained_assignment', None)

    for key, _ in PAY_RATE.items():
        trainer_rates[key] = trainer_rates[key].apply(str)

    for key, value in PAY_RATE.items():
        trainer_rates[value[0]] = trainer_rates[key].apply(lambda row: extract_pay_type(pay_rate=row))

    for key, value in PAY_RATE.items():
        trainer_rates[value[1]] = trainer_rates[key].apply(lambda row: extract_pay_amt(pay_rate=row))
        trainer_rates[value[1]] = pd.to_numeric(trainer_rates[value[1]], errors='coerce')

    trainer_rates = trainer_rates.drop(PAY_RATE.keys(), axis=1)

    trainer_rates_converted = convert_trainer_pay_rates(trainer_rates)

    return trainer_rates_converted


def convert_trainer_pay_rates(df: pd.DataFrame) -> pd.DataFrame():
    for _, value in PAY_RATE.items():
        if 'PER_HOUR' in df[value[0]]:
            df[value[1]] = df[value[1]] * 52 * 40
        elif 'PER_WEEK' in df[value[0]]:
            df[value[1]] = df[value[1]] * 52
        elif 'PER_MONTH' in df[value[0]]:
            df[value[1]] = df[value[1]] * 12
        else:
            df[value[1]]

    converted_rates = df.drop(df.iloc[:, 0:4], axis=1)

    return converted_rates


if __name__ == '__main__':
    process_data()