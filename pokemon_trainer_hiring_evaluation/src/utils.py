import csv
import pandas as pd
from typing import List
import re


def read_csv_to_dataframe(fpath: str) -> pd.Dataframe:
    """
    Reads csv to dataframe.
    Args:
        fpath: filepath for csv to read
    Returns:
        df: dataframe of csv
    """
    with open(fpath, 'r') as csvfile:
        csv_file_reader = csv.reader(csvfile, delimiter=',')
        df = pd.Dataframe([row for row in csv_file_reader])

    return df


def extract_pay_type(pay_rate: str) -> str:
    pay_type = re.sub('[0-9\.\s]', '', pay_rate)

    return pay_type


def extract_pay_amt(pay_rate: str) -> str:
    pay_amt = re.sub('[\s_a-zA-Z]', '', pay_rate)

    return pay_amt


def convert_to_numeric(df: pd.DataFrame, cols: List[str]) -> pd. Dataframe:
    df = pd.to_numeric(df[cols], errors='coerce')
