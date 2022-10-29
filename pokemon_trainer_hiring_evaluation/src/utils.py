import csv
import pandas as pd
from typing import List
import re


def read_csv_to_df(fpath: str) -> pd.DataFrame:
    """
    Reads csv to dataframe.

    Args:
        fpath: filepath for csv to read

    Returns:
        df: dataframe of csv
    """
    df = pd.read_csv(fpath)

    return df


def write_df_to_csv(df: pd.DataFrame, fpath: str):
    """
    Reads csv to dataframe.

    Args:
        df: dataframe to save
        fpath: filepath to save csv

    Returns:
    """
    df.to_csv(fpath)


def extract_pay_type(pay_rate: str) -> str:
    pay_type = re.sub('[0-9\.\s]', '', pay_rate)

    return pay_type


def extract_pay_amt(pay_rate: str) -> str:
    pay_amt = re.sub('[\s_a-zA-Z]', '', pay_rate)

    return pay_amt


def convert_to_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = pd.to_numeric(df[cols], errors='coerce')

    return df
