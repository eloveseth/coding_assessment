import csv
import pandas as pd


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
