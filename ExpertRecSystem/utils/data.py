import json
import pandas as pd


def read_json(path: str) -> dict:
    """
    Read a JSON file and return its content as a dictionary.

    Args:
        `path` (`str`): The path to the JSON file.

    Returns:
        `dict`: A dictionary containing the data from the JSON file.
    """
    with open(path, "r") as f:
        return json.load(f)


def read_expert_data(data_path: str) -> pd.DataFrame:
    """
    Read expert data from a CSV file and return it as a pandas DataFrame.

    Args:
        `data_path` (`str`): The path to the CSV file containing expert data.

    Returns:
        `pd.DataFrame`: A pandas DataFrame containing the expert data.
    """
    data = pd.read_csv(data_path, encoding="utf-8")
    return data
