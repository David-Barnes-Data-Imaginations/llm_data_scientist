from smolagents import tool
from dataframe_memory import dataframe_store
import pandas as pd


def create_dataframe(data: list, name: str = "df"):
    """
    Creates and stores a DataFrame from a list of dicts.
    """
    df = pd.DataFrame(data)
    dataframe_store[name] = df
    return f"Created DataFrame '{name}' with shape {df.shape}"

