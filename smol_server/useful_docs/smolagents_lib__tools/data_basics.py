import pandas as pd
import numpy as np

df = pd.DataFrame()
import pandas.core.dtypes.common


def check_dataframe(df_copy: pd.DataFrame) -> None:
    """
    check_dataframe inspects a pandas DataFrame for any non-numeric, NaN, or infinite values.

    Parameters:
    df_copy (DataFrame): The DataFrame to check.

    Raises:
    ValueError: If the DataFrame contains any non-numeric, NaN, or infinite values.
    """

    # Ensure it contains only numeric data
    if not pandas.core.dtypes.common.is_numeric_dtype(
            df_copy.dtypes):  # Call `.all()` to make sure it checks all columns
        raise ValueError("DataFrame contains non-numeric data. Consider encoding these columns.")

    # Check for NaN values
    if df_copy.isnull().any().any():
        raise ValueError("DataFrame contains NaN values. Consider filling or dropping these columns.")

    # Check for Inf values
    if np.isinf(df_copy.values).any():
        raise ValueError("DataFrame contains Inf values. Consider handling these columns.")


############################################################################################################

def inspect_dataframe(df_copy):
    """
    This function prints essential information about a pandas DataFrame including its head, shape, and columns.
    Reusable for all my python notebooks
    :type df_copy: object
    :param df_copy: pandas.DataFrame
    """
    print(df_copy.head())
    print("\nShape:")
    print(df_copy.shape)
    print("\nColumns:")
    print(df_copy.columns)
    print("\nDescriptive Statistics:")
    # describe() only analyses numeric data by default.
    # To include categorical columns
    # in the summary statistics, an argument can be added to the describe() method.
    print(df_copy.describe(include='all'))

############################################################################################################
