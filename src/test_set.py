import pandas as pd


def get_test_set(df: pd.DataFrame, date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get the test set from the dataframe.

    Args:
        df (pd.DataFrame): The dataframe to get the test set from.
        date (str): The date to filter the test set by.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: The test set and train set.
    """
    df_copy = df.copy()
    df_copy["issue_d"] = pd.to_datetime(df_copy["issue_d"], format="%b-%Y")

    df_copy = df_copy.sort_values("issue_d").reset_index(drop=True)

    test = df_copy[df_copy["issue_d"] >= date].copy()
    train = df_copy[df_copy["issue_d"] < date].copy()

    return (train, test)
