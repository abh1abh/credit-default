import pandas as pd

RESOLVED_STATUSES = ["Charged Off", "Default", "Fully Paid"]


def get_resolve_loan_status(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resolve the loan_status column in the dataframe to only include "Charged Off", "Default", and "Fully Paid".
    All other statuses will be set to NaN.

    Args:
        df (pd.DataFrame): The input dataframe.

    Returns:
        pd.DataFrame: The dataframe with the resolved loan_status column.
    """
    resolved = df[df["loan_status"].isin(RESOLVED_STATUSES)].copy()
    return resolved


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a target column to the dataframe based on the loan_status column.
    The target column will be 1 if the loan_status is "Charged Off" or "Default",
    and 0 otherwise.

    Args:
        df (pd.DataFrame): The input dataframe.

    Returns:
        pd.DataFrame: The dataframe with the added target column and removed loan_status column.
    """
    resolved = df.copy()
    resolved["target"] = (
        resolved["loan_status"].isin(["Charged Off", "Default"]).astype(int)
    )
    resolved = resolved.drop(columns=["loan_status"])
    return resolved
