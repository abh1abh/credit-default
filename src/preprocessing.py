import pandas as pd


def get_features(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Preprocess the dataframe by removing leakage columns.

    Args:
        df (pd.DataFrame): The input dataframe.
    Returns:
        pd.DataFrame: The preprocessed dataframe.
    """
    cols_free_text = ["emp_title", "desc", "title", "url", "zip_code"]

    cols_joint = df.columns[df.columns.str.contains("_joint")].tolist()
    cols_sec_app = df.columns[df.columns.str.startswith("sec_app_")].tolist()
    cols_vintage_limited = [
        "open_acc_6m",
        "open_act_il",
        "open_il_12m",
        "open_il_24m",
        "mths_since_rcnt_il",
        "total_bal_il",
        "il_util",
        "open_rv_12m",
        "open_rv_24m",
        "max_bal_bc",
        "all_util",
        "inq_fi",
        "total_cu_tl",
        "inq_last_12m",
    ]
    cols_to_drop = [
        *cols_free_text,
        *cols_joint,
        *cols_sec_app,
        *cols_vintage_limited,
        "policy_code",
        "target",
        "earliest_cr_line",  # This column is dropped because of SMOTE
    ]

    numeric_features = (
        df.drop(columns=cols_to_drop).select_dtypes(include=["number"]).columns.tolist()
    )

    categorical_features = (
        df.drop(columns=cols_to_drop)
        .select_dtypes(include=["str", "category"])
        .columns.tolist()
    )

    return numeric_features, categorical_features
