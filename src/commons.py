import holidays
import pandas as pd
import pickle
from datetime import datetime
from sklearn.base import TransformerMixin
from typing import Any

BRAZILIAN_HOLIDAYS = holidays.Brazil()

WEEK_DAY_ORDER = {
    0: "Mon",
    1: "Tue",
    2: "Wed",
    3: "Thu",
    4: "Fri",
    5: "Sat",
    6: "Sun",
}


def is_holiday(date: datetime):
    """Check if the impute is a brazilian holiday.

    Parameters
    ----------
    date : datetime
        Datetime with the date to be analyzed.

    Returns
    -------
    int
        Returns 1 if the passed date is a holiday and 0 otherwise.
    """
    return int(date in BRAZILIAN_HOLIDAYS)


def transform_dataframe(transformer: TransformerMixin, dataframe: pd.DataFrame):
    """Applies sklearn transform in pandas dataframe and
    converts the result in pandas dataframe

    Parameters
    ----------
    transformer : TransformerMixin
       A pandas-like transform.
    dataframe : pd.DataFrame
        A dataframe to be transformer.

    Returns
    -------
    pd.DataFrame
        Returns the transformed dataframe.
    """

    return pd.DataFrame(
        transformer.transform(dataframe),
        columns=dataframe.columns,
        index=dataframe.index,
    )


def dump_pickle(variable: Any, path: str):
    """Write a pickled representation of obj to the open file object file.

    Parameters
    ----------
    variable : Any
        Object to be seriaized to a pickle file
    path : str
        Complete file path to the dumped file
    """
    with open(path, "wb") as f:
        pickle.dump(variable, f)


def load_pickle(path: str):
    """Read and return an object from the pickle data stored in a file.

    Parameters
    ----------
    path : str
        Complete path to the pickle file.

    Returns
    -------
    Any
        Object in the pickle file.
    """
    with open(path, "rb") as f:
        variable = pickle.load(f)
    return variable


def to_date(datetime: datetime):
    """Converts a datetime variable to a string in the format YYYY-MM-DD

    Parameters
    ----------
    datetime : datetime
        Date to be converted to the desired format.

    Returns
    -------
    string
        Formated date.
    """
    return datetime.strftime("%Y-%m-%d")
