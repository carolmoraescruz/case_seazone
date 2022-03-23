import holidays
import pandas as pd
import pickle

WEEK_DAY_ORDER = {
    0: "Mon",
    1: "Tue",
    2: "Wed",
    3: "Thu",
    4: "Fri",
    5: "Sat",
    6: "Sun",
}

BRAZILIAN_HOLIDAYS = holidays.Brazil()


def is_holiday(date):
    return int(date in BRAZILIAN_HOLIDAYS)


def transform_dataframe(transformer, dataframe):
    """Applies sklearn transform in pandas dataframe and
    converts the result in pandas dataframe
    """

    return pd.DataFrame(
        transformer.transform(dataframe),
        columns=dataframe.columns,
        index=dataframe.index,
    )


def dump_pickle(variable, path):
    with open(path, "wb") as f:
        pickle.dump(variable, f)


def load_pickle(path):
    with open(path, "rb") as f:
        variable = pickle.load(f)
    return variable
