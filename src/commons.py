import holidays
import pandas as pd

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
