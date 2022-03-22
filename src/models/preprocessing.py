# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

def one_hot_encode_column(dataframe, column_name):
    return pd.concat(
        [
            dataframe,
            pd.get_dummies(dataframe[column_name], prefix=column_name, drop_first=True),
        ],
        axis=1,
    ).drop(columns=column_name)