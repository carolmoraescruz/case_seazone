# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from src.commons import transform_dataframe


def one_hot_encode_column(dataframe, column_name):
    """ """
    return pd.concat(
        [
            dataframe,
            pd.get_dummies(dataframe[column_name], prefix=column_name, drop_first=True),
        ],
        axis=1,
    ).drop(columns=column_name)


def preprocess_transform(X, preprocess):
    """ """

    for p in preprocess:
        X = transform_dataframe(p, X)

    return X


def fit_preprocess_revenue_model_q2(X_train, y_train=None):
    """ """
    imputer = SimpleImputer(strategy="mean")
    imputer.fit(X_train)
    X_train = imputer.transform(X_train)

    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)

    return imputer, scaler


def fit_preprocess_price_model_q1(X_train, y_train=None):
    """ """
    imputer = SimpleImputer(strategy="mean")
    imputer.fit(X_train)
    X_train = imputer.transform(X_train)

    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)

    return imputer, scaler