# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from src.commons import transform_dataframe


def one_hot_encode_column(dataframe: pd.DataFrame, column_name: str):
    """Applies the one-hot enconding method to a specified categorical
    column of a given dataframe.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataframe with the objective column.
    column_name : str
        Column name to be encoded

    Returns
    -------
    pd.DataFrame
        Return the input dataframe with the on-hot encoded columns added.
    """
    return pd.concat(
        [
            dataframe,
            pd.get_dummies(dataframe[column_name], prefix=column_name, drop_first=True),
        ],
        axis=1,
    ).drop(columns=column_name)


def preprocess_transform(X: pd.DataFrame, preprocess: list):
    """Applies a sequence of transformations in a given dataframe.

    Parameters
    ----------
    X : pd.DataFrame
        Dataframe to be transformed.
    preprocess : list
        List of sklearn type transformations to be applied on dataframe

    Returns
    -------
    pd.DataFrame
        Returns the transformed dataframe
    """

    for p in preprocess:
        X = transform_dataframe(p, X)

    return X


def fit_preprocess_price_model_q1(X_train: pd.DataFrame):
    """Train the preprocess pipeline stages of the model for the question 1
    with a given dataframe

    Parameters
    ----------
    X_train : pd.DataFrame
        Dataframe to train the preprocess pipeline

    Returns
    -------
    tuple
        Returns a tuple with all stages of the pipeline trained.
    """
    imputer = SimpleImputer(strategy="mean")
    imputer.fit(X_train)
    X_train = imputer.transform(X_train)

    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)

    return imputer, scaler


def fit_preprocess_revenue_model_q2(X_train: pd.DataFrame):
    """Train the preprocess pipeline stages of the model for the question 2
    with a given dataframe

    Parameters
    ----------
    X_train : pd.DataFrame
        Dataframe to train the preprocess pipeline

    Returns
    -------
    tuple
        Returns a tuple with all stages of the pipeline trained.
    """
    imputer = SimpleImputer(strategy="mean")
    imputer.fit(X_train)
    X_train = imputer.transform(X_train)

    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)

    return imputer, scaler


def fit_preprocess_reservations_model_q3(X_train: pd.DataFrame):
    """Train the preprocess pipeline stages of the model for the question 3
    with a given dataframe

    Parameters
    ----------
    X_train : pd.DataFrame
        Dataframe to train the preprocess pipeline

    Returns
    -------
    tuple
        Returns a tuple with all stages of the pipeline trained.
    """
    imputer = SimpleImputer(strategy="mean")
    imputer.fit(X_train)
    X_train = imputer.transform(X_train)

    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)

    return imputer, scaler
