# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from src import (
    FEATURES_PRICE_MODEL_Q1,
    FEATURES_REVENUE_MODEL_Q1,
    PATH_DAILY_REVENUE,
    PATH_LISTINGS,
    REFERENCE_DATE,
)
from src.commons import WEEK_DAY_ORDER, is_holiday
from src.features.build_features import (
    build_daily_features,
    build_date_features,
    build_listings_features,
)
from src.models.preprocessing import one_hot_encode_column


def load_data():
    """Loads the datasets to be used on analysis.

    Returns
    -------
    tuple
        Returns respectively the listings and the daily revenue datasets.
    """
    # Importing Datasets
    df_listings = pd.read_csv(PATH_LISTINGS)
    df_daily_revenue = pd.read_csv(PATH_DAILY_REVENUE)

    # Data Cleaning
    df_listings = clean_listings_dataset(df_listings)
    df_daily_revenue = clean_daily_revenue_dataset(df_daily_revenue)

    # Building Features
    df_listings = build_listings_features(df_listings)
    df_daily_revenue = build_daily_features(df_daily_revenue)

    return df_listings, df_daily_revenue


def clean_listings_dataset(df_listings: pd.DataFrame):
    """Data cleaning and casting process for listings dataset.

    Parameters
    ----------
    df_listings : pd.DataFrame
        Pandas dataframe with information about listings.

    Returns
    -------
    pd.DataFrame
        Returns the listing dataframe with the casting and missing
        treatment made.
    """

    df_listings["Comissão"] = (
        df_listings["Comissão"].str.replace(",", ".").astype(float)
    )

    df_listings["Cama Casal"] = (
        df_listings["Cama Casal"]
        .replace("Quantidade de Camas Casal", np.nan)
        .str.replace(",", ".")
        .astype(float)
        .clip(-128, 127)
        .astype("Int8")
    )

    df_listings["Cama Solteiro"] = (
        df_listings["Cama Solteiro"]
        .replace("Quantidade de Camas Solteiro", np.nan)
        .str.replace(",", ".")
        .astype(float)
        .clip(-128, 127)
        .astype("Int8")
    )

    df_listings["Cama Queen"] = (
        df_listings["Cama Queen"]
        .replace("Quantidade de Camas Queen", np.nan)
        .str.replace(",", ".")
        .astype(float)
        .clip(-128, 127)
        .astype("Int8")
    )

    df_listings["Cama King"] = (
        df_listings["Cama King"]
        .replace("Quantidade de Camas King", np.nan)
        .str.replace(",", ".")
        .astype(float)
        .clip(-128, 127)
        .astype("Int8")
    )

    df_listings["Sofá Cama Solteiro"] = (
        df_listings["Sofá Cama Solteiro"]
        .replace("Quantidade de Sofás Cama Solteiro", np.nan)
        .str.replace(",", ".")
        .astype(float)
        .clip(-128, 127)
        .astype("Int8")
    )

    df_listings["Travesseiros"] = (
        df_listings["Travesseiros"]
        .str.replace(",", ".")
        .astype(float)
        .clip(-128, 127)
        .astype("Int8")
    )

    df_listings["Banheiros"] = (
        df_listings["Banheiros"]
        .replace("Banheiros", np.nan)
        .str.replace(",", ".")
        .astype(float)
        .round(0)
        .clip(-128, 127)
        .astype("Int8")
    )

    df_listings["Taxa de Limpeza"] = (
        df_listings["Taxa de Limpeza"].str.replace(",", ".").astype(float)
    )

    df_listings["Capacidade"] = (
        df_listings["Capacidade"]
        .replace("Capacidade", np.nan)
        .str.replace(",", ".")
        .astype(float)
        .clip(-128, 127)
        .astype("Int8")
    )

    df_listings["Data Inicial do contrato"] = pd.to_datetime(
        df_listings["Data Inicial do contrato"], dayfirst=True
    )

    return df_listings


def clean_daily_revenue_dataset(df_daily_revenue: pd.DataFrame):
    """Data cleaning and casting process for daily revenue dataset.

    Parameters
    ----------
    df_daily_revenue : pd.DataFrame
        Pandas dataframe with information about daily revenue.

    Returns
    -------
    pd.DataFrame
        Returns the daily revenue dataframe with the casting and missing
        treatment made.
    """

    df_daily_revenue["date"] = pd.to_datetime(df_daily_revenue["date"])

    df_daily_revenue["occupancy"] = (
        df_daily_revenue["occupancy"].clip(0, 1).astype("Int8")
    )

    df_daily_revenue["blocked"] = df_daily_revenue["blocked"].clip(0, 1).astype("Int8")

    df_daily_revenue["creation_date"] = pd.to_datetime(
        df_daily_revenue["creation_date"]
    )

    df_daily_revenue = df_daily_revenue.loc[
        df_daily_revenue["date"] <= pd.to_datetime(REFERENCE_DATE)
    ]

    return df_daily_revenue


def make_predict_dataset_price_q1():
    """Creates the dataset to apply the price model
    to answer question 1.

    Returns
    -------
    pd.DataFrame
        A pandas series with the dataframe ready to be inputed
        on the preprocessing pipeline and model predict method.
    """

    data_pred = pd.DataFrame({})

    # Selecionando os dias do mes de março nos tres anos
    data_pred["date"] = pd.date_range(
        start=pd.to_datetime("2022-03-01"), end=pd.to_datetime("2022-03-31")
    ).to_list()
    data_pred["Categoria"] = 5

    data_pred["Quartos"] = 2

    data_pred["Localização_JUR"] = 1

    data_pred = build_date_features(data_pred, "date")

    for col in FEATURES_PRICE_MODEL_Q1:
        if col not in data_pred.columns:
            data_pred[col] = 0

    data_pred = data_pred[FEATURES_PRICE_MODEL_Q1]

    return data_pred


def make_predict_dataset_revenue_q1():
    """Creates the dataset to apply the revenue model
    to answer question 1.

    Returns
    -------
    pd.DataFrame
        A pandas series with the dataframe ready to be inputed
        on the preprocessing pipeline and model predict method.
    """

    data_pred = pd.DataFrame({})

    data_pred["date"] = pd.date_range(
        start=pd.to_datetime("2022-03-01"), end=pd.to_datetime("2022-03-31")
    ).to_list()

    data_pred["Categoria"] = 5

    data_pred["Quartos"] = 2

    data_pred["Localização_JUR"] = 1

    data_pred = build_date_features(data_pred, "date")

    for col in FEATURES_REVENUE_MODEL_Q1:
        if col not in data_pred.columns:
            data_pred[col] = 0

    data_pred = data_pred[FEATURES_REVENUE_MODEL_Q1]

    return data_pred
