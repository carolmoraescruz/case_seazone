# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from src.features.build_features import build_daily_features


def main():
    pass


def clean_listings_dataset(df_listings):
    """ """

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


def clean_daily_revenue_dataset(df_daily_revenue):
    """ """

    df_daily_revenue["date"] = pd.to_datetime(df_daily_revenue["date"])

    df_daily_revenue["occupancy"] = (
        df_daily_revenue["occupancy"].clip(0, 1).astype("Int8")
    )

    df_daily_revenue["blocked"] = df_daily_revenue["blocked"].clip(0, 1).astype("Int8")

    df_daily_revenue["creation_date"] = pd.to_datetime(
        df_daily_revenue["creation_date"]
    )

    df_daily_revenue = df_daily_revenue.loc[
        df_daily_revenue["date"] <= pd.to_datetime("2022-03-15")
    ]

    return df_daily_revenue


def load_data():

    # Importing Datasets
    df_listings = pd.read_csv("data/raw/listings-challenge.csv")
    df_daily_revenue = pd.read_csv("data/raw/daily_revenue.csv")

    # Data Cleaning
    df_listings = clean_listings_dataset(df_listings)
    df_daily_revenue = clean_daily_revenue_dataset(df_daily_revenue)
    df_daily_revenue = build_daily_features(df_daily_revenue)

    return df_listings, df_daily_revenue
