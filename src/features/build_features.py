# -*- coding: utf-8 -*-

from datetime import timedelta
import pandas as pd
import numpy as np
from src import FEATURES_PRICE_MODEL_Q1, FEATURES_REVENUE_MODEL_Q1, REFERENCE_DATE
from src.models.preprocessing import one_hot_encode_column
from src.commons import WEEK_DAY_ORDER, is_holiday
from statsmodels.tsa.seasonal import seasonal_decompose


def build_daily_features(df_daily_revenue):
    df_daily_revenue["reservation_advance_days"] = (
        df_daily_revenue["date"] - df_daily_revenue["creation_date"]
    ).dt.days

    df_daily_revenue.loc[
        df_daily_revenue["reservation_advance_days"] < 0, "reservation_advance_days"
    ] = np.nan

    return df_daily_revenue


def build_listings_features(df_listings):
    """ """

    de_para_categoria = {
        "SIM": 1,
        "JR": 2,
        "SUP": 3,
        "TOP": 4,
        "MASTER": 5,
    }

    df_listings["Quartos"] = df_listings["Categoria"].str[-2]

    df_listings["Quartos"] = (
        df_listings["Quartos"]
        .where(~df_listings["Quartos"].str.isalpha(), np.nan)
        .astype(float)
        .astype("Int8")
    )
    df_listings["Categoria"] = (
        df_listings["Categoria"].str.replace("HOU", "").str.replace("TOPM", "TOP")
    )

    for i in range(1, 10):
        df_listings["Categoria"] = df_listings["Categoria"].str.replace(
            str(i) + "Q", ""
        )

    df_listings["Categoria"] = df_listings["Categoria"].replace(de_para_categoria)

    return df_listings


def build_features_price_model_q1(df_listings, df_daily_revenue):
    """ """

    data = pd.merge(
        df_daily_revenue,
        df_listings[["Código", "Comissão", "Categoria", "Quartos", "Localização"]],
        left_on="listing",
        right_on="Código",
        how="left",
    )

    data_revenue = data.drop(
        columns=[
            "listing",
            "revenue",
            "occupancy",
            "blocked",
            "creation_date",
            "Código",
            "Comissão",
            "reservation_advance_days",
        ]
    )

    data_revenue["year"] = data_revenue["date"].dt.year
    data_revenue["month"] = data_revenue["date"].dt.month
    data_revenue["day"] = data_revenue["date"].dt.day

    data_revenue["day_of_week"] = data_revenue["date"].dt.dayofweek.replace(
        WEEK_DAY_ORDER
    )

    data_revenue["holiday"] = data_revenue["date"].apply(is_holiday)
    data_revenue = one_hot_encode_column(data_revenue, "day_of_week")
    data_revenue = one_hot_encode_column(data_revenue, "Localização")

    data_revenue = data_revenue.drop(columns="date")

    data_revenue = data_revenue.loc[data_revenue["last_offered_price"] > 0]

    X = data_revenue.drop(columns="last_offered_price")[FEATURES_PRICE_MODEL_Q1].astype(
        float
    )

    y = data_revenue["last_offered_price"]

    return X, y


def build_features_revenue_model_q1(df_listings, df_daily_revenue):
    """ """

    data = pd.merge(
        df_daily_revenue,
        df_listings[["Código", "Comissão", "Categoria", "Quartos", "Localização"]],
        left_on="listing",
        right_on="Código",
        how="left",
    )

    data["company_revenue"] = data["Comissão"] * data["revenue"]

    data_revenue = data.drop(
        columns=[
            "listing",
            "last_offered_price",
            "occupancy",
            "blocked",
            "creation_date",
            "Código",
            "Comissão",
            "reservation_advance_days",
            "revenue",
        ]
    )

    data_revenue = (
        data_revenue.groupby(["date", "Categoria", "Quartos", "Localização"])[
            ["company_revenue"]
        ]
        .sum()
        .reset_index()
    )

    data_revenue["year"] = data_revenue["date"].dt.year

    data_revenue["month"] = data_revenue["date"].dt.month

    data_revenue["day"] = data_revenue["date"].dt.day

    data_revenue["day_of_week"] = data_revenue["date"].dt.dayofweek.replace(
        WEEK_DAY_ORDER
    )

    data_revenue["holiday"] = data_revenue["date"].apply(is_holiday)

    data_revenue = one_hot_encode_column(data_revenue, "day_of_week")

    data_revenue = one_hot_encode_column(data_revenue, "Localização")

    data_revenue = data_revenue.drop(columns="date")

    X = data_revenue.drop(columns="company_revenue")[FEATURES_REVENUE_MODEL_Q1].astype(
        float
    )

    y = data_revenue["company_revenue"]

    return X, y


def build_features_revenue_model_q2(df_listings, df_daily_revenue):
    data = pd.merge(
        df_daily_revenue,
        df_listings[["Código", "Comissão"]],
        left_on="listing",
        right_on="Código",
        how="left",
    )

    data["company_revenue"] = data["Comissão"] * data["revenue"]

    data_revenue = (
        data.groupby("date")
        .agg(company_revenue=("company_revenue", "sum"))
        .reset_index()
    )

    data_revenue["year"] = data_revenue["date"].dt.year
    data_revenue["month"] = data_revenue["date"].dt.month
    data_revenue["day"] = data_revenue["date"].dt.day

    data_revenue["day_of_week"] = data_revenue["date"].dt.dayofweek.replace(
        WEEK_DAY_ORDER
    )

    data_revenue["holiday"] = data_revenue["date"].apply(is_holiday)

    data_revenue = one_hot_encode_column(data_revenue, "day_of_week")

    data_revenue = data_revenue.drop(columns="date")

    data = data_revenue.loc[data_revenue["company_revenue"].notna()]

    X = data.drop(columns="company_revenue").astype(float)

    y = data["company_revenue"]

    return X, y


def build_features_reservations_model_q3(df_daily_revenue):

    df_q3 = df_daily_revenue[
        (df_daily_revenue["occupancy"] == 1) & (df_daily_revenue["blocked"] == 0)
    ]

    data_q3 = df_q3.groupby(["creation_date"]).count().iloc[:, 0:1].reset_index()
    data_q3.columns = ["creation_date", "qt_reservations"]

    data_q3["year"] = data_q3["creation_date"].dt.year
    data_q3["month"] = data_q3["creation_date"].dt.month
    data_q3["day"] = data_q3["creation_date"].dt.day

    data_q3["day_of_week"] = data_q3["creation_date"].dt.dayofweek.replace(
        WEEK_DAY_ORDER
    )

    data_q3["holiday"] = data_q3["creation_date"].apply(is_holiday)

    data_q3 = one_hot_encode_column(data_q3, "day_of_week")

    data_q3 = data_q3.drop(columns="creation_date")

    tsmodel = seasonal_decompose(
        data_q3["qt_reservations"],
        model="additive",
        extrapolate_trend="freq",
        freq=365,
    )

    X = data_q3.drop(columns="qt_reservations").astype(float)

    y = tsmodel.trend + tsmodel.seasonal

    return X, y


def return_date_of_quantile_sold_q4(df_daily_revenue, percent):
    df_q4 = df_daily_revenue[
        (df_daily_revenue["date"].dt.month == 12)
        & (df_daily_revenue["date"].dt.day == 31)
        & (df_daily_revenue["occupancy"] == 1)
        & (df_daily_revenue["blocked"] == 0)
    ]
    day = df_q4["reservation_advance_days"].quantile(1 - percent)
    return pd.to_datetime("2022-12-31") - timedelta(day)
