# -*- coding: utf-8 -*-

from datetime import timedelta
import pandas as pd
import numpy as np
from src import FEATURES_PRICE_MODEL_Q1, FEATURES_REVENUE_MODEL_Q1

from src.models.preprocessing import one_hot_encode_column
from src.commons import add_day_of_week, decompose_date_ymd, is_holiday
from statsmodels.tsa.seasonal import seasonal_decompose


def build_date_features(dataframe: pd.DataFrame, date_column: str):
    """Decomposes date in year, month and day. Adds a one hot
    encoding structure for 'day of week'. Adds a flag for holiday.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Pandas dataframe containing a date column.
    date_column : str
        A column containing date in datetime format.

    Returns
    -------
    pd.DataFrame
        Returns the dataframe containing a column indicating whether a date
        is a holiday, columns for decomposed date and one hot encoding
        structure for 'day of week'.
    """

    dataframe = decompose_date_ymd(dataframe, date_column)
    dataframe = add_day_of_week(dataframe, date_column)
    dataframe["holiday"] = dataframe[date_column].apply(is_holiday)
    dataframe = one_hot_encode_column(dataframe, "day_of_week")
    dataframe = dataframe.drop(columns=date_column)

    return dataframe


def build_daily_features(df_daily_revenue: pd.DataFrame):
    """Constructs the features related to daily revenue
    dataset.

    Parameters
    ----------
    df_daily_revenue : pd.DataFrame
        Pandas dataframe with information aboutt daily revenue.

    Returns
    -------
    pd.DataFrame
        Returns the input dataframe with the columns
        'reservation_advance_days' and 'reservation_advance_days' added.
    """
    df_daily_revenue["reservation_advance_days"] = (
        df_daily_revenue["date"] - df_daily_revenue["creation_date"]
    ).dt.days

    df_daily_revenue.loc[
        df_daily_revenue["reservation_advance_days"] < 0, "reservation_advance_days"
    ] = np.nan

    return df_daily_revenue


def build_listings_features(df_listings: pd.DataFrame):
    """Constructs the features related to listings properties.

    Parameters
    ----------
    df_listings : pd.DataFrame
        Pandas dataframe with information about listings.

    Returns
    -------
    pd.DataFrame
        Returns the input dataframe with the columns
        'Quartos' added and the feature 'Categoria'
        numerically encoded.
    """

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


def build_features_price_model_q1(
    df_listings: pd.DataFrame, df_daily_revenue: pd.DataFrame
):
    """Builds the features to be used on the price modelling for
    answer question 1.

    Parameters
    ----------
    df_listings : pd.DataFrame
        Pandas dataframe with information about listings.
    df_daily_revenue : pd.DataFrame
        Pandas dataframe with information about daily revenue.

    Returns
    -------
    pd.DataFrame
         Returns the input pandas dataframe with the new features added.
    """

    data = pd.merge(
        df_daily_revenue,
        df_listings[["C??digo", "Comiss??o", "Categoria", "Quartos", "Localiza????o"]],
        left_on="listing",
        right_on="C??digo",
        how="left",
    )

    data_revenue = data.drop(
        columns=[
            "listing",
            "revenue",
            "occupancy",
            "blocked",
            "creation_date",
            "C??digo",
            "Comiss??o",
            "reservation_advance_days",
        ]
    )

    data_revenue = build_date_features(data_revenue, "date")

    data_revenue = one_hot_encode_column(data_revenue, "Localiza????o")

    data_revenue = data_revenue.loc[data_revenue["last_offered_price"] > 0]

    X = data_revenue.drop(columns="last_offered_price")[FEATURES_PRICE_MODEL_Q1].astype(
        float
    )

    y = data_revenue["last_offered_price"]

    return X, y


def build_features_revenue_model_q1(
    df_listings: pd.DataFrame, df_daily_revenue: pd.DataFrame
):
    """Builds the features to be used on the revenue modelling for
    answer question 1.

    Parameters
    ----------
    df_listings : pd.DataFrame
        Pandas dataframe with information about listings.
    df_daily_revenue : pd.DataFrame
        Pandas dataframe with information about daily revenue.

    Returns
    -------
    pd.DataFrame
         Returns the input pandas dataframe with the new features added.
    """

    data = pd.merge(
        df_daily_revenue,
        df_listings[["C??digo", "Comiss??o", "Categoria", "Quartos", "Localiza????o"]],
        left_on="listing",
        right_on="C??digo",
        how="left",
    )

    data["company_revenue"] = data["Comiss??o"] * data["revenue"]

    data_revenue = data.drop(
        columns=[
            "listing",
            "last_offered_price",
            "occupancy",
            "blocked",
            "creation_date",
            "C??digo",
            "Comiss??o",
            "reservation_advance_days",
            "revenue",
        ]
    )

    data_revenue = (
        data_revenue.groupby(["date", "Categoria", "Quartos", "Localiza????o"])[
            ["company_revenue"]
        ]
        .sum()
        .reset_index()
    )

    data_revenue = build_date_features(data_revenue, "date")

    data_revenue = one_hot_encode_column(data_revenue, "Localiza????o")

    X = data_revenue.drop(columns="company_revenue")[FEATURES_REVENUE_MODEL_Q1].astype(
        float
    )

    y = data_revenue["company_revenue"]

    return X, y


def build_features_revenue_model_q2(
    df_listings: pd.DataFrame, df_daily_revenue: pd.DataFrame
):
    """Builds the features to be used on the revenue modelling for
    answer question 2.

    Parameters
    ----------
    df_listings : pd.DataFrame
        Pandas dataframe with information about listings.
    df_daily_revenue : pd.DataFrame
        Pandas dataframe with information about daily revenue.

    Returns
    -------
    pd.DataFrame
         Returns the input pandas dataframe with the new features added.
    """
    data = pd.merge(
        df_daily_revenue,
        df_listings[["C??digo", "Comiss??o"]],
        left_on="listing",
        right_on="C??digo",
        how="left",
    )

    data["company_revenue"] = data["Comiss??o"] * data["revenue"]

    data_revenue = (
        data.groupby("date")
        .agg(company_revenue=("company_revenue", "sum"))
        .reset_index()
    )

    data_revenue = build_date_features(data_revenue, "date")

    data = data_revenue.loc[data_revenue["company_revenue"].notna()]

    X = data.drop(columns="company_revenue").astype(float)

    y = data["company_revenue"]

    return X, y


def build_features_reservations_model_q3(df_daily_revenue: pd.DataFrame):
    """Builds the features to be used on the reservations modelling for
    answer question 2.

    Parameters
    ----------
    df_daily_revenue : pd.DataFrame
        Pandas dataframe with information about daily revenue.

    Returns
    -------
    pd.DataFrame
         Returns the input pandas dataframe with the new features added.
    """

    df_q3 = df_daily_revenue[
        (df_daily_revenue["occupancy"] == 1) & (df_daily_revenue["blocked"] == 0)
    ]

    data_q3 = df_q3.groupby(["creation_date"]).count().iloc[:, 0:1].reset_index()
    data_q3.columns = ["creation_date", "qt_reservations"]

    data_q3 = build_date_features(data_q3, "creation_date")

    try:
        tsmodel = seasonal_decompose(
            data_q3["qt_reservations"],
            model="additive",
            extrapolate_trend="freq",
            freq=365,
        )
    except Exception as err:
        tsmodel = seasonal_decompose(
            data_q3["qt_reservations"],
            model="additive",
            extrapolate_trend="freq",
            period=365,
        )

    X = data_q3.drop(columns="qt_reservations").astype(float)

    y = tsmodel.trend + tsmodel.seasonal

    return X, y


def return_date_of_quantile_sold_q4(df_daily_revenue: pd.DataFrame, percent: float):
    """Returns the date in which a specified percent of the bookings
    is made for all rent rooms.

    Parameters
    ----------
    df_daily_revenue : pd.DataFrame
        Pandas dataframe with information about daily revenue.
    percent : float
        A real value between 0 and 1 related to the percent of bookings
        to be analysed.

    Returns
    -------
    pd.Series
        A pandas series with the data of n-th percentile (given by percent)
        of the distribution of booking date distributions for each rented room.
    """
    df_q4 = df_daily_revenue[
        (df_daily_revenue["date"].dt.month == 12)
        & (df_daily_revenue["date"].dt.day == 31)
        & (df_daily_revenue["occupancy"] == 1)
        & (df_daily_revenue["blocked"] == 0)
    ]

    day = df_q4["reservation_advance_days"].quantile(1 - percent)

    return pd.to_datetime("2022-12-31") - timedelta(day)


def build_features_covid_impact_model(
    df_listings: pd.DataFrame, df_daily_revenue: pd.DataFrame
):
    """Builds the features to be used on the revenue modelling to
    evaluate the impact of covid-19 on company revenue.

    Parameters
    ----------
    df_listings : pd.DataFrame
        Pandas dataframe with information about listings.
    df_daily_revenue : pd.DataFrame
        Pandas dataframe with information about daily revenue.

    Returns
    -------
    pd.DataFrame
         Returns the input pandas dataframe with the new features added.
    """
    data = pd.merge(
        df_daily_revenue,
        df_listings[["C??digo", "Comiss??o"]],
        left_on="listing",
        right_on="C??digo",
        how="left",
    )

    data["company_revenue"] = data["Comiss??o"] * data["revenue"]

    data = (
        data.groupby("date")
        .agg(company_revenue=("company_revenue", "sum"))
        .reset_index()
    )

    df = data.copy()
    df = df[
        (df["date"] <= pd.to_datetime("2020-02-29"))
        | (df["date"] > pd.to_datetime("2021-08-31"))
    ]

    df = build_date_features(df, "date")

    df["company_revenue"] = df["company_revenue"].rolling(window=7, center=True).mean()

    df = df[df["company_revenue"].notna()]

    X = df.drop(columns="company_revenue").astype(float)

    y = df["company_revenue"]

    return X, y
