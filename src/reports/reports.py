# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import holidays
from src.models.preprocessing import one_hot_encode_column, preprocess_revenue_model
from src.commons import WEEK_DAY_ORDER, load_pickle


def print_reservation_advance_quantiles(df_daily_revenue):

    print("\n{}".format(89 * "*"))

    perc = 0.99917
    df_daily_revenue["reservation_advance_days"].quantile(perc)
    print(
        "Amount of customers with reservation advance of 1 year or more: {:.2f} %".format(
            (1 - perc) * 100
        )
    )

    perc = 0.984
    df_daily_revenue["reservation_advance_days"].quantile(perc)
    print(
        "Amount of customers with reservation advance of 6 months or more: {:.2f} %".format(
            (1 - perc) * 100
        )
    )

    perc = 0.73
    df_daily_revenue["reservation_advance_days"].quantile(perc)
    print(
        "Amount of customers with reservation advance of 31 days or more: {:.2f} %".format(
            (1 - perc) * 100
        )
    )

    perc = 0.35
    df_daily_revenue["reservation_advance_days"].quantile(perc)
    print(
        "Amount of customers with reservation advance of 7 days or more: {:.2f} %".format(
            (1 - perc) * 100
        )
    )


def answer_first_question(df_listings, df_daily_revenue):
    data = pd.merge(
        df_daily_revenue,
        df_listings,
        left_on="listing",
        right_on="Código",
        how="left",
    )

    temp = data[
        (data["Localização"] == "JUR")
        & (data["Categoria"] == "MASTER1Q")
        & (data["date"].dt.month == 3)
    ]

    avg_last_offered_price_M1Q = temp[temp["last_offered_price"] > 0][
        "last_offered_price"
    ].mean()

    temp = data[
        (data["Localização"] == "JUR")
        & (data["Categoria"] == "MASTER3Q")
        & (data["date"].dt.month == 3)
    ]
    avg_last_offered_price_M3Q = temp[temp["last_offered_price"] > 0][
        "last_offered_price"
    ].mean()

    avg_last_offered_price_M2Q = np.mean(
        [avg_last_offered_price_M1Q, avg_last_offered_price_M3Q]
    )

    temp = data[
        (data["Localização"] == "JUR")
        & (data["Categoria"] == "MASTER1Q")
        & (data["date"].dt.month == 3)
    ]

    avg_revenue_M1Q = temp[temp["revenue"] > 0]["revenue"].sum()

    temp = data[
        (data["Localização"] == "JUR")
        & (data["Categoria"] == "MASTER3Q")
        & (data["date"].dt.month == 3)
    ]
    avg_revenue_M3Q = temp[temp["revenue"] > 0]["revenue"].sum()

    avg_revenue_M2Q = np.mean([avg_revenue_M1Q, avg_revenue_M3Q])

    print(
        "\n{}\nWhat is the expected price and revenue for a listing tagged as JUR MASTER2Q in March?".format(
            89 * "*"
        )
    )
    print("Expected price R$: {:.2f}".format(avg_last_offered_price_M2Q))
    print("Expected revenue R$: {:.2f}".format(avg_revenue_M2Q))


def answer_second_question():
    data_pred = pd.DataFrame()
    data_pred["date"] = pd.date_range(
        start=pd.to_datetime("2022-03-16"), end=pd.to_datetime("2022-12-31")
    )

    data_pred["year"] = data_pred["date"].dt.year
    data_pred["month"] = data_pred["date"].dt.month
    data_pred["day"] = data_pred["date"].dt.day

    data_pred["day_of_week"] = data_pred["date"].dt.dayofweek.replace(WEEK_DAY_ORDER)

    brazilian_holidays = holidays.Brazil()

    def is_holiday(date):
        return int(date in brazilian_holidays)

    data_pred["holiday"] = data_pred["date"].apply(is_holiday)
    data_pred = one_hot_encode_column(data_pred, "day_of_week")

    data_pred = data_pred.drop(columns="date")

    preprocessor = load_pickle("models/preprocessor_revenue_model.pickle")
    model = load_pickle("models/regressor_revenue_model.pickle")

    X_pred = preprocess_revenue_model(data_pred, preprocessor)

    model.predict(X_pred).sum()
