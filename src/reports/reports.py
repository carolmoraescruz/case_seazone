# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import holidays
from src import FEATURES_PRICE_MODEL_Q1
from src.data.make_dataset import (
    make_predict_dataset_price_q1,
    make_predict_dataset_revenue_q1,
)
from src.models.preprocessing import one_hot_encode_column, preprocess_transform
from src.commons import WEEK_DAY_ORDER, is_holiday, load_pickle, transform_dataframe
from src.models.train_model import train_revenue_model_q2, train_price_model_q1


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


def header_q1():
    print(
        "\n{}\n1)What is the expected price and revenue for a listing tagged as JUR MASTER2Q in March?".format(
            89 * "*"
        )
    )


def header_q2():
    print("\n{}\n2)What is Seazone's expected revenue for 2022?".format(89 * "*"))


def header_q3():
    print(
        "\n{}\n2)How many reservations should we expect to sell per day?".format(
            89 * "*"
        )
    )


def answer_first_question():
    """ """

    # Price

    data_pred_price = make_predict_dataset_price_q1()

    price_preprocessor = load_pickle("models/preprocessor_price_model_q1.pickle")

    price_model = load_pickle("models/regressor_price_model_q1.pickle")

    X_pred = preprocess_transform(data_pred_price, price_preprocessor)

    print("Modeled price R$: {:.2f}".format(price_model.predict(X_pred).mean()))

    # Revenue
    data_pred_revenue = make_predict_dataset_revenue_q1()

    revenue_preprocessor = load_pickle("models/preprocessor_revenue_model_q1.pickle")

    revenue_model = load_pickle("models/regressor_revenue_model_q1.pickle")

    X_pred = preprocess_transform(data_pred_revenue, revenue_preprocessor)

    print("Modeled revenue R$: {:.2f}".format(revenue_model.predict(X_pred).sum()))


def answer_second_question():

    data_pred = pd.DataFrame()
    data_pred["date"] = pd.date_range(
        start=pd.to_datetime("2022-01-01"), end=pd.to_datetime("2022-12-31")
    )

    data_pred["year"] = data_pred["date"].dt.year
    data_pred["month"] = data_pred["date"].dt.month
    data_pred["day"] = data_pred["date"].dt.day

    data_pred["day_of_week"] = data_pred["date"].dt.dayofweek.replace(WEEK_DAY_ORDER)

    data_pred["holiday"] = data_pred["date"].apply(is_holiday)
    data_pred = one_hot_encode_column(data_pred, "day_of_week")

    data_pred = data_pred.drop(columns="date")

    preprocessor = load_pickle("models/preprocessor_revenue_model_q2.pickle")
    model = load_pickle("models/regressor_revenue_model_q2.pickle")

    X_pred = preprocess_transform(data_pred, preprocessor)

    valor_2022 = model.predict(X_pred).sum()

    print("Expected revenue for 2022 R$: {:.2f}".format(valor_2022))


def answer_third_question():

    data_pred = pd.DataFrame()
    dates_2022 = pd.date_range(
        start=pd.to_datetime("2019-08-22"), end=pd.to_datetime("2022-12-31")
    ).to_list()

    data_pred["creation_date"] = dates_2022

    data_pred["year"] = data_pred["creation_date"].dt.year
    data_pred["month"] = data_pred["creation_date"].dt.month
    data_pred["day"] = data_pred["creation_date"].dt.day

    data_pred["day_of_week"] = data_pred["creation_date"].dt.dayofweek.replace(
        WEEK_DAY_ORDER
    )

    data_pred["holiday"] = data_pred["creation_date"].apply(is_holiday)
    data_pred = one_hot_encode_column(data_pred, "day_of_week")

    data_pred = data_pred.drop(columns="creation_date")

    preprocessor = load_pickle("models/preprocessor_reservations_model_q3.pickle")
    model = load_pickle("models/regressor_reservations_model_q3.pickle")

    X_pred = preprocess_transform(data_pred, preprocessor)

    reservations = model.predict(X_pred)

    mean_reservations_per_day = reservations.mean().round(0).astype(int)

    print(
        "Expected reservations per day for 2022: {:d}".format(mean_reservations_per_day)
    )
