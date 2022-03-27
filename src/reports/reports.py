# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from src.data.make_dataset import (
    make_predict_dataset_price_q1,
    make_predict_dataset_revenue_q1,
)
from src.features.build_features import (
    build_date_features,
    return_date_of_quantile_sold_q4,
)
from src.models.preprocessing import preprocess_transform
from src.commons import (
    get_date_from_ymd,
    load_pickle,
    to_date,
)
from src.visualization.visualize import (
    plot_hist_reservation_advance,
    plot_real_pred_data,
    plot_revenue_loss_due_to_covid,
    plot_revenue_per_date,
    plot_seasonal_decomposed_q2,
    plot_seasonal_decomposed_q3,
)


def print_reservation_advance_quantiles(df_daily_revenue: pd.DataFrame):
    """Prints distinct quantiles for the total booking advance dates distribution.

    Parameters
    ----------
    df_daily_revenue : pd.DataFrame
        Pandas dataframe with information about daily revenue.
    """

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
    """Prints the header message for question 1."""
    print(
        "\n{}\n1) What is the expected price and revenue for a listing tagged as JUR MASTER2Q in March?".format(
            89 * "*"
        )
    )


def header_q2():
    """Prints the header message for question 2."""
    print("\n{}\n2) What is Seazone's expected revenue for 2022?".format(89 * "*"))


def header_q3():
    """Prints the header message for question 3."""
    print(
        "\n{}\n3) How many reservations should we expect to sell per day?".format(
            89 * "*"
        )
    )


def header_q4():
    """Prints the header message for question 4."""
    print(
        "\n{}\n4) At what time of the year should we expect to have sold 10 percent of our new year’s nights? And 50? And 80?".format(
            89 * "*"
        )
    )


def header_covid_impact_on_revenue():
    """Prints the header message for the answer about covid impact on revenue."""
    print(
        "\n{}\nCan we estimate Seazone's revenue loss due to the pandemic? Has the industry recovered?".format(
            89 * "*"
        )
    )


def answer_first_question():
    """Script to obtain the answers to question 1."""

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


def answer_second_question(df_listings: pd.DataFrame, df_daily_revenue: pd.DataFrame):
    """Script to obtain the answers to question 2.

    Parameters
    ----------
    df_listings : pd.DataFrame
        Pandas dataframe with information about listings.
    df_daily_revenue : pd.DataFrame
        Pandas dataframe with information about daily revenue.
    """

    data_pred = pd.DataFrame()
    data_pred["date"] = pd.date_range(
        start=pd.to_datetime("2022-01-01"), end=pd.to_datetime("2022-12-31")
    )

    data_pred = build_date_features(data_pred, "date")

    preprocessor = load_pickle("models/preprocessor_revenue_model_q2.pickle")
    model = load_pickle("models/regressor_revenue_model_q2.pickle")

    X_pred = preprocess_transform(data_pred, preprocessor)

    revenue_2022 = model.predict(X_pred).sum()

    print("Expected revenue for 2022 R$: {:.2f}".format(revenue_2022))

    plot_real_pred_data(df_listings, df_daily_revenue)
    plot_seasonal_decomposed_q2(df_listings, df_daily_revenue)


def answer_third_question(df_daily_revenue: pd.DataFrame):
    """Script to obtain the answers to question 3.

    Parameters
    ----------
    df_daily_revenue : pd.DataFrame
        Pandas dataframe with information about daily revenue.
    """

    data_pred = pd.DataFrame()
    dates_2022 = pd.date_range(
        start=pd.to_datetime("2019-08-22"), end=pd.to_datetime("2022-12-31")
    ).to_list()

    data_pred["creation_date"] = dates_2022

    data_pred = build_date_features(data_pred, "creation_date")

    preprocessor = load_pickle("models/preprocessor_reservations_model_q3.pickle")
    model = load_pickle("models/regressor_reservations_model_q3.pickle")

    X_pred = preprocess_transform(data_pred, preprocessor)

    reservations = model.predict(X_pred)

    mean_reservations_per_day = reservations.mean().round(0).astype(int)

    print(
        "Expected reservations per day for 2022: {:d}".format(mean_reservations_per_day)
    )

    plot_seasonal_decomposed_q3(df_daily_revenue)


def answer_fourth_question(df_daily_revenue):
    """Script to obtain the answers to question 4."""

    for percent in [0.1, 0.5, 0.8]:
        date_of_new_year_reservations = return_date_of_quantile_sold_q4(
            df_daily_revenue, percent
        )
        print(
            "{:d} percent of new year's nights should be sold by: ".format(
                int(np.round(percent * 100))
            )
            + to_date(date_of_new_year_reservations)
        )


def answer_covid_impact_on_revenue(df_listings, df_daily_revenue):
    """Script to obtain the answers to covid impact on revenue.

    Parameters
    ----------
    df_listings : pd.DataFrame
        Pandas dataframe with information about listings.
    df_daily_revenue : pd.DataFrame
        Pandas dataframe with information about daily revenue.
    """

    data = pd.merge(
        df_daily_revenue,
        df_listings[["Código", "Comissão"]],
        left_on="listing",
        right_on="Código",
        how="left",
    )

    data["company_revenue"] = data["Comissão"] * data["revenue"]

    data = (
        data.groupby("date")
        .agg(company_revenue=("company_revenue", "sum"))
        .reset_index()
    )

    data_pred = pd.DataFrame()
    data_pred["date"] = pd.date_range(
        start=data["date"].min(), end=data["date"].max()
    ).to_list()

    data_pred = build_date_features(data_pred, "date")

    preprocessor = load_pickle("models/preprocessor_covid_impact_model.pickle")
    model = load_pickle("models/regressor_covid_impact_model.pickle")

    X_pred = preprocess_transform(data_pred, preprocessor)

    data_pred["predicted_company_revenue"] = model.predict(X_pred)
    data_pred["date"] = get_date_from_ymd(data_pred)

    loss_due_to_pandemic = np.sum(
        data_pred["predicted_company_revenue"] - data["company_revenue"]
    )

    print(
        "Estimated loss on revenue due to Covid-19 pandemic (R$): {:.2f}".format(
            loss_due_to_pandemic
        )
    )

    plot_revenue_loss_due_to_covid(df_listings, df_daily_revenue)


def answer_complementary_data_analysis(df_daily_revenue):
    """Complementary data analysis of the dataset provided.

    Parameters
     ----------
     df_daily_revenue : pd.DataFrame
         Pandas dataframe with information about daily revenue.
    """

    plot_revenue_per_date(df_daily_revenue)
    plot_hist_reservation_advance(df_daily_revenue)
    print_reservation_advance_quantiles(df_daily_revenue)
