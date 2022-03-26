# -*- coding: utf-8 -*-

# Dependencies:

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from src.commons import WEEK_DAY_ORDER, load_pickle
from src.features.build_features import (
    build_date_features,
    build_features_revenue_model_q2,
)
from src.models.preprocessing import preprocess_transform


def plot_revenue_per_date(df_daily_revenue: pd.DataFrame):
    """Plots a graph of revenue per date.

    Parameters
    ----------
    df_daily_revenue : pd.DataFrame
        Pandas dataframe with information about daily revenue.
    """
    path = "reports/figures/revenue_per_date.png"
    temp = df_daily_revenue.groupby("date")[["revenue"]].mean().reset_index()

    plt.style.use("seaborn")
    sns.lineplot(data=temp, x="date", y="revenue")
    plt.xticks(rotation=45)
    plt.xlabel("Date")
    plt.ylabel("Revenue (R$)")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    print("\n{}".format(89 * "*"))
    print("Exporting graph revenue_per_date to path: " + path)


def plot_hist_reservation_advance(df_daily_revenue: pd.DataFrame):
    """Plots a histogram with the distribution of booking advance days.

    Parameters
    ----------
    df_daily_revenue : pd.DataFrame
        Pandas dataframe with information about daily revenue.
    """
    path = "reports/figures/histogram_reservation_advance.png"

    plt.style.use("seaborn")
    plt.hist(df_daily_revenue["reservation_advance_days"].dropna(), bins=100)
    plt.xlabel("Reservation advance (days)")
    plt.ylabel("Number of reservations")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    print("\n{}".format(89 * "*"))
    print("Exporting graph histogram_reservation_advance to path: " + path)


def plot_real_pred_data(df_listings: pd.DataFrame, df_daily_revenue: pd.DataFrame):
    """Plots a graph comparing the real and the predicted revenue.

    Parameters
    ----------
    df_listings : pd.DataFrame
        Pandas dataframe with information about listings.
    df_daily_revenue : pd.DataFrame
        Pandas dataframe with information about daily revenue.
    """

    X, y = build_features_revenue_model_q2(df_listings, df_daily_revenue)

    X["date"] = X.apply(
        lambda x: pd.to_datetime(
            str(int(x["year"])) + "-" + str(int(x["month"])) + "-" + str(int(x["day"]))
        ),
        axis=1,
    )

    data_pred = pd.DataFrame()
    data_pred["date"] = pd.date_range(start=X["date"].min(), end=X["date"].max())

    data_pred = build_date_features(data_pred, "date")

    preprocessor = load_pickle("models/preprocessor_revenue_model_q2.pickle")
    model = load_pickle("models/regressor_revenue_model_q2.pickle")

    X_pred = preprocess_transform(data_pred, preprocessor)

    y_pred = model.predict(X_pred)

    path = "reports/figures/real_versus_predicted_revenue.png"

    plt.style.use("seaborn")
    plt.plot(X["date"], y, label="Real revenue")
    plt.plot(X["date"], y_pred, label="Predicted revenue")
    plt.xticks(rotation=45)
    plt.xlabel("Date")
    plt.ylabel("Revenue (R$)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    print("\n{}".format(89 * "*"))
    print("Exporting graph real_versus_predicted_revenue to path: " + path)


def plot_seasonal_decomposed_q2(
    df_listings: pd.DataFrame, df_daily_revenue: pd.DataFrame
):
    """Plots the graphs of seasonal decomposition for question 2.

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

    data_revenue = (
        data.groupby("date")
        .agg(company_revenue=("company_revenue", "sum"))
        .reset_index()
    )

    data_revenue = build_date_features(data_revenue, "date")

    data = data_revenue.loc[data_revenue["company_revenue"].notna()]

    tsmodel = seasonal_decompose(
        data["company_revenue"],
        model="additive",
        extrapolate_trend="freq",
        freq=365,
    )

    path = "reports/figures/seasonal_decompose_revenue_q2.png"

    plt.style.use("seaborn")
    plt.rcParams.update({"figure.figsize": (10, 10)})
    tsmodel.plot()
    plt.xlabel("Days")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    print("\n{}".format(89 * "*"))
    print("Exporting graph seasonal_decompose_revenue to path: " + path)


def plot_seasonal_decomposed_q3(df_daily_revenue: pd.DataFrame):
    """Plots the graphs of seasonal decomposition for question 3.

    Parameters
    ----------
    df_daily_revenue : pd.DataFrame
        Pandas dataframe with information aboutt daily revenue.
    """

    df_q3 = df_daily_revenue[
        (df_daily_revenue["occupancy"] == 1) & (df_daily_revenue["blocked"] == 0)
    ]

    data_q3 = df_q3.groupby(["creation_date"]).count().iloc[:, 0:1].reset_index()
    data_q3.columns = ["creation_date", "qt_reservations"]

    data_q3 = build_date_features(data_q3, "creation_date")

    tsmodel = seasonal_decompose(
        data_q3["qt_reservations"],
        model="additive",
        extrapolate_trend="freq",
        freq=365,
    )

    path = "reports/figures/seasonal_decompose_reservations_q3.png"

    plt.style.use("seaborn")
    plt.rcParams.update({"figure.figsize": (10, 10)})
    tsmodel.plot()
    plt.xlabel("Days")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    print("\n{}".format(89 * "*"))
    print("Exporting graph seasonal_decompose_reservations to path: " + path)
