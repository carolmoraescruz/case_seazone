# -*- coding: utf-8 -*-

# Dependencies:

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from src import (
    PATH_COVID_IMPACT_GRAPH,
    PATH_HISTOGRAM_BOOKINGS,
    PATH_PLOT_REVENUE_PER_DATE,
    PATH_PREPROCESSOR_COVID_IMPACT,
    PATH_PREPROCESSOR_REVENUE_MODEL_Q2,
    PATH_REGRESSOR_COVID_IMPACT,
    PATH_REGRESSOR_REVENUE_MODEL_Q2,
    PATH_REVENUE_COMPARISON,
    PATH_SEASONAL_DECOMPOSE_RESERVATIONS,
    PATH_SEASONAL_DECOMPOSE_REVENUE,
)
from src.commons import get_date_from_ymd, load_pickle
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
    temp = df_daily_revenue.groupby("date")[["revenue"]].mean().reset_index()

    plt.style.use("seaborn")
    sns.lineplot(data=temp, x="date", y="revenue")
    plt.xticks(rotation=45)
    plt.xlabel("Date")
    plt.ylabel("Revenue (R$)")
    plt.tight_layout()
    plt.savefig(PATH_PLOT_REVENUE_PER_DATE)
    plt.close()

    print("Exporting graph revenue_per_date to path: " + PATH_PLOT_REVENUE_PER_DATE)


def plot_hist_reservation_advance(df_daily_revenue: pd.DataFrame):
    """Plots a histogram with the distribution of booking advance days.

    Parameters
    ----------
    df_daily_revenue : pd.DataFrame
        Pandas dataframe with information about daily revenue.
    """

    plt.style.use("seaborn")
    plt.hist(df_daily_revenue["reservation_advance_days"].dropna(), bins=100)
    plt.xlabel("Reservation advance (days)")
    plt.ylabel("Number of reservations")
    plt.tight_layout()
    plt.savefig(PATH_HISTOGRAM_BOOKINGS)
    plt.close()

    print(
        "Exporting graph histogram_reservation_advance to path: "
        + PATH_HISTOGRAM_BOOKINGS
    )


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

    preprocessor = load_pickle(PATH_PREPROCESSOR_REVENUE_MODEL_Q2)
    model = load_pickle(PATH_REGRESSOR_REVENUE_MODEL_Q2)

    X_pred = preprocess_transform(data_pred, preprocessor)

    y_pred = model.predict(X_pred)

    plt.style.use("seaborn")
    plt.plot(X["date"], y, label="Real revenue", alpha=0.8)
    plt.plot(X["date"], y_pred, label="Predicted revenue", color="orange", alpha=0.8)
    plt.xticks(rotation=45)
    plt.xlabel("Date")
    plt.ylabel("Revenue (R$)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PATH_REVENUE_COMPARISON)
    plt.close()

    print(
        "Exporting graph real_versus_predicted_revenue to path: "
        + PATH_REVENUE_COMPARISON
    )


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

    try:
        tsmodel = seasonal_decompose(
            data["company_revenue"],
            model="additive",
            extrapolate_trend="freq",
            freq=365,
        )
    except Exception as err:
        tsmodel = seasonal_decompose(
            data["company_revenue"],
            model="additive",
            extrapolate_trend="freq",
            period=365,
        )

    plt.style.use("seaborn")
    plt.rcParams.update({"figure.figsize": (10, 10)})
    tsmodel.plot()
    plt.xlabel("Days")
    plt.tight_layout()
    plt.savefig(PATH_SEASONAL_DECOMPOSE_REVENUE)
    plt.close()

    print(
        "Exporting graph seasonal_decompose_revenue to path: "
        + PATH_SEASONAL_DECOMPOSE_REVENUE
    )


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

    plt.style.use("seaborn")
    plt.rcParams.update({"figure.figsize": (10, 10)})
    tsmodel.plot()
    plt.xlabel("Days")
    plt.tight_layout()
    plt.savefig(PATH_SEASONAL_DECOMPOSE_RESERVATIONS)
    plt.close()

    print(
        "Exporting graph seasonal_decompose_reservations to path: "
        + PATH_SEASONAL_DECOMPOSE_RESERVATIONS
    )


def plot_revenue_loss_due_to_covid(
    df_listings: pd.DataFrame, df_daily_revenue: pd.DataFrame
):
    """Plots the graph comparing the revenue expected in comparison to
    real revenue in order to compare loss due to covid-19 pandemic.

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

    preprocessor = load_pickle(PATH_PREPROCESSOR_COVID_IMPACT)
    model = load_pickle(PATH_REGRESSOR_COVID_IMPACT)

    X_pred = preprocess_transform(data_pred, preprocessor)

    data_pred["predicted_company_revenue"] = model.predict(X_pred)
    data_pred["date"] = get_date_from_ymd(data_pred)

    plt.style.use("seaborn")
    plt.rcParams.update({"figure.figsize": (10, 10)})
    plt.plot(
        data["date"], data["company_revenue"], label="Real Company Revenue", alpha=0.8
    )
    plt.plot(
        data_pred["date"],
        data_pred["predicted_company_revenue"],
        label="Predicted Company Revenue",
        color="orange",
        alpha=0.8,
    )
    plt.xlabel("Date")
    plt.ylabel("Company Revenue (R$)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PATH_COVID_IMPACT_GRAPH)
    plt.close()

    print("Exporting graph covid_impact_on_revenue to path: " + PATH_COVID_IMPACT_GRAPH)
