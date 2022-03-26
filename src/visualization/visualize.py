# -*- coding: utf-8 -*-

# Dependencies:

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.commons import WEEK_DAY_ORDER, is_holiday, load_pickle

from src.features.build_features import build_features_revenue_model_q2
from src.models.preprocessing import one_hot_encode_column, preprocess_transform


def plot_revenue_per_date(df_daily_revenue):
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


def plot_hist_reservation_advance(df_daily_revenue):
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


def plot_real_pred_data(df_listings, df_daily_revenue):

    X, y = build_features_revenue_model_q2(df_listings, df_daily_revenue)

    X["date"] = X.apply(
        lambda x: pd.to_datetime(
            str(int(x["year"])) + "-" + str(int(x["month"])) + "-" + str(int(x["day"]))
        ),
        axis=1,
    )

    data_pred = pd.DataFrame()
    data_pred["date"] = pd.date_range(start=X["date"].min(), end=X["date"].max())

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
    print("Exporting graph histogram_reservation_advance to path: " + path)
