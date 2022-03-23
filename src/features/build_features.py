# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from src.models.preprocessing import one_hot_encode_column
from src.commons import WEEK_DAY_ORDER, is_holiday


def build_daily_features(df_daily_revenue):
    df_daily_revenue["reservation_advance_days"] = (
        df_daily_revenue["date"] - df_daily_revenue["creation_date"]
    ).dt.days

    df_daily_revenue.loc[
        df_daily_revenue["reservation_advance_days"] < 0, "reservation_advance_days"
    ] = np.nan

    return df_daily_revenue


def build_features_revenue_model(df_daily_revenue, df_listings):
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
