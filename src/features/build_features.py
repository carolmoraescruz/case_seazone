# -*- coding: utf-8 -*-


def build_daily_features(df_daily_revenue):
    df_daily_revenue["reservation_advance_days"] = (
        df_daily_revenue["date"] - df_daily_revenue["creation_date"]
    ).dt.days

    df_daily_revenue.loc[
        df_daily_revenue["reservation_advance_days"] < 0, "reservation_advance_days"
    ] = np.nan

    return df_daily_revenue
