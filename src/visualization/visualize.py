# -*- coding: utf-8 -*-

# Dependencies:

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_revenue_per_date(df_daily_revenue):
    path = "reports/figures/revenue_per_date.png"
    temp = df_daily_revenue.groupby("date")[["revenue"]].mean().reset_index()

    sns.lineplot(data=temp, x="date", y="revenue")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    print("\n{}".format(89 * "*"))
    print("Exporting graph revenue_per_date to path: " + path)


def plot_hist_reservation_advance(df_daily_revenue):
    path = "reports/figures/histogram_reservation_advance.png"

    plt.hist(df_daily_revenue["reservation_advance_days"].dropna(), bins=100)
    plt.tight_layout()
    plt.savefig("reports/figures/histogram_reservation_advance.png")
    plt.close()

    print("\n{}".format(89 * "*"))
    print("Exporting graph histogram_reservation_advance to path: " + path)
