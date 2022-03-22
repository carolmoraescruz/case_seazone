# -*- coding: utf-8 -*-

from src.data.make_dataset import load_data
from src.reports.reports import (
    answer_first_question,
    print_reservation_advance_quantiles,
)
from src.visualization.visualize import (
    plot_hist_reservation_advance,
    plot_revenue_per_date,
)


def main():
    """Main function"""

    df_listings, df_daily_revenue = load_data()

    # Question 01
    answer_first_question(df_listings, df_daily_revenue)

    # Complementary Data Analysis
    plot_revenue_per_date(df_daily_revenue)
    plot_hist_reservation_advance(df_daily_revenue)
    print_reservation_advance_quantiles(df_daily_revenue)


if __name__ == "__main__":
    main()
