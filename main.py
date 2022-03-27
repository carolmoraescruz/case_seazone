# -*- coding: utf-8 -*-

import warnings

warnings.filterwarnings("ignore")

from src.data.make_dataset import load_data
from src.reports.reports import (
    answer_covid_impact_on_revenue,
    answer_first_question,
    answer_fourth_question,
    answer_second_question,
    answer_third_question,
    header_covid_impact_on_revenue,
    header_q1,
    header_q2,
    header_q3,
    header_q4,
    print_reservation_advance_quantiles,
)
from src.visualization.visualize import (
    plot_hist_reservation_advance,
    plot_real_pred_data,
    plot_revenue_loss_due_to_covid,
    plot_revenue_per_date,
    plot_seasonal_decomposed_q2,
    plot_seasonal_decomposed_q3,
)
from src.models.train_model import (
    train_covid_impact_model,
    train_price_model_q1,
    train_reservations_model_q3,
    train_revenue_model_q1,
    train_revenue_model_q2,
)


def main():
    """Main function"""

    df_listings, df_daily_revenue = load_data()

    # Question 01
    header_q1()
    train_price_model_q1(df_listings, df_daily_revenue)
    train_revenue_model_q1(df_listings, df_daily_revenue)
    answer_first_question()

    # Question 02
    header_q2()
    train_revenue_model_q2(df_listings, df_daily_revenue)
    answer_second_question()
    plot_real_pred_data(df_listings, df_daily_revenue)
    plot_seasonal_decomposed_q2(df_listings, df_daily_revenue)

    # Question 03
    header_q3()
    train_reservations_model_q3(df_daily_revenue)
    answer_third_question()
    plot_seasonal_decomposed_q3(df_daily_revenue)

    # Question 04
    header_q4()
    answer_fourth_question(df_daily_revenue)

    # Impact of Covid-19 pandemic on revenue
    header_covid_impact_on_revenue()
    train_covid_impact_model(df_listings, df_daily_revenue)
    answer_covid_impact_on_revenue(df_listings, df_daily_revenue)
    plot_revenue_loss_due_to_covid(df_listings, df_daily_revenue)

    # Complementary Data Analysis
    plot_revenue_per_date(df_daily_revenue)
    plot_hist_reservation_advance(df_daily_revenue)
    print_reservation_advance_quantiles(df_daily_revenue)


if __name__ == "__main__":
    main()
