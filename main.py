# -*- coding: utf-8 -*-

import warnings

warnings.filterwarnings("ignore")

from src.data.make_dataset import load_data
from src.reports.reports import (
    answer_complementary_data_analysis,
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
    answer_second_question(df_listings, df_daily_revenue)

    # Question 03
    header_q3()
    train_reservations_model_q3(df_daily_revenue)
    answer_third_question(df_daily_revenue)

    # Question 04
    header_q4()
    answer_fourth_question(df_daily_revenue)

    # Impact of Covid-19 pandemic on revenue
    header_covid_impact_on_revenue()
    train_covid_impact_model(df_listings, df_daily_revenue)
    answer_covid_impact_on_revenue(df_listings, df_daily_revenue)

    # Complementary Data Analysis
    answer_complementary_data_analysis(df_daily_revenue)


if __name__ == "__main__":
    main()
