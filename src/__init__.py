__version__ = "1.0.0"

__short_version__ = "1.0"

PATH_LISTINGS = "data/raw/listings-challenge.csv"

PATH_DAILY_REVENUE = "data/raw/daily_revenue.csv"

PATH_PLOT_REVENUE_PER_DATE = "reports/figures/revenue_per_date.png"

PATH_HISTOGRAM_BOOKINGS = "reports/figures/histogram_reservation_advance.png"

PATH_REVENUE_COMPARISON = "reports/figures/real_versus_predicted_revenue.png"

PATH_SEASONAL_DECOMPOSE_REVENUE = "reports/figures/seasonal_decompose_revenue_q2.png"

PATH_SEASONAL_DECOMPOSE_RESERVATIONS = (
    "reports/figures/seasonal_decompose_reservations_q3.png"
)

PATH_COVID_IMPACT_GRAPH = "reports/figures/covid_impact_on_revenue.png"

PATH_PREPROCESSOR_PRICE_MODEL_Q1 = "models/preprocessor_price_model_q1.pickle"

PATH_REGRESSOR_PRICE_MODEL_Q1 = "models/regressor_price_model_q1.pickle"

PATH_PREPROCESSOR_REVENUE_MODEL_Q1 = "models/preprocessor_revenue_model_q1.pickle"

PATH_REGRESSOR_REVENUE_MODEL_Q1 = "models/regressor_revenue_model_q1.pickle"

PATH_PREPROCESSOR_REVENUE_MODEL_Q2 = "models/preprocessor_revenue_model_q2.pickle"

PATH_REGRESSOR_REVENUE_MODEL_Q2 = "models/regressor_revenue_model_q2.pickle"

PATH_PREPROCESSOR_RESERVATIONS_MODEL_Q3 = (
    "models/preprocessor_reservations_model_q3.pickle"
)

PATH_REGRESSOR_RESERVATIONS_MODEL_Q3 = "models/regressor_reservations_model_q3.pickle"

PATH_PREPROCESSOR_COVID_IMPACT = "models/preprocessor_covid_impact_model.pickle"

PATH_REGRESSOR_COVID_IMPACT = "models/regressor_covid_impact_model.pickle"

REFERENCE_DATE = "2022-03-15"

FEATURES_PRICE_MODEL_Q1 = [
    "Categoria",
    "Quartos",
    "year",
    "month",
    "day",
    "holiday",
    "day_of_week_Mon",
    "day_of_week_Sat",
    "day_of_week_Sun",
    "day_of_week_Thu",
    "day_of_week_Tue",
    "day_of_week_Wed",
    "Localização_BOM",
    "Localização_CAM",
    "Localização_CAN",
    "Localização_CEN",
    "Localização_CON",
    "Localização_GRA",
    "Localização_ILC",
    "Localização_ING",
    "Localização_ITA",
    "Localização_ITP",
    "Localização_JBV",
    "Localização_JUR",
    "Localização_LAG",
    "Localização_PBL",
    "Localização_SAN",
    "Localização_SLA",
    "Localização_STO",
    "Localização_TBM",
    "Localização_UFSC",
]

FEATURES_REVENUE_MODEL_Q1 = [
    "Categoria",
    "Quartos",
    "year",
    "month",
    "day",
    "holiday",
    "day_of_week_Mon",
    "day_of_week_Sat",
    "day_of_week_Sun",
    "day_of_week_Thu",
    "day_of_week_Tue",
    "day_of_week_Wed",
    "Localização_BOM",
    "Localização_CAM",
    "Localização_CAN",
    "Localização_CEN",
    "Localização_CON",
    "Localização_GRA",
    "Localização_ING",
    "Localização_ITA",
    "Localização_ITP",
    "Localização_JUR",
    "Localização_LAG",
    "Localização_PBL",
    "Localização_SAN",
    "Localização_STO",
    "Localização_TBM",
    "Localização_UFSC",
]
