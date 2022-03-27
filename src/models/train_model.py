# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error as mae
from src import (
    PATH_PREPROCESSOR_COVID_IMPACT,
    PATH_PREPROCESSOR_PRICE_MODEL_Q1,
    PATH_PREPROCESSOR_RESERVATIONS_MODEL_Q3,
    PATH_PREPROCESSOR_REVENUE_MODEL_Q1,
    PATH_PREPROCESSOR_REVENUE_MODEL_Q2,
    PATH_REGRESSOR_COVID_IMPACT,
    PATH_REGRESSOR_PRICE_MODEL_Q1,
    PATH_REGRESSOR_RESERVATIONS_MODEL_Q3,
    PATH_REGRESSOR_REVENUE_MODEL_Q1,
    PATH_REGRESSOR_REVENUE_MODEL_Q2,
)
from src.features.build_features import (
    build_features_covid_impact_model,
    build_features_reservations_model_q3,
    build_features_revenue_model_q1,
    build_features_revenue_model_q2,
    build_features_price_model_q1,
)
from src.models.preprocessing import (
    fit_preprocess_covid_impact_model,
    fit_preprocess_price_model_q1,
    fit_preprocess_reservations_model_q3,
    fit_preprocess_revenue_model_q2,
    preprocess_transform,
)
from src.commons import dump_pickle


def train_price_model_q1(df_listings: pd.DataFrame, df_daily_revenue: pd.DataFrame):
    """Trains the price estimator to be used on question 1.

    Parameters
    ----------
    df_listings : pd.DataFrame
        Pandas dataframe with information about listings.
    df_daily_revenue : pd.DataFrame
        Pandas dataframe with information about daily revenue.

    Returns
    -------
    pd.DataFrame
         Returns a sklearn-like regressor instance.
    """

    print("Training price model - Q1")

    X, y = build_features_price_model_q1(df_listings, df_daily_revenue)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    preprocessor = fit_preprocess_price_model_q1(X_train)

    X_train = preprocess_transform(X_train, preprocessor)

    X_test = preprocess_transform(X_test, preprocessor)

    model = XGBRegressor(max_depth=6, n_estimators=300).fit(X_train, y_train)

    score = mae(y_test, model.predict(X_test))

    dump_pickle(preprocessor, PATH_PREPROCESSOR_PRICE_MODEL_Q1)

    dump_pickle(model, PATH_REGRESSOR_PRICE_MODEL_Q1)

    print("MAE(teste) = {:.2f}".format(score))


def train_revenue_model_q1(df_listings: pd.DataFrame, df_daily_revenue: pd.DataFrame):
    """Trains the revenue estimator to be used on question 1.

    Parameters
    ----------
    df_listings : pd.DataFrame
        Pandas dataframe with information about listings.
    df_daily_revenue : pd.DataFrame
        Pandas dataframe with information about daily revenue.

    Returns
    -------
    pd.DataFrame
         Returns a sklearn-like regressor instance.
    """

    print("Training revenue model - Q1")

    X, y = build_features_revenue_model_q1(df_listings, df_daily_revenue)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    preprocessor = fit_preprocess_revenue_model_q2(X_train)

    X_train = preprocess_transform(X_train, preprocessor)

    X_test = preprocess_transform(X_test, preprocessor)

    model = XGBRegressor(max_depth=6, n_estimators=300).fit(X_train, y_train)

    score = mae(y_test, model.predict(X_test))

    dump_pickle(preprocessor, PATH_PREPROCESSOR_REVENUE_MODEL_Q1)

    dump_pickle(model, PATH_REGRESSOR_REVENUE_MODEL_Q1)

    print("MAE(teste) = {:.2f}".format(score))


def train_revenue_model_q2(df_listings: pd.DataFrame, df_daily_revenue: pd.DataFrame):
    """Trains the revenue estimator to be used on question 2.

    Parameters
    ----------
    df_listings : pd.DataFrame
        Pandas dataframe with information about listings.
    df_daily_revenue : pd.DataFrame
        Pandas dataframe with information about daily revenue.

    Returns
    -------
    pd.DataFrame
         Returns a sklearn-like regressor instance.
    """

    print("Training revenue model - Q2")

    X, y = build_features_revenue_model_q2(df_listings, df_daily_revenue)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    preprocessor = fit_preprocess_revenue_model_q2(X_train)

    X_train = preprocess_transform(X_train, preprocessor)

    X_test = preprocess_transform(X_test, preprocessor)

    model = MLPRegressor(
        hidden_layer_sizes=(5, 10, 10, 5, 5),
        solver="lbfgs",
        learning_rate="adaptive",
        learning_rate_init=0.03,
        max_iter=10000,
        random_state=42,
    ).fit(X_train, y_train)

    score = mae(y_test, model.predict(X_test))

    dump_pickle(preprocessor, PATH_PREPROCESSOR_REVENUE_MODEL_Q2)

    dump_pickle(model, PATH_REGRESSOR_REVENUE_MODEL_Q2)

    print("MAE(teste) = {:.2f}".format(score))


def train_reservations_model_q3(df_daily_revenue: pd.DataFrame):
    """Trains the revenue estimator to be used on question 3.

    Parameters
    ----------
    df_daily_revenue : pd.DataFrame
        Pandas dataframe with information about daily revenue.

    Returns
    -------
    pd.DataFrame
         Returns a sklearn-like regressor instance.
    """

    print("Training reservations model - Q3")

    X, y = build_features_reservations_model_q3(df_daily_revenue)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    preprocessor = fit_preprocess_reservations_model_q3(X_train)

    X_train = preprocess_transform(X_train, preprocessor)

    X_test = preprocess_transform(X_test, preprocessor)

    model = XGBRegressor(max_depth=6, n_estimators=100, reg_alpha=0.5).fit(
        X_train, y_train
    )

    score = mae(y_test, model.predict(X_test))

    dump_pickle(preprocessor, PATH_PREPROCESSOR_RESERVATIONS_MODEL_Q3)

    dump_pickle(model, PATH_REGRESSOR_RESERVATIONS_MODEL_Q3)

    print("MAE(teste) = {:.2f}".format(score))


def train_covid_impact_model(df_listings: pd.DataFrame, df_daily_revenue: pd.DataFrame):
    """Trains the revenue estimator to be used to estimate covid-19 impact.

    Parameters
    ----------
    df_listings : pd.DataFrame
        Pandas dataframe with information about listings.
    df_daily_revenue : pd.DataFrame
        Pandas dataframe with information about daily revenue.

    Returns
    -------
    pd.DataFrame
         Returns a sklearn-like regressor instance.
    """

    print("Training model for covid-19 impact on revenue")

    X, y = build_features_covid_impact_model(df_listings, df_daily_revenue)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    preprocessor = fit_preprocess_covid_impact_model(X_train)

    X_train = preprocess_transform(X_train, preprocessor)

    X_test = preprocess_transform(X_test, preprocessor)

    model = RandomForestRegressor(n_estimators=100, random_state=42).fit(
        X_train, y_train
    )

    score = mae(y_test, model.predict(X_test))

    dump_pickle(preprocessor, PATH_PREPROCESSOR_COVID_IMPACT)

    dump_pickle(model, PATH_REGRESSOR_COVID_IMPACT)

    print("MAE(teste) = {:.2f}".format(score))
