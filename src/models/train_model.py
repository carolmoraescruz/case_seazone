# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error as mae
from src.features.build_features import (
    build_features_reservations_model_q3,
    build_features_revenue_model_q1,
    build_features_revenue_model_q2,
    build_features_price_model_q1,
)
from src.models.preprocessing import (
    fit_preprocess_reservations_model_q3,
    fit_preprocess_revenue_model_q2,
    preprocess_transform,
)
from src.commons import dump_pickle


def train_price_model_q1(df_listings, df_daily_revenue):

    print("Training price model - Q1")

    X, y = build_features_price_model_q1(df_listings, df_daily_revenue)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    preprocessor = fit_preprocess_revenue_model_q2(X_train)

    X_train = preprocess_transform(X_train, preprocessor)

    X_test = preprocess_transform(X_test, preprocessor)

    model = XGBRegressor(max_depth=6, n_estimators=300).fit(X_train, y_train)

    score = mae(y_test, model.predict(X_test))

    dump_pickle(preprocessor, "models/preprocessor_price_model_q1.pickle")

    dump_pickle(model, "models/regressor_price_model_q1.pickle")

    print("MAE(teste) = {:.2f}".format(score))


def train_revenue_model_q1(df_listings, df_daily_revenue):

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

    dump_pickle(preprocessor, "models/preprocessor_revenue_model_q1.pickle")

    dump_pickle(model, "models/regressor_revenue_model_q1.pickle")

    print("MAE(teste) = {:.2f}".format(score))


def train_revenue_model_q2(df_listings, df_daily_revenue):

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

    dump_pickle(preprocessor, "models/preprocessor_revenue_model_q2.pickle")

    dump_pickle(model, "models/regressor_revenue_model_q2.pickle")

    print("MAE(teste) = {:.2f}".format(score))


def train_reservations_model_q3(df_daily_revenue):

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

    dump_pickle(preprocessor, "models/preprocessor_reservations_model_q3.pickle")

    dump_pickle(model, "models/regressor_reservations_model_q3.pickle")

    print("MAE(teste) = {:.2f}".format(score))
