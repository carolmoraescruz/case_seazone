# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error as mae
from src.features.build_features import build_features_revenue_model
from src.models.preprocessing import (
    fit_preprocess_revenue_model,
    preprocess_revenue_model,
)
from src.commons import dump_pickle


def train_revenue_model(df_listings, df_daily_revenue):

    X, y = build_features_revenue_model(df_listings, df_daily_revenue)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    preprocessor = fit_preprocess_revenue_model(X_train)

    X_train = preprocess_revenue_model(X_train, preprocessor)

    X_test = preprocess_revenue_model(X_test, preprocessor)

    model = MLPRegressor(
        hidden_layer_sizes=(5, 10, 10, 5, 5),
        solver="lbfgs",
        learning_rate="adaptive",
        learning_rate_init=0.03,
        max_iter=10000,
        random_state=42,
    ).fit(X_train, y_train)

    print(
        "\n{}\nRegression (MAE): {:.2f}".format(
            89 * "*", mae(y_test, model.predict(X_test))
        )
    )

    dump_pickle(preprocessor, "models/preprocessor_revenue_model.pickle")
    dump_pickle(model, "models/regressor_revenue_model.pickle")
