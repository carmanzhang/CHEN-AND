import time
import warnings

import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')

from enum import Enum


class ModelName(Enum):
    linear = 'Linear'
    logistic = 'Logistic'
    dt = 'DecisionTree'
    c45 = 'CART4.5'
    svm = 'SVM'
    xgboost = 'XGBoost'
    randomforest = 'RandomForest'
    gb = 'GradientBoosting'
    mlp = 'MultiLayerPerceptron'

    @classmethod
    def available_modes(self):
        # return [self.linear, self.logistic, self.dt, self.randomforest, self.svm, self.mlp]  #  self.svm, self.gb self.xgboost,
        # return [self.linear, self.logistic, self.dt, self.randomforest]  # , self.svm, self.mlp
        return [self.logistic, self.dt, self.randomforest, self.svm]  # , self.svm, self.mlp
        # return [self.linear, self.logistic, self.dt, self.randomforest] # self.svm,

    @classmethod
    def get_short_name(self, model_name):
        return \
            dict(zip(
                [self.linear, self.logistic, self.dt, self.c45, self.svm, self.xgboost, self.randomforest, self.mlp],
                ['Linear', 'LR', 'DecisionTree', 'C45', 'SVM', 'XGB', 'RF', 'MLP']))[model_name]  # , self.gb


def use_regression(X_train, Y_train, X_test, model_switch):
    if model_switch == ModelName.linear:
        model, pred_y, feature_importance = linear_regressor(X_train, Y_train, X_test)
    elif model_switch == ModelName.logistic:
        model, pred_y, coefs, _ = logistic_regressor(X_train, Y_train, X_test)
        feature_importance = np.array(coefs[0])
    elif model_switch == ModelName.dt:
        model, pred_y, feature_importance = dt_regressor(X_train, Y_train, X_test)
    elif model_switch == ModelName.svm:
        model, pred_y, feature_importance = svm_regressor(X_train, Y_train, X_test)
    elif model_switch == ModelName.xgboost:
        model, pred_y, feature_importance = xgboost_regressor(X_train, Y_train, X_test)
    elif model_switch == ModelName.randomforest:
        model, pred_y, feature_importance = randomforest_regressor(X_train, Y_train, X_test)
    elif model_switch == ModelName.gb:
        model, pred_y, feature_importance = gb_regressor(X_train, Y_train, X_test)
    else:
        pass
    return model, pred_y, feature_importance


def linear_regressor(X_train, Y_train, X_test):
    model = LinearRegression()
    model.fit(X_train, Y_train)
    s = time.time()
    y_pred = model.predict(X_test)
    # print('used time: ', time.time() - s)
    return model, y_pred, model.coef_


def logistic_regressor(X_train, Y_train, X_test):
    # model = LogisticRegression(max_iter=1000, solver='newton-cg', tol=1e-5)
    model = LogisticRegression(max_iter=1000, tol=1e-4, class_weight='balanced', C=2)
    model.fit(X_train, Y_train)
    s = time.time()
    y_pred = model.predict_proba(X_test)
    # print('used time: ', time.time() - s)
    # metrics = calc_metrics(Y_train, [p1 for (p0, p1) in y_pred])
    # pprint(metrics, pctg=True)
    y_pred = [p1 for (p0, p1) in y_pred]
    print(model.coef_, model.intercept_)
    return model, y_pred, model.coef_, model.intercept_


def dt_regressor(X_train, Y_train, X_test):
    model = DecisionTreeRegressor(max_depth=6)  # max_depth=,
    model.fit(X_train, Y_train)
    # depth = model.get_depth()
    # for i in range(depth):
    #     print(model.get_params(i+1))
    s = time.time()
    y_pred = model.predict(X_test)
    # print('used time: ', time.time() - s)
    return model, y_pred, model.feature_importances_


def svm_regressor(X_train, Y_train, X_test):
    model = SVC()  # max_depth=,
    model.fit(X_train, Y_train)
    s = time.time()
    y_pred = model.predict(X_test)
    # print('used time: ', time.time() - s)
    return model, y_pred, np.array([])


def xgboost_regressor(X_train, Y_train, X_test):
    # params = {'colsample_bytree': 0.9, 'reg_alpha': 3, 'reg_lambda': 1, 'random_state': int(time.time()),
    #           'learning_rate': 0.01,
    #           'n_estimators': 100,
    #           'max_depth': 8,
    #           'min_child_weight': 2,
    #           'gamma': 0.1,
    #           'subsample': 0.7,
    #           }
    # model = XGBRegressor(**params)
    model = XGBRegressor()
    # return model
    model.fit(X_train, Y_train)
    s = time.time()
    y_pred = model.predict(X_test)
    # print('used time: ', time.time() - s)
    return model, y_pred, model.feature_importances_


def randomforest_regressor(X_train, Y_train, X_test):
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, Y_train)
    s = time.time()
    y_pred = model.predict(X_test)
    # print('used time: ', time.time() - s)
    return model, y_pred, model.feature_importances_


def gb_regressor(X_train, Y_train, X_test):
    model = GradientBoostingRegressor(learning_rate=0.05, n_estimators=100)
    model.fit(X_train, Y_train)
    s = time.time()
    y_pred = model.predict(X_test)
    # print('used time: ', time.time() - s)
    return model, y_pred, model.feature_importances_
