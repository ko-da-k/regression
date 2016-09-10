#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
回帰分析を行う
"""

import numpy as np
from sklearn import linear_model,svm
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Regression:
    """線形重回帰分析"""

    def __init__(self,dataFrame: pd.DataFrame, explanatory_columns: list, criterion_column: str):
        """
        :param dataFrame: データセット
        :param explanatory_columns: 説明変数,単回帰でもlist
        :param criterion_column:  目的変数
        :param clf: 回帰式
        :param coef: 偏回帰係数
        :param intercept: 誤差
        """
        self.data = dataFrame
        self.explanatory_columns = explanatory_columns
        self.criterion_column = criterion_column
        self.explanatory_variables = dataFrame[self.explanatory_columns].as_matrix()
        self.criterion_variables = dataFrame[self.criterion_column].as_matrix()

    def linear_regression(self):
        self.rm = linear_model.LinearRegression(fit_intercept=True,normalize=False,
                                                 copy_X=True,n_jobs=-1)
        """
        :param fit_intercept: Falseにすると切片を求めない(原点を通る場合に有効)
        :param normalize: Trueにすると説明変数を正規化する
        :param copy_X: メモリないでデータを複製するかどうか
        """
        self.rm.fit(self.explanatory_variables,self.criterion_variables)
        self.coef = pd.DataFrame({"Name":self.explanatory_columns,
                                  "Coefficients":self.rm.coef_})
        self.intercept = self.rm.intercept_
        self.predict = lambda x: self.rm.predict(x)

    def svr(self):
        self.rm = svm.SVR(kernel='rbf',C=1,gamma=0.1)
        self.rm.fit(self.explanatory_variables,self.criterion_variables)
        self.predict = lambda x: self.rm.predict(x)

    def random_forest_regression(self):
        self.rm = RandomForestRegressor(n_estimators=100,criterion="mse",
                                   max_features="auto",n_jobs=-1)
        self.rm.fit(self.explanatory_variables,self.criterion_variables)

        features_importance_rank = np.argsort(self.rm.feature_importances_)[::-1]
        features_importance_value = self.rm.feature_importances_[features_importance_rank]
        features_importance_key = self.data[features_importance_rank].keys()
        importance = pd.DataFrame(
            {
                "key": features_importance_key,
                "value": features_importance_value
            }
        )
        sns.barplot(x='value', y='key', data=importance)
        plt.show()
