#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
回帰分析を行う
"""

import numpy as np
from sklearn import linear_model,svm
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
        self.explanatory_columns = explanatory_columns
        self.criterion_column = criterion_column
        self.explanatory_variables = dataFrame[self.explanatory_columns].as_matrix()
        self.criterion_variables = dataFrame[self.criterion_column].as_matrix()

    def linear_regression(self):
        self.clf = linear_model.LinearRegression(fit_intercept=True,normalize=False,
                                                 copy_X=True,n_jobs=-1)
        """
        :param fit_intercept: Falseにすると切片を求めない(原点を通る場合に有効)
        :param normalize: Trueにすると説明変数を正規化する
        :param copy_X: メモリないでデータを複製するかどうか
        """
        self.clf.fit(self.explanatory_variables,self.criterion_variables)
        self.coef = pd.DataFrame({"Name":self.explanatory_columns,
                                  "Coefficients":self.clf.coef_})
        self.intercept = self.clf.intercept_
        self.predict = lambda x: self.clf.predict(x)

    def svr(self):
        self.clf = svm.SVR(kernel='rbf',C=1e3,gamma=0.1)
        self.clf.fit(self.explanatory_variables,self.criterion_variables)
        self.predict = lambda x: self.clf.predict(x)
