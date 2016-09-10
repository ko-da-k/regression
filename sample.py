#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
import regression

if __name__ == "__main__":
    iris = datasets.load_iris()
    data = pd.DataFrame(iris.data, columns=['SepalLength', 'SepalWidth',
                                            'PetalLength', 'PetalWidth'])
    data["Spiecies"] = iris.target

    model = regression.Regression(data, ["SepalLength", "SepalWidth", "PetalLength"], "PetalWidth")
    model.linear_regression()

    plt.subplot(311)
    plt.plot(data["PetalWidth"], "b")
    plt.subplot(312)
    plt.plot(model.predict(data[["SepalLength", "SepalWidth", "PetalLength"]]), "g")
    plt.subplot(313)
    plt.plot(data["PetalWidth"], "b")
    plt.plot(model.predict(data[["SepalLength", "SepalWidth", "PetalLength"]]), "g")
    plt.show()

    model.random_forest_regression()

