# regression
回帰分析サンプル
irisデータで実験

単回帰、重回帰の場合
```
model = regression.Regression(data, ["SepalLength", "SepalWidth", "PetalLength"], "PetalWidth")
model.linear_regression()
```

サポートベクター回帰の場合
```
model = regression.Regression(data, ["SepalLength", "SepalWidth", "PetalLength"], "PetalWidth")
model.svr()
```

予測をしたい場合
```
model.predict(x)
```
