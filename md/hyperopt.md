ハイパーパラメータをハイパー賢くサーチする  
hyperoptについて

---

### 目次
- 今回のモチベーションと文献
- パラメータチューニングって何？
- パラメータ探索方法
  - グリッドサーチ
  - ランダムサーチ
- hyperopt
  - Sequential Model-based Optimization(SMBO)
  - GPによるSMBO
  - TPEによるSMBO
- まとめ

>>>

### 今回のコード！

https://drive.google.com/open?id=1KsGf0iki7NsPHSBAYj28nF5FTHDvuVwq

>>>

### 今回のモチベーションと文献
- 読んだ文献: Algorithms for Hyper-Parameter Optimization
- 初出学会: NIPS 2011
- 目的: 仕事でデータ分析をする際に精度の高い機械学習モデルを作らなければならない。  
  そのときに学習モデルのハイパーパラメータチューニングが重要となってくる。  
  ハイパーパラメータチューニングの定番手法hyperoptの中の仕組みを知ることで、この分野ではどのような手法で最適化しているかを理解したい。

---

## ハイパーパラメータチューニングとは？

>>>

- 以下の問題を考える。
  - タイタニック号の乗客リスト（名前、性別、部屋の等級などの情報）が与えられており、そこからタイタニックが沈没した際の生死を予測したい
  - 出展: kaggle https://www.kaggle.com/

>>>

まず簡単に予測してみる:

```
# データ読み込み処理略

### なにもせずに学習

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

columns = [col for col in df_train.columns if col != objective_variable_name]
X_train = df_train[columns]
y_train = df_train[objective_variable_name]

# cv scoreを出す
clf = RandomForestClassifier()
cv_score = cross_val_score(clf, X_train, y_train, cv=5)
cv_score.mean()
```

- 汎化性能: `0.8125`

>>>

ここで、機械学習アルゴリズム(RandomForest分類器)のパラメータを調整して精度を高くしたい、このように目的となる指標を最大化する問題があるので**ハイパーパラメータチューニング**の重要となる。

実際やってみるが

```
class sklearn.ensemble.RandomForestClassifier(n_estimators=’warn’, criterion=’gini’, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=’auto’, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None)[source]
```

地獄のようにパラメータが多い。  
( https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html )

>>>

ここで、以下の方法でパラメータを調整して精度を高めようと考える

1. いくつかのパラメータを試してみて最も良かったものを採用する
2. データ分析力が高まって、最も良いパラメータが発見される
3. できない。現実は非常である。

2,3を選ぶと今回の発表が終わってしまうので、1の方法で精度を高める。  
(傍から見てる割と2が多いんじゃないかって思うけど)

---

## パラメータ探索方法

>>>

- GridSearch
  - パラメータごとに探索する値を指定して、それらのすべての組み合わせを探索する。
  - 指定した値のパターンが特徴ごとに`$a_1, \cdots ,a_n$`あったら`$\prod_{i} a_i$`必要になる。

![画像](img/hyperopt/gridsearch.png)

>>>

コードの例

```
### GridSearch
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


params = {
  'n_estimators': [10, 25, 50, 75, 100],
  'max_depth': [None, 5, 10, 25],
  'min_samples_split': [5, 10, 15]
}

# 与えられたパラメータ全てのパターンを学習した結果をcvしてモデルの精度を見る
gs = GridSearchCV( RandomForestClassifier(), params, cv=5)
gs.fit(X_train, y_train)
gs.best_score_
```

- 最も良かった汎化性能: `0.8338`

>>>

- RandomSearch
  - 探索するパラメータごとに確率分布を指定して、分布に従い乱数を発生させて探索する。
  - ユーザーが指定した回数の試行を繰り返す。

![画像](img/hyperopt/randomsearch.png)

>>>

コードの例

```
param_dists = {
  'n_estimators': sp_randint(10, 100),
  'max_depth': sp_randint(5, 50),
  'min_samples_split': sp_randint(5, 20)
}

rs = RandomizedSearchCV( RandomForestClassifier(), param_dists, n_iter=60, cv=5)
rs.fit(X_train, y_train)
```

- 最も良かった汎化性能: `0.8350`

---

## hyperoptとは

>>>

- GridSearchはユーザーが指定した値を総当たりしていて、RandomSearchは乱数で探索しているだけである。
- もっと賢くパラメータ探索できないか？
- hyperoptはハイパーパラメータチューニングに適した探索を行います！

>>>

### Sequential Model-based Optimization(SMBO)
- hyperoptはSMBOというベースとなるアルゴリズムを使ってる。

![画像](img/hyperopt/SMBO.PNG)

- 最適化したい関数`$f$`に対して、最適な値（上ではlossを考えてfの最小値を求めようとしている）を求めるアルゴリズムである。

>>>

- 最適化する`$f$`の代わりになる確率関数(surrogate)を考えて、最適な値を求めていく。
- 探索するたびにsurrogateの確率モデルを更新していき最適化を行う。
- `$S$`: 期待値関数、surrogateを評価するための関数
  - surrogateの期待値が最小となるようなハイパーパラメータを求めるために使われる。
- `$M$`: surrogateの関数を表現する確率モデル。探索済みデータ`$D_t$`を受け取るごとにパラメータ`$x$`を受け取ったときの`$y$`の確率`$p(y|x, D_t)$`の分布となっている。

>>>

- SMBOは`$S$`と`$M$`を予め指定しなければいけない
- hyperoptでは`$S$`はExpected Improvement(EI)という関数を用意(`$y^*$`はその時点でのベストな値)

`$$EI_{y^*} (x) = \int_{\mathbb{R}} max(y^*-y,0)p_M(y|x)dy$$`

- `$M$`にはGaussian ProcessとTree Parzen Estimatorを用いる。

>>>

- SMBOの直観的な図

![画像](img/hyperopt/expected-improvement-example.png)

- 探索したデータを使い逐次最適な解を見つけていく

>>> 

コード例

WIP

---

## GPとTPE

>>>

### GP

>>>

### TPE

---

## まとめ

>>>

- わかったこと
  - ハイパーパラメータチューニングの手法として知られるhyperoptの中身を理解することができた
    - SMBOという枠組みとなるアルゴリズムを用いて、GP,TPEによってノンパラメトリックベイズ推定する
    - スコアを出力する関数はブラックボックスなので、基準となる関数を代理のものとして近似する
    - ハイパーパラメータをサンプリングしながら、逐次基準となる分布を更新していく
- わからなかったこと
  - 出来ればSMBOで得られる解がブラックボックス関数の最適解に近似することの理論的裏付けを知りたかったけど、関連論文を読み切れなかった。
    - SMBOのSとMがブラックボックス関数の構造とどれほど関係があるのかなどの数理的裏付けを知りたい。
  - ノンパラメトリックベイズ推定がふんわりとしか理解できてないので、やってることは分かるけど、具体的な計算をどうするかまで踏み込めなかった。
  
>>>

- 参考
  - hyperoptって何してんの？
    https://qiita.com/kenchin110100/items/ac3edb480d789481f134
  - Hyperoptとその周辺について
    https://www.slideshare.net/hskksk/hyperopt
  - Hyperparameter optimization for Neural Networks
    http://neupy.com/2016/12/17/hyperparameter_optimization_for_neural_networks.html