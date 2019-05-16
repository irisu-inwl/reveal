### ハイパーパラメータをハイパー賢くサーチする  
### hyperoptについて

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
- URL: https://dl.acm.org/citation.cfm?id=2986743
- 目的: 仕事でデータ分析をする際に精度の高い機械学習モデルを作らなければならない。  
  そのときに学習モデルのハイパーパラメータチューニングが重要となってくる。  
  ハイパーパラメータチューニングの定番手法hyperoptの中の仕組みを知ることで、この分野ではどのような手法で最適化しているかを理解したかった。

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

<div style="text-align:center;">
<img src="img/hyperopt/gridsearch.png" height="30%" width="30%">
</div>

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

<div style="text-align:center;">
<img src="img/hyperopt/randomsearch.png" height="30%" width="30%">
</div>

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
- hyperoptではハイパーパラメータチューニングに適した探索を行える！

>>>

### 問題の定式化
- 最適化したい対象: モデルの汎化性能や、テストデータの精度
- 探索範囲: 学習モデルのハイパーパラメータ
- 問題: 最適化関数は逐次学習しないと得られず、学習のたびに時間がかかってしまう(ブラックボックス関数最適化)

<div style="text-align:center;">
<img src="img/hyperopt/blackbox.png" height="50%" width="50%">
</div>

>>>

### Sequential Model-based Optimization(SMBO)
- hyperoptはSMBOというベースとなるアルゴリズムを使ってる。

<div style="text-align:center;">
<img src="img/hyperopt/SMBO.PNG">
</div>

- 最適化したい関数`$f$`に対して、最適な値（上ではlossを考えてfの最小値を求めようとしている）を求めるアルゴリズムである。

>>>

- 最適化する`$f$`の代わりになる確率関数(surrogate)を考えて、最適な値を求めていく。
- 探索するたびにsurrogateの確率モデルを更新していき最適化を行う。
- `$S$`: 期待値関数、surrogateを評価するための関数
  - surrogateの期待値が最小となるようなハイパーパラメータを求めるために使われる。
- `$M$`: surrogateの関数を表現する確率モデル。探索済みデータ`$D_t$`を受け取るごとにパラメータ`$x$`を受け取ったときの`$y$`の確率`$p(y|x, D_t)$`の分布となっている。

>>>

- SMBOは`$S$`と`$M$`を予め指定しなければいけない
- hyperoptでは
  - `$S$`はExpected Improvement(EI)という関数を用意(`$y^*$`はその時点でのベストな値)

`$$EI_{y^*} (x) = \int_{\mathbb{R}} max(y^*-y,0)p_M(y|x)dy$$`

  - `$M$`にはガウス過程(Gaussian Process, GP)とTree Parzen Estimator(TPE)を用いる。

>>>

- SMBOの直観的な図（ニューラルネットワークの隠れ層を最適化する例）

<div style="text-align:center;">
<img src="img/hyperopt/expected-improvement-example.png" height="50%" width="50%">
</div>

- 図のように探索したデータを使い逐次最適な解を見つけていくのがこの手法の肝

>>>

理論の解説は疲れるので、とりあえずコード例

```
from hyperopt import hp, tpe, Trials, fmin

# 探索範囲
hyperopt_params = {
    'n_estimators': 10 + hp.randint('n_estimators', 91),
    'max_depth': 5 + hp.randint('max_depth', 46),
    'min_samples_split': 5 + hp.randint('min_samples_split', 16)
}

scores = []
def objective(params, X_train, y_train):
    clf = RandomForestClassifier(**params)
    cv_score = cross_val_score(clf, X_train, y_train, cv=5)
    scores.append(cv_score.mean())
    return -1 * cv_score.mean()

obj = lambda params: objective(params, X_train, y_train)
iter_num = 60 # iteration回数
trials = Trials() # 試行の過程を記録するインスタンス

# 探索方法はTPEを使う
best = fmin(obj, hyperopt_params, algo=tpe.suggest, max_evals=iter_num, trials=trials, verbose=1)
```

- 結果: `0.836`

>>>

### ハイパーパラメータチューニングの精度比較

<div style="text-align:center;">
<img src="img/hyperopt/param_tune_boxplot.png" height="50%" width="50%">
</div>

---

## GPとTPE

>>>

### GP
- SMBOの確率モデル`$M$`にガウス過程(GP)を用いる。
- GPとは確率過程`$(X_t)_{t\in T}$`に対して、`$k$`個の確率変数を抜き出したとき`$k$`次元ガウス分布になる確率過程のことをいう: `$(X_{t_1},\cdots , X_{t_k})\sim N_k(\mu, \Sigma)$`
  - 平均と分散は`$t$`に関する1変数関数、2変数関数となるので、無限次元のガウス分布と言われたりする。
- ガウス過程モデルの推定は、データ`$D=(x_1,y_1),\cdots , (x_n,y_n)$`が与えられたとき、新しいデータ`$x$`が来た時の`$y$`の分布を推定する。
  - `$y\sim N(\mu(x;D),K(x;D))$`を求める。(Pythonで学ぶ統計的機械学習 13章)

>>>

- hyperoptではGPの推定に以下の工夫をしている
  - 離散値データに対して[EDA](https://en.wikipedia.org/wiki/Estimation_of_distribution_algorithm)(Estimation of Distribution)を利用する
  - 連続値データに対しては[CMA-ES](https://ja.wikipedia.org/wiki/CMA-ES)(Covariance Matrix Adaptation - Evolution Strategy)を利用する
    - ざっくりいえばEDAもCMA-ESもGAみたいなもの（[EA](https://en.wikipedia.org/wiki/Evolutionary_algorithm)という分類）
  - パラメータ間の関連はGPで表現できないので、別途木構造とグループで管理して、グループごとにGPを定義する。

>>>

### TPE
- GPは`$p(y|x)$`をモデル化していたが、TPEでは以下のモデルをする
`
\[
p(x|y)=\left\{ 
  \begin{array}{ll}
  l(x) & y < y^{*} \\
  g(x) & y \geq y^{*}
  \end{array}
\right.
\]
`
- `$l,g$`は確率密度関数で、データから推定する（[カーネル密度推定](https://ja.wikipedia.org/wiki/%E3%82%AB%E3%83%BC%E3%83%8D%E3%83%AB%E5%AF%86%E5%BA%A6%E6%8E%A8%E5%AE%9A)）
- 特に、`$y$`と`$y^*$`の分位数を`$p(y < y^{*})=\gamma$`と定める。
- TPEの推定のため以下の式を得られる。
`
\[
  \begin{align*}
  EI_{y^*}(x)&=\int^{y^*}_{-\infty} (y^*-y)p(y|x)dy = \int^{y^*}_{-\infty} (y^*-y)\frac{y(x|y)p(y)}{p(x)}dy\\
  &\propto (\gamma+(1-\gamma)\frac{g(x)}{l(x)})^{-1}
  \end{align*}
\]
`

>>>

### 一応計算する
- 微妙に論文に計算ミスがあったのでやってみる
- まず、`$p(y < y^{*}) = \int_{-\infty}^{y^*} p(y)dy$`に注意する。
- `$EI_{y^*}(x)$`の分母の`$p(x)$`を計算する:
`
\[
p(x) = \int_{\mathbb{R}} p(x|y)p(y)dy = \int_{-\infty}^{y^*}+\int_{y^*}^\infty = \gamma l(x)+(1-\gamma)g(x)
\]
`

>>>

### 計算②
- 続いて、`$EI_{y^*}(x)$`の分子を計算する:

`$$\int^{y^*}_{-\infty} (y^*-y)y(x|y)p(y)dy = l(x)\int^{y^*}_{-\infty} (y^*-y)p(y)dy$$`
に対して、`$y^*-y$`をばらし、`$\gamma=\int_{-\infty}^{y^*} p(y)dy$`を使うと
`$$ l(x)\int^{y^*}_{-\infty} (y^*-y)p(y)dy = l(x)\left( y^*\gamma - \int^{y^*}_{-\infty}yp(y)dy\right)$$`

>>>

### 計算③

よって、`$\gamma y^*-\int^{y^*}_{-\infty}yp(y)dy$`は`$x$`に無関係なので

`\[
  \begin{align*}
  EI_{y^*}(x)&=\frac{l(x)}{\gamma l(x)+(1-\gamma)g(x)}\left(\gamma y^*-\int^{y^*}_{-\infty}yp(y)dy\right)\\
  &\propto \left(\gamma+(1-\gamma)\frac{g(x)}{l(x)}\right)^{-1}
  \end{align*}
\]`

が得られる。

---

## まとめ

>>>

- わかったこと
  - チューニングの代表手法であるhyperoptの仕組みを理解することができた
    - 枠組みとなるアルゴリズム(SMBO)を用いて、GP,TPEという確率過程についてベイズ推定する
    - スコアを出力する関数はブラックボックスなので、基準となる関数を代理のものとして近似する
    - ハイパーパラメータをサンプリングしながら、逐次基準となる分布を更新していく
- わからなかったこと
  - SMBOがブラックボックス関数の最適解を得ることの理論を知りたかったが、関連論文を読み切れなかった。
    - SMBOのSとMがブラックボックス関数の構造とどれほど関係があるのかなどの数理的裏付けを知りたい。
  - ベイズ推定がふんわりとしか理解できてないので、具体的な計算をどうするかまで踏み込めなかった。
  
>>>

- 参考文献
  - [Algorithms for Hyper-Parameter Optimization](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf)
  - [Sequential Model-Based Optimization for General Algorithm Configuration](https://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf)
  - Trevor Hastie, 統計的機械学習の基礎 (共立出版)
  - 金森敬文, Pythonで学ぶ統計的機械学習 (オーム社)
  - 佐藤一誠, ノンパラメトリックベイズ (講談社 MLP)
  - 渡辺澄夫, ベイズ統計の理論と方法 (オーム社)
- 参考ページ
  - [hyperoptって何してんの？](https://qiita.com/kenchin110100/items/ac3edb480d789481f134)
  - [Hyperoptとその周辺について](https://www.slideshare.net/hskksk/hyperopt)
  - [Hyperparameter optimization for Neural Networks](http://neupy.com/2016/12/17/hyperparameter_optimization_for_neural_networks.html)
  - [シンプルなベイズ最適化について](https://adtech.cyberagent.io/research/archives/24)