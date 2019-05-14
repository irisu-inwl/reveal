## 秒速で理解する信頼できるAI  
## LIMEでわかる機械学習

---

### 目次
- 機械学習とは？
  - 教師あり学習
  - 教師なし学習
- 信頼できるAIとは？
  - LIME
- LIMEのアルゴリズムを知る

---

## 第1章
## 機械学習とは？

>>>

機械学習には以下の二つがある。
- データの特徴と正解のペアを学習する教師あり学習
- データの特徴のみから結果を出す教師なし学習

>>>

# といってもわからないので例
Google Colaboratoryという無料で機械学習環境を作れるサービスをgoogleが出してるのでそれを使います！

https://drive.google.com/file/d/1d_P68Z9hsLSoKT9RptaBr5nZHYXIgPXy/view?usp=sharing

>>>

例1（教師あり学習 分類）.  
<div style="font-size:24pt">
ワインに含まれる化学物質（例えば、アルコール度数、リンゴ酸、pH, マグネシウムの量, etc...)からワインの等級を当てることを考える。  

![data](img/wine_data.PNG) ![画像](img/wine_class.PNG)

ここではワインの化学物質を特徴、等級を正解として、学習し、新しく入力されるデータの予測を行う。  
教師あり学習では、特徴を、特徴の数`$m$`を次元とするベクトル`$\mathbb{R}^m$`から正解の集合`$X$`を返す関数`$h: \mathbb{R}^m\to X$`を求めることと考えることが出来る。
</div>

>>>

### コード例

```
from sklearn.datasets import load_boston, load_wine
wine_data = load_wine()

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 正解の比率を変えずにデータをX_train:X_test = 4:1に分割（訓練するデータとテストするデータに分割）
X_train, X_test, y_train, y_test = train_test_split(wine_data.data, wine_data.target, test_size=0.2, stratify = wine_data.target)
# 学習
clf = RandomForestClassifier().fit(X_train, y_train)
# 予測
predict_data = clf.predict(X_test)
score = accuracy_score(predict_data, y_test)
print(f'正解率: {score}')
```

>>>

例2（教師あり学習 回帰）.  
<div style="font-size:24pt">
地域の情報（例えば、犯罪発生数, 住居区画の割合, 住居の平均部屋数 etc...)から地域の住宅価格を当てることを考えます。  

![data](img/house_data.PNG) ![画像](img/house_class.PNG)

とくに、予測するときの正解データの集合が**離散値**のとき**分類**、（今回みたいに）**連続値**のとき**回帰**といいます。
</div>

>>>

### コード例

```
from sklearn.datasets import load_boston, load_wine
house_data = load_boston()
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# 正解の比率を変えずにデータをX_train:X_test = 4:1に分割
X_train, X_test, y_train, y_test = train_test_split(house_data.data, house_data.target, test_size=0.2)
# 学習
regressor = RandomForestRegressor().fit(X_train, y_train)
# 予測
predict_data = regressor.predict(X_test)
score = r2_score(predict_data, y_test)
print(f'誤差: {score}')
```

>>>

例3 (教師なし学習 クラスタリング).  
<div style="font-size:24pt">
例1のワインデータの特徴をベクトルの距離で近いもの同士まとめてみる。  
同じデータをまとめる処理を**クラスタリング**といいます。  
※ 今回の発表では関係なし
</div>

>>>

### コード例

```
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans

cls = KMeans(n_clusters=3).fit_predict(wine_data.data)
# 本物の正解とどれくらい一致してるか？
score = adjusted_rand_score(cls, wine_data.target)
print(f'正解率: {score}')
```

>>>

# ここで紹介した以外にもいろいろなアルゴリズムがある！

参考: https://scikit-learn.org/stable/user_guide.html

---

## 第2章
## 信頼できるAIとは？

>>>

<div style="font-size:24pt; text-align: left">
先ほどの例1,例2では、学習した結果にデータを渡すと予測値が<br>
返ってきたが、その予測は信頼できるのだろうか？

どれだけ精度が高くても、その判断を信頼できる根拠がなければ、その結果を人間が使うことは難しい。

![画像](img/distopia.png)
</div>
<div style="font-size:18pt; text-align: left">
マイノリティリポートみたいになる
</div>

>>>

## 機械学習の結果を説明する
## そんな技術あるの？

>>>

## 実はあるんです！

>>>

### LIME

- 初出論文: "Why Should I Trust You?": Explaining the Predictions of Any Classifier
  https://www.kdd.org/kdd2016/subtopic/view/why-should-i-trust-you-explaining-the-predictions-of-any-classifier
- 学会: SIGKDD 2016
- github: https://github.com/marcotcr/lime
- 読むモチベーション: 実際に仕事で使ってるが、中を詳しく知らないので読んだ。


>>>

### LIME

- どんな手法？ : 予測に対してどの特徴が影響を与えてるか見せることで機械学習を説明する。
  - 例えば、githubに載っている例では、ニュースをキリスト教についてか宗教に関係ないかを判定する機械学習について、
  NNTP, Hostという言葉が宗教に関係ない分類に影響していることを示している。

![画像](img/twoclass.png)

>>>

### とりあえずやってみよう！　コード例 (例1の機械学習結果を説明する)

```
from lime.lime_tabular import LimeTabularExplainer

train_data = wine_data.data
class_names = wine_data.target_names
feature_names = wine_data.feature_names

explainer = LimeTabularExplainer(train_data ,class_names=class_names, 
                                feature_names = feature_names, kernel_width=3, verbose=False)

# 正解の比率を変えずにデータをX_train:X_test = 4:1に分割
X_train, X_test, y_train, y_test = train_test_split(wine_data.data, wine_data.target, test_size=0.2, stratify = wine_data.target)
# 学習
clf = RandomForestClassifier().fit(X_train, y_train)

explain_data = X_test[0]
explain_result = explainer.explain_instance(explain_data, clf.predict_proba, num_features=6, top_labels=2)

explain_result.show_in_notebook(show_table=True, show_all=True)
```

>>>

### 結果

![画像](img/lime_result.PNG)

>>>

# できた！
# 完！

<span style="display: inline-block;" class="fragment fade-right">ではない…</span>

---

## 第3章
## LIMEのアルゴリズムを知る

>>>

<div style="font-size:24pt">
gitのコード動いた！　これで実用できるとはならない。<br>
中の仕組みを知らないで「機械学習の説明できるようになった！　どういう原理で説明しているかわかってないけど…」で使うのは危険。<br>
</div>

>>>

論文で提示されているアルゴリズムは2つ
- LIME: 機械学習で予測した値が正しいかを局所的に説明する手法
- SP-LIME: 機械学習したモデルが正しいか大域的に説明する手法

>>>

### アルゴリズムの概要(LIME)
予測するデータの周辺と、そのデータが影響するかしないかのバイナリベクトルをサンプリングし、<br>
線形回帰することで予測データの付近の決定境界を線形モデルで近似する。

>>>

### 用語の準備①
- `$f(x):\mathbb{R}^d\to\mathbb{R}$` をデータ`$x$`があるクラスに入ってる確率
- データのベクトル`$x\in \mathbb{R}^d$`に対して人が解釈しやすい空間へと変換した点`$x^\prime \in \{0,1\}^{d^\prime}$`を考える
- 例えば、アルコール度数(`$\in\mathbb{R}$`)を、`ある範囲からある範囲に入っていれば1となる`バイナリベクトルに変換する。
<div style="font-size:20pt">
<table border=0><tr><td>
<table>
<tr>
<th>アルコール度数</th>
</tr>
<tr>
<td>14.38</td>
</tr>
</table>
</td>
<td valign="top">→</td>
<td valign="top">
<table>
<tr>
<th>x <= 12.36</th>
<th>12.36 < x <= 13.05</th>
<th>13.05 < x <= 13.68</th>
<th>13.68 < x</th>
</tr>
<tr>
<td>0</td>
<td>0</td>
<td>0</td>
<td>1</td>
</tr>
</table></td></tr></table>
</div>

- `$g: \mathbb{R}^{d^\prime}\to\mathbb{R}$`: 解釈可能モデル, `$G$`: 解釈可能モデルの集合

>>>

### 用語の準備②
- `$\pi_x(z)$`:`$x$`からどれくらい離れているかの尺度。`$z$`が`$x$`に近ければ小さい値を取る。
- `$\Omega(g)$`: 解釈するためのモデル`$g$`の複雑度
- 例えば、線形モデル`$g(x^\prime)=w^T \cdot x^{\prime}$`だったら係数の非ゼロ成分数
- `$L(f,g,\pi_x)$`: 学習モデル`$f$`を説明する`$g$`が信頼できない尺度
- `$Z$`: `$x,x^\prime$`の周辺からサンプリングされた点の集合
- `$f$`を説明するモデルを求めるのは以下の式によって求められる。

`$$\xi(x)=argmin_{g\in G} L(f,g,\pi_x))+\Omega(g)$$`

- この方法によってどんな学習方法でも予測の説明ができる。

>>>

### LIMEの仮定
- LIMEでは前述の定義に具体的な対象を代入してアルゴリズムを考えてます。
- `$g$`は線形モデルのみを考えてる。つまり、`$g(x^\prime)=w^T \cdot x^{\prime}$`として考えてる。
- 離れている尺度は`$\pi_x(z)=\exp(-distance(x,z)^2/\sigma^2)$`
- 信頼の損失は`$L(f,g,\pi_x)=\sum_{z,z^\prime\in Z} \pi_x(z)(f(z)-g(z^\prime))^2$`
- `$g$`を求めることは即ちパラメータ`w`を求めることである。
- `$\xi$`を求める式を考えてみると、`$L$`が最小二乗誤差、`$\Omega$`の部分は線形モデルの罰則項と考えることで、Lassoによって推定できることがわかる。
- なんかgithubのコード見るとRidgeで回帰してるけど・・・

>>>

### LIMEのアルゴリズム②

<div style="text-align:center;">
<img src="img/lime_algorithm.PNG" height="50%" width="50%">
</div>

- 予測する点`$x$`（と解釈可能ベクトル`$x^\prime$`）の近傍をサンプリングして、`$f$`の確率に近くなるように`$g$`を決定する。
- どの解釈が機械学習モデルに影響を与えてるかを`$w$`の要素で決定している。
- 

>>>

### SP-LIMEのアルゴリズム概要
- データの集合`$X$`の要素全てに対してLIMEを行う。
- `$X$`の中から、よく使われる特徴(LIMEのアウトプット`$w$`が大きいもの)を持っているデータを`$B$`個抜き出す。
  （実装では`$X$`からサンプリングした中から抜き出している。）

<div style="text-align:center;">
<img src="img/splime_example.PNG">
</div>

- 特徴を網羅するようなデータを抜き出し、それらの説明を見せることで、学習モデル自体がどんなデータでどんな判断をするのか説明する。

>>>

### SP-LIMEのアルゴリズム

<div style="text-align:center;">
<img src="img/splime_algorithm.PNG" height="50%" width="50%">
</div>

- `$W=(W_{ij})$`は`$i$`番目のデータについての説明した結果で得られた`$w$`の`$j$`番目の要素
- 本当は`$argmax_{V,|V|\leq B} c(V,W,I)$`を求めたいがNP-hardになる。
  しかし、`$c(V,\cdot,\cdot)$`が`$V$`に対して劣モジュラになるため、貪欲法で近似できる。

>>>

### 例

```
from lime import submodular_pick

clf = RandomForestClassifier().fit(wine_data.data, wine_data.target)
sp_obj = submodular_pick.SubmodularPick(explainer, wine_data.data, clf.predict_proba, sample_size=50, num_features=13, num_exps_desired=5)

[exp.as_pyplot_figure() for exp in sp_obj.sp_explanations];
```

---

## まとめ

>>>

- LIMEのアルゴリズムの解説を行った。
  - ある学習モデルの与えられたデータに対する判断の説明を、データの近傍をサンプリングして線形モデルで学習することで実現している。
- SP-LIMEのアルゴリズムの解説を行った。
  - その学習モデル自体の説明を、各データの説明結果を特徴を網羅するようにデータを抜き出すことで実現している。
- reveal.js+github pagesが良い感じだった
- google colabも良い感じ

>>>

## 参考文献
- Marco Tulio Ribeiro, "Why Should I Trust You?": Explaining the Predictions of Any Classifier
  - https://www.kdd.org/kdd2016/subtopic/view/why-should-i-trust-you-explaining-the-predictions-of-any-classifier
- 平井有三, はじめてのパターン認識 (森北出版)
- Trevor Hastie, 統計的機械学習の基礎(共立出版)
- 金森敬文, Pythonで学ぶ統計的機械学習