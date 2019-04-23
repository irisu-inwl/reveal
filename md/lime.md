秒速で理解する信頼できるAI  
LIMEでわかる機械学習

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
Google Colabという無料で機械学習環境を作れるサービスをgoogleが出してるのでそれを使います！
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

>>> 

# でゎない

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
- SP-LIME: 機械学習した結果自体が正しいか大域的に説明する手法

>>>

### アルゴリズムの概要(LIME)
予測するデータの周辺と、そのデータが影響するかしないかのバイナリベクトルをサンプリングし、<br>
Logistic回帰することで予測データの付近の決定境界を線形モデルで近似する。

>>>

### 用語の準備
- x
- `$G$`: 解釈可能モデルの集合, 解釈可能モデルは`$g\in G: \mathbb{R}^{d^\prime}\to\mathbb{R}$`である。


