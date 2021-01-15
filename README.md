# 実行方法
execute.pyから実行できます  
requirements.txtでimportしてください


# 学習について
models/recognize/内で基本行う
img/ : 学習画像のストック
db.csv　: dataset本体、 labelとimgのpathを持つ
models.py : データの追加とか
analyse.ipynb : 認識とかラベリングとか


# メモ
1/14 水の採点コード
1/15 学習セットの作成
1/16 学習を行う
1/17 学習を行う/できれば組み込む


#採点のアイデア
3段階評価
１つのステージをクリアしたら次のすてーじてきなアイデア（もう一回やり直させる感じが出るので微妙）

段階に応じて画像とメッセージが変わる
1. 漢字の骨組みが正しいか（contourの数とかを見る)　2値
 　間違えてればお手本を重ねた画像
　合ってれば花丸を重ねる

2. 形が崩れていないか（バランス） はなまる丸てきなひょうか
     お手本を重ねる
いいところ
-  みぎうえのあがっていくぶぶんがきれいだね 
- 文章で細かく指定する
アドバイス
-  ななめにあがっていっているか確認しよう
- 文章で細かく指定する

3. はねはらい、止めはしっかりできているか 点数出る(はらい,ととめ:中心)(ハネ:根本と先を通る円の中心と直径）
- はね、はらい、とめはんてい部分に青丸と赤丸で正否を文字上にプロット
- 画像上に1とか２とかかく
- 全部丸なら上から花丸

総合スコアはサーバで減点方式で採点して渡す！

# DBの変更
段階は番号（）
それぞれの段階の画像3つ
段階２のメッセージ+いいか悪いかの評価
段階3のハネ払いのメッセージ＋画像中の数字と対応した番号
