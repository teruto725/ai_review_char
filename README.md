# 実行方法
execute.pyから実行できます  
requirements.txtでimportしてください

pathの調整

# 構成
- img_chars/ : 学習・推論用画像のストック用ディレクトリ(文字全体画像)
- img_exps/ : お手本画像をストックするためのディレクトリ
- img_recs/ : 学習・推論用画像のストック用ディレクトリ(矩形画像)<= こっちが学習用
- img_scores/ : 採点用の赤ペン画像
- model/ : スクリプト
     - database.py : 学習・推論用画像の擬似的なデータベース(保存・取り出しを行う)
     - models.py : 採点コードのメイン部分
     - my_cv.py : 採点時やデバッグ時に用いる汎用コード
     - path.py : path管理ファイル(デバッグ時とサーバ上でpathが異なるため)
     - recognizer.py : 学習・推論用スクリプト
- sample_images/ : サンプル画像が入っているディレクトリ
     - train/ : 学習・推論用画像が入ったディレクトリ
     - trash/ : ゴミ
     - debug/ : デバッグ用画像
- tmp/ : tempディレクトリ
- analyse2.ipynb : デバッグ用ファイル
- cnn.net : ネットワークファイル
- execute.py : 実行用ファイル
- recoginze.ipynb : 予測推論を行うコード 
