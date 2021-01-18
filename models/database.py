#学習系のクラス
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from .models import my_cv
import pandas as pd
#払い0 止め1
LABELS = ["harai","tome"]


#databaseクラス 学習用に画像をストックする必要がある
class DB():
    DFPATH = "db.csv"
    IDXPATH = "idx.txt"
    #ここの連番を工夫する
    static_id=0
    #データの追加
    @staticmethod
    def add(img_rec,img_char,kanji):
        df = DB.get_df()
        idx = None

        #indexを取ってくる
        with open(DB.IDXPATH) as f:
            idx = int(f.read())
        
        label = "0"
        img_rec_path = "img_recs/"+str(idx)+".png"
        img_char_path  = "img_chars/"+str(idx)+".png"
        cv2.imwrite(img_rec_path, img_rec)#画像の保存
        cv2.imwrite(img_char_path, img_char)#画像の保存
        #足す
        si = pd.Series([img_rec_path, img_char_path, kanji,label], index=['img_rec_path', 'img_char_path', 'kanji', "label"])
        df = df.append(si, ignore_index=True)
        #my_cv,display_gray(img_rec)
        #indexに１足して保存
        with open(DB.IDXPATH,mode="w") as f:
            f.write(str(idx+1))
        
        
        df.to_csv(DB.DFPATH,index=False)

    #データの一覧取得
    @staticmethod
    def get_df():
        df = pd.read_csv("db.csv")
        return df
    #データの保存
    @staticmethod    
    def save(df):
        df.to_csv(DB.DFPATH,index=False)

