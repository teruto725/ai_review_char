#学習系のクラス
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from models import my_cv
import pandas as pd
#左の払い:0,右の払い:1,下の払い:2,止め3,左へのハネ4,右へのハネ5
LABELS = ["harai","tome"]


#databaseクラス 学習用に画像をストックする必要がある
class DB():
    #ここの連番を工夫する
    static_id=0
    #データの追加
    @staticmethod
    def add(img_rec,img_char,kanji):
        df = DB.get_df()
        idx = None

        #indexを取ってくる
        with open("idx.txt") as f:
            idx = int(f.read())
        
        label = "0"
        img_rec_path = "img_recs/"+str(idx)+".png"
        img_char_path  = "img_chars/"+str(idx)+".png"
        cv2.imwrite(img_rec_path, img_rec)#画像の保存
        cv2.imwrite(img_char_path, img_char)#画像の保存
        #足す
        si = pd.Series([img_rec_path, img_char_path, kanji,label], index=['img_rec_path', 'img_char_path', 'kanji', "label"])
        df = df.append(si, ignore_index=True)
        
        #indexに１足して保存
        with open("idx.txt",mode="w") as f:
            f.write(str(idx+1))
        
        
        df.to_csv("db.csv",index=False)

    #データの一覧取得
    @staticmethod
    def get_df():
        df = pd.read_csv("db.csv")
        return df


#前処理用のクラス
class Preprocesser():
    @staticmethod
    def preprocessing(self,img):
        return img
    


class Recognizer():
    def __init__(self,image):
        pass
    
    def _preprocessing(self):
        pass

    # label の予測を行う
    def predict(self,image):
        label = 0
        score = 0.8
        return label, score


DATA_WIDTH = 20
# pandas データ蓄積用のクラス
class ImageStacker():
    def __init__(self):
        self.data = None #このpandasにデータを蓄積していく

    def add_img(self,img,data):
        pass

    