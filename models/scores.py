import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
from . import my_cv
from .database import DB #databaseからインポートするs
from .models import *
class Score():
    def __init__(self,img_char, img_exp,kanji):
        self.img_char = img_char#漢字本体の画像
        self.img_exp = img_exp#お手本画像=co
        self.img_overlay = self._img_overlay()#漢字にお手本画像を重ねたもの
        self.kanji = kanji # strの漢字
        self.items_phase1 = list()#第１フェーズのscorelist
        self.items_phase2 = list()#第２フェーズのscorelist
        self.items_phase3 = list()#第３フェーズのscorelist
        self.items_phase4 = list()#第４フェーズのスコアリストs
        
    # お手本をimg?hcarに重ねたものを返す #TODO 実装する
    def _img_overlay(self):
        return self.img_exp

    #はねはらいとめ画像を生成する # TODO 実装する
    def _create_img_phase3(self):
        return self.img_exp

    #アイテム追加よう
    def add_item_phase1(self,message,is_ok):
        item = None
        if is_ok:
            item= ScoreItem(label=0,message=message)
        else:
            item= ScoreItem(label=1,message=message)
        self.items_phase1.append(item)
    
    #アイテム追加用 
    def add_item_phase2(self,message,contours,score,is_good):
        item = None
        if is_good:
            item= ScoreItem(label=2,contours=contours,message=message,score=score)
        else:
            item= ScoreItem(label=3,contours=contours,message=message,score=score)
        self.items_phase2.append(item)

    #phase3アイテム追加用 #is_okははねはらいがあってたかどうか #idxがどこか#centroid(x,y)のポイント座標
    def add_item_phase3(self,message,centroid,score,is_ok):
        idx = len(self.items_phase3)
        item = ScoreItem(label=4,message=message,idx=idx,is_ok=is_ok,centroid = centroid,score=score)#idx どれかを示す
        self.items_phase3.append(item)

    #アイテム追加用 
    def add_item_phase4(self,message,contours,score,is_good):
        item = None
        if is_good:
            item= ScoreItem(label=5,contours=contours,message=message,score=score)
        else:
            item= ScoreItem(label=6,contours=contours,message=message,score=score)
        self.items_phase4.append(item)

    #フェーズ1の画像を返す 255*255*3
    def get_img_phase1(self):
        return self.img_overlay
    
    #フェーズ１の合否を返す bool
    def get_result_phase1(self):
        return True  
    #フェーズ２の画像を返す 255*255*3
    def get_img_phase2(self):
        return self.img_overlay
    #フェーズ２のこうひょうかこめんとを返す
    def get_items_good_phase2(self):
        return ["goood!","Nice!"]
    #フェーズ２のアドバイスを返す
    def get_items_tips_phase2(self):
        return ["２かくめをもっとつよくしっかりかこう","1かくめをもっとつよくしっかりかこう"]    
    #フェーズ２のスコアを返す: int 30点満点
    def get_result_phase2(self):
        return 35
    #フェーズ３の画像を返す # TODO 実装
    def get_img_phase3(self):
        return self._create_img_phase3()
    
    # フェーズ３のitemを返す #TODO 実装
    def get_items_phase3(self):#[idx,message,is_ok]のリストを返す is_ok:True=できてる :Falseできてない
        return [[0,"１かくめのおわりがただしくはねれてますね",True]]

    #フェーズ３のスコアを返す: int 20点満点 # TODO 実装
    def get_result_phase3(self):
        return 10

    #漢字を返す
    def get_kanji(self):
        return self.kanji

    #文字の画像のndarrayを返す(255*255)
    def get_img(self):
        return self.img_char

    #総得点を返す 0~100
    def get_all_score_point(self):
        return 100

    # debug 用
    def print_debug(self):
        print("====="+self.kanji+"======")
        my_cv.display_color(self.img_char)
        print("phase1")
        for item in self.items_phase1:
            print(item.message)
        print("phase2")
        for item in self.items_phase2:
            print(item.message)
        print("phase3")
        for item in self.items_phase3:
            print(item.message)
        print("phase4")
        for item in self.items_phase4:
            print(item.message)
        



#評価項目クラス label 0~2  message: 内容 score単体スコア, contour: contour クラス
class ScoreItem():
    def __init__(self,label,contours=None,message=None,score=None,idx=None,is_ok=None,centroid=None):
        self.label = label# 0:構造に関するエラー 1:形が崩れていないかの部分で高評価 　2:形が崩れていないかの部分で低評価 3:止めはね払いの部分で綺麗かどうか       
        self.message = message # エラーメッセージ 
        self.contours = contours# 領域
        self.score = score # スコアポイント

    def __str__(self):
        return self.message

    def get_label(self): # 0:褒めている,1:できれば直したい(斜め向いてるとか),2:絶対に治そう(はらってないとか)
        return self.label
    def get_message(self):#メッセージ(内容)を返す
        return self.message
    def get_centroid(self): #重心を返す 領域複数の場合は平均
        x_stock = list()
        y_stock = list()
        for contour in self.contours:
            x_stock.append(contour.centroid[0])
            y_stock.append(contour.centroid[1])
        return [ int(np.average(x_stock)),int(np.average(y_stock)) ]
    def get_contour(self): #領域を返す
        return ([contour.cnt for contour in self.contours])
