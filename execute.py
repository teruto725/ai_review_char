import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
from pprint import pprint 
import sys
from .models.models import *
#自作モジュール
from .models import my_cv

#TODO asyncで書かないとダメそう


#実行　cv2型のndarrayを受け取る
#戻り値　bool:実行できたかどうか, str:エラーメッセージ
def execute(img_ori):
    img_paper = None
    try:
        img_paper = my_cv.cutting_paper(img_ori)
    except:
        return False, "cutting_paper"
    paper = None
    try:
        paper = Paper(img_paper,"Sho","Mizu")#TODO この文字は後で認識すること
    except:
        
        return False, "creating_paper"
    try:
        score1s, score2s = paper.get_scores(False) #trueなら画像をストックする
        return score1s, score2s
    except:
        return False, "getting_score"
    


#デバッグ用のメソッド(エラーで止まる)
def execute_debug(img_ori):
    img_paper = my_cv.cutting_paper(img_ori)
    print("done_cutting_paper")
    paper = Paper(img_paper,"Sho","Mizu")#TODO この文字は後で認識すること
    print("done paper")
    score1s, score2s = paper.get_scores(False)
    print("done score")
    return score1s, score2s

#学習用に矩形領域を切り出してストックする
def stock_recs(img_ori):
    pass

#デバックで実行する用
if __name__=="__main__":
    img_paper = cv2.imread('./sample_images/work8.png')#debug用
    score1s, score2s = execute_debug(img_paper)
    for s in score1s:
        s.print_debug()