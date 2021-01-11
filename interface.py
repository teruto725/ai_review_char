import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
from pprint import pprint 
from model import Factory, Paper,  Char,  Mizu,  Sho, Score,Contour, Recognizer
#自作モジュール
import my_cv

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
        scores1, scores2 = paper.get_scores()
        return scores1, scores2
    except:
        return False, "getting_score"
    


#デバッグ用のメソッド(エラーで止まる)
def execute_debug(img_ori):
    img_paper = my_cv.cutting_paper(img_ori)
    paper = Paper(img_paper,"Sho","Mizu")#TODO この文字は後で認識すること
    scores1, scores2 = paper.get_scores()
    return scores1, scores2