#一文字に対してのスコア
class Score():
    def __init__(self):
    #listでのインデックスを返す[0~3]
    def get_idx(self):
        pass
    
    #文字の画像のndarrayを返す(255*255)
    def get_img(self):
        return 
    #文字のお手本画像のndarrayを返す(255*255)
    def get_img_exp(self):
        pass
    #Score Itemリストを返す
    def get_items(self):
        pass
    #全体の総合スコアを返す(0~100)
    def get_score(self):
        pass
    

class ScoreItem():
    def __init__(self):
    # 0:褒めている,1:できれば直したい(斜め向いてるとか),2:絶対に治そう(はらってないとか)
    def get_label(self):
    #メッセージ(内容)を返す
    def get_message(self):
    #重心を返す
    def get_centroid(self):
    #領域を返す(デバッグがしんどいのでとりあえずNone)    
    def get_contour(self):


def evaluate():
    #水のクラス
    mizu_scores = list()
    #小のクラス
    shou_scores = list()
    return 

