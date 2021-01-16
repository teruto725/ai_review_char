import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
from . import my_cv
from .database import DB #databaseからインポートするs

class Factory():
    def create_char(self,char_name,char):#charの名前とcharインスタンスを渡す
        #char.display()
        if char_name == "Mizu":
            char = Mizu(char.points_ori,char.img_paper)
        elif char_name == "Sho":
            char= Sho(char.points_ori,char.img_paper)
        return char


class Paper():
    def __init__(self,img_paper,char1_name,char2_name):#紙の領域が切り取られたimgを受け取る
        self.img_paper = img_paper 
        self.char1_name = char1_name#紙で言う右の文字(小)
        self.char2_name = char2_name#上で言う左の文字(水)
        self.chars = self._find_chars(img_paper)#文字の集合16個で固定
        self.char1s = list() # 採点する文字１の配列
        self.char2s = list() # 採点する文字２の配列
        self.char1_exp = None # 文字１の見本
        self.char2_exp = None # 文字２の見本
        self._labeling()#ラベリングして上に振り分ける
        self.char1s = self._sort_chars(self.char1s)#chars配列をソートする
        self.char2s = self._sort_chars(self.char2s)#chars配列をソートする
        _ = list(map(lambda char1:char1.set_img_exp(self.char1_exp.img_char),self.char1s))#例題画像をセットする
        _ = list(map(lambda char2:char2.set_img_exp(self.char2_exp.img_char),self.char2s))#例題画像をセットする
        
    #label付けを行う
    def _labeling(self):
        fty = Factory()
        for char in self.chars:
            #char.display()
            #print(char.get_lu_point())
            if 1200<char.get_lu_point()[0]<1700:
                #print("char1_exp")
                self.char1_exp = char
            elif 900<char.get_lu_point()[0]<1100:
                #print("char2_exp")
                self.char2_exp = char
            elif 400<char.get_lu_point()[0]<600:
                if char.get_lu_point()[1]< 650:
                    pass #なぞりの部分は評価しない
                else:
                    #print("char1")
                    self.char1s.append(fty.create_char(self.char1_name,char))
            elif 100<char.get_lu_point()[0]<200:
                if char.get_lu_point()[1]<650:
                    pass #なぞりの部分は評価しない
                else:
                    #print("char2")
                    self.char2s.append(fty.create_char(self.char2_name,char))
            else:
                print("labeling_error")
                #char.display()
                print(char.get_lu_point())

    #chars配列のソート #y座標が小さい順にソートする
    def _sort_chars(self,chars):
        chars = np.array(chars)#事前にarrayにしとかんとバグる
        char_ys = [char.get_lu_point()[1] for char in chars]
        #print(char_ys)
        sorted_idx = np.argsort(char_ys)
        #print(sorted_idx)
        char_sorted = chars[sorted_idx]
        return char_sorted.tolist()  


    #文字領域を見つけてそれらを格納したcharsを生成する #60000 70000      TODO 枠線が残っているとエラーが出るっぽい    
    def _find_chars(self,img, lower_thre=59000, upper_thre = 70000,sq_num = 16):#img
        gray0 = np.zeros(img.shape[:2], dtype=np.uint8)
        rows, cols, _channels = map(int, img.shape)        # down-scale and upscale the image to filter out the noise
        pyr = cv2.pyrDown(img, dstsize=(cols//2, rows//2))
        timg = cv2.pyrUp(pyr, dstsize=(cols, rows))
        cv2.mixChannels([timg], [gray0], (0, 0))# 画像のBGRの色平面で正方形を見つける ( 0で固定した)
        for l in range(0, 1):# いくつかのしきい値レベルを試す
            if l == 0:
                # Cannyを適用
                # スライダーから上限しきい値を取得し、下限を0に設定します
                # （これによりエッジが強制的にマージ）
                gray = cv2.Canny(gray0,50, 5)
                #Canny出力を拡張して、エッジセグメント間の潜在的な穴を削除します
                gray = cv2.dilate(gray, None)
            else:
                # apply threshold if l!=0:
                gray[gray0 >= (l+1)*255/5] = 0
                gray[gray0 < (l+1)*255/5] = 255
            contours, _ = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            #print(type(contours))
            #print(np.shape(contours))
            #my_cv.display_con(img,contours)#debug
            #print(np.shape(contours))#ここまではok
            chars = list()
            for i, cnt in enumerate(contours):
                arclen = cv2.arcLength(cnt, True)# 輪郭の周囲を取得
                approx = cv2.approxPolyDP(cnt, arclen*0.05, True)# 輪郭の近似
                #my_cv.display_approx(img,approx)
                area = abs(cv2.contourArea(approx))# 面積
                #長方形の輪郭は、近似後に4つの角をもつ、
                #比較的広い領域
                #（ノイズの多い輪郭をフィルターで除去するため）
                #凸性(isContourConvex)になります。
                if approx.shape[0] == 4 and upper_thre > area > lower_thre and cv2.isContourConvex(approx) :
                    maxCosine = 0
                    #my_cv.display_con(img,[cnt])
                    
                    for j in range(2, 5):   # ジョイントエッジ間の角度の最大コサインを見つけます
                        cosine = abs(my_cv.angle(approx[j%4], approx[j-2], approx[j-1]))
                        maxCosine = max(maxCosine, cosine)
                    if maxCosine < 0.3 :# すべての角度の余弦定理が小さい場合（すべての角度が約90度）、
                        char = Char(approx,img)
                        #print("a")
                        for ch in chars:
                            if ch.is_same(char):
                                break
                        else:
                            #char.display()
                            #print(area)
                            chars.append(char)
            print(len(chars))
            if sq_num+1>=len(chars)>=sq_num-1:
                return chars



    #スコア一覧を返す(ここでスコアを作成する)
    def get_scores(self,is_stock_rec):
        score1s = list(map(lambda char:char.scoreing(is_stock_rec),self.char1s))#Score配列
        score2s = list(map(lambda char:char.scoreing(is_stock_rec),self.char2s))#Score配列
        
        return score1s, score2s

class Char():
    THRESH_NUM = 200 #この値より大きい画素を白にする
    def __init__(self, points_ori,img_paper):# np.shape(points) = (4,1,2)
        #self.points_ori = points_ori
        self.points_ori = my_cv.arrange_approx_points(points_ori) #is_sameメソッドで使用する
        self.img_paper = img_paper
        self.img_char = self._fit_image(img_paper,points_ori)# 255*255の正方形に変換した画像
        
        self.img_thresh = self._get_img_thresh()
        self.basic_contours = self._get_basic_contours()#基本的な領域
        self.img_exp = None #見本カラー画像
        self.kanji = None #漢字

        #my_cv.display_color(self.img_char)
        #self.img_fltr = self._filter_image()#フィルターがかけられた2値の画像
        #my_cv.display_gray(self.img_thresh) #debug
    #見本をセットする
    def set_img_exp(self,img_exp):
        #print("a")
        #my_cv.display_color(img_exp)
        self.img_exp = img_exp

    #四角に整形して結果を保持する
    def _fit_image(self,img,points_ori,x=255,y=255):
        #epsilon = 0.1*cv2.arcLength(points_ori,True)#周辺長の取得
        #paper_corners= cv2.approxPolyDP(points_ori,epsilon,True)#紙の頂点座標
        fix_con = np.array([[[0,0]],[[x,0]],[[x,y]],[[0,y]]], dtype="int32")#整形後のサイズ
        trans_arr = cv2.getPerspectiveTransform(np.float32(points_ori),np.float32(fix_con))#変換行列の生成
        return  cv2.warpPerspective(img,trans_arr,(x,y))#変換

    #２値化画像 #hsvからの大津の２値化
    def _get_img_thresh(self):
        result = np.copy(self.img_char)
        hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV)
        _, img_thresh = cv2.threshold(hsv[:,:,2], 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        img_thresh = cv2.bitwise_not(img_thresh)
        return img_thresh

    # 単純な二値化imgからcontours を生成する
    def _get_basic_contours(self):
        cnts, hie = cv2.findContours(self.img_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        #print(np.shape(cnts))
        for i in reversed(range(len(cnts))):
            if cv2.contourArea(cnts[i]) >= (250*250) or cv2.contourArea(cnts[i]) <= 30:#領域がほぼ全体or小さすぎるなら削除する
                hie = np.delete(hie,i)
                cnts = np.delete(cnts,i)
        cnts = np.array(cnts)
        ctrs_x = [my_cv.get_centroid(cnt)[0] for cnt in cnts] #重心座標のリスト
        sorted_idx = np.argsort(ctrs_x)#sort
        #print(sorted_idx)#debug
        #print(np.shape(cnts))#debug
        #print(type(sorted_idx[-1]))#debug

        cnts_sorted = cnts[sorted_idx]
        #hie_sorted = hie[sorted_idx]
        return [Contour(cnts_sorted[i],self.img_char,hie) for i in range(len(cnts_sorted))]

    #左上の座標が同じかどうか
    def is_same(self, other):
        op = other.points_ori[0][0]
        mp = self.points_ori[0][0]
        if abs(op[0] - mp[0]) + abs(op[1]-mp[1]) <=50:
            return True
        return False    
                               
    #左上のアドレスを(2)で返す
    def get_lu_point(self):
        return self.points_ori[0][0]
    
    #表示する
    def display(self):
        cv2.imwrite("temp.png", self.img_char)
        plt.imshow(plt.imread("temp.png"))
        plt.axis('off')
        plt.show()
        
"""
    #　ポイントを出力する   
    def get_feature_points(self):
        thre_img = np.copy(self.img_thresh)
        contours, _ = cv2.findContours(thre_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        rate_parms = [800]
        for n, rate in enumerate(rate_parms):
            resimg = self.img_char.copy() // 2 + 128
            for i, cnt in enumerate(contours):
                # 輪郭の周囲に比例する精度で輪郭を近似する
                size = cv2.contourArea(cnt)
                if size == 0:
                    continue
                #1.2をかけることで大きい図形に対してのバイアスを強くしている
                approx = cv2.approxPolyDP(cnt, rate/size*1.5, True) #第２引数が小さければ細かい近似大きければ大雑把な近似 1.2をかけること
                print(int(size))
                print(rate/size)
                if size>(255*255-5000):
                    continue
                cv2.polylines(resimg, [approx.reshape(-1,2)], True, 
                            (0,0,255), thickness=1, lineType=cv2.LINE_8)
                for app in approx:
                    cv2.circle(resimg, (app[0][0],app[0][1]), 2, (0, 255, 0), thickness=-1)
            #my_cv.display_color(resimg)
"""
    


#文字１つに対するスコアクラス。
#items_phase1のメッセージは以下の３つ
# かんじのかたちがへんだよ、Nかくめのかたちがへんだよ、



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

    
#領域クラス
class Contour():
    def __init__(self,cnt,img_char,hie):
        self.cnt = cnt # 領域アドレス列
        self.img_char = img_char # 文字の画像
        self.hie = hie #階層情報(今はそのまま渡しているので使えない)
        self.img_thresh = self._create_thresh()
        self.img_masked = self._create_masked()
        self._define_features()
    #２値画像を生成する
    def _create_thresh(self):
        img_gray = cv2.cvtColor(self.img_char, cv2.COLOR_BGR2GRAY)
        img_black = np.zeros(np.shape(img_gray)).astype(img_gray.dtype)
        img_black = cv2.drawContours(img_black,[self.cnt],0,255,-1)
        return img_black

    #特徴量を生成する
    def _define_features(self):
        self.area = cv2.contourArea(self.cnt)#面積
        self.length = cv2.arcLength(self.cnt,True)# 周囲長
        self.centroid = my_cv.get_centroid(self.cnt)#重心座標
        #print(np.shape(self.cnt))#debug
        self.min_x_point = None#一番左の点[1,2]
        self.min_y_point = None#一番下の点
        self.max_x_point = None#一番右の点
        self.max_y_point = None#一番上の点
        self.min_x_index = 0#一番左の点(X)
        self.min_y_index = 0#一番左の点(X)
        self.max_x_index = 0#一番左の点(X) 
        self.max_y_index = 0#一番左の点(X)
        self.right_bottom_point = None#右下の点
        self.right_bottom_index = 0#右下のidx
        self.right_top_point = None #右上の点
        self.right_top_index = 0 #右上のidx
        self.left_bottom_point = None#左下の点
        self.left_bottom_index = 0#左下のidx
        self.left_top_point = None #左上の点
        self.left_top_index = 0 #左上のidx
        for i,point in enumerate(self.cnt):
            point = point[0]
            if self.min_x_point is None:
                self.min_x_point = point
                self.min_y_point = point
                self.max_x_point = point
                self.max_y_point = point
                self.right_bottom_point = point
                self.right_top_point = point
                self.left_bottom_point = point
                self.left_top_point = point
                continue
            if self.min_x_point[0] > point[0]:
                self.min_x_point = point
                self.min_x_index = i
            if self.min_y_point[1] > point[1]:
                self.min_y_point = point
                self.min_y_index = i
            if self.max_x_point[0] < point[0]:
                self.max_x_point = point
                self.max_x_index = i
            if self.max_y_point[1] < point[1]:
                self.max_y_point = point
                self.max_y_index = i
            if self.right_bottom_point[0] + self.right_bottom_point[1] < point[0]+point[1] :
                self.right_bottom_point = point#右下
            if self.right_top_point[0] - self.right_top_point[1] < point[0]-point[1]:
                self.right_top_point = point#右上
            if self.left_bottom_point[1] - self.left_bottom_point[0]  < point[1]-point[0]:
                self.left_bottom_point = point#左下
            if self.left_top_point[0] + self.left_top_point[1] > point[0]+point[1]:
                self.left_top_point = point #左上
        self.width = abs(self.max_x_point[0] - self.min_x_point[0]) #幅
        self.height = abs(self.max_y_point[1] - self.min_y_point[1]) #高さ

    #右上から左下にかけて引かれている線でx軸方面に飛び出していないかmin_x_point max_y_point 
    def is_migiue_to_hidarishita(self):
        #右上が点１つ 左下が点1つ 
        return my_cv.distance(self.max_x_point,self.min_y_point) <20 and my_cv.distance(self.min_x_point,self.max_y_point) <20

    #二値化画像をプロット
    def display_thresh(self):
        my_cv.display_gray(self.img_thresh)

    #二値化画像をプロット
    def display(self):
        result = np.copy(self.img_char)
        result = cv2.drawContours(result,[self.cnt],0,(255,0,255),1)
        my_cv.display_gray(result)

    #indexからpointを返す
    def get_point(self,i):
        
        return self.cnt[(i%len(self.cnt))][0]

    def get_idx(self,i):
        return int(i%len(self.cnt))

    #近似線を出力するap_num:ほしい特徴点の数 loopnum:何周するか、init_num:loopの初期値, alphaループ枚の減算値
    #True なら成功　Falseなら失敗
    def get_approx(self, ap_num, loop_num, init_num, alpha):
        #self.display_thresh()
        img_fltr =  my_cv.mor_clear_filter(self.img_thresh)#clearフィルターで文字をなめらかにする
        cnts,_ = cv2.findContours(img_fltr,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#領域抽出する 
        if len(cnts)  !=  1 :
            print("APPROXERROR")
            return False, None #フィルタによって文字が分裂してしまうと発生する恐れがある
        #1.2をかけることで大きい図形に対してのバイアスを強くしている
        for i in range(loop_num):
            #print(init_num-i*alpha)
            approx = cv2.approxPolyDP(cnts[0], init_num-i*alpha, True) #第２引数が小さければ細かい近似大きければ大雑把な近似 1.2をかけること
            #dprint(np.shape(approx))
            #左
            if ap_num+2 >= len(approx) >= ap_num:  
                return True, approx
        return False, None

    def _create_masked(self):
        img = np.copy(self.img_char)
        img_thre_color = np.stack([self.img_thresh,self.img_thresh,self.img_thresh],axis = 2)
        dst = cv2.bitwise_and(img, img_thre_color)
        return dst


    #左下の矩形領域を取り出す
    def get_left_bottom_rec(self):
        left_bottom_point = [self.min_x_point[0],self.max_y_point[1]]#左下のポイント
        width = 20
        left_x = left_bottom_point[0]-4
        right_x = left_x+width
        bottom_y = left_bottom_point[1]+4
        top_y = bottom_y-width
        img_slice = self.img_masked[top_y:bottom_y,left_x:right_x,:]#トリミング
        return img_slice
    
    #右下の矩形領域を取り出す
    def get_right_bottom_rec(self):
        right_bottom_point = [self.max_x_point[0],self.max_y_point[1]]#右下のポイント
        width = 20
        right_x = right_bottom_point[0]+2
        left_x = right_x-width
        bottom_y = right_bottom_point[1]+2
        top_y = bottom_y-width
        img_slice = self.img_masked[top_y:bottom_y,left_x:right_x,:]#トリミング
        return img_slice


    #少なくとも上の値はmin_x_point[1]-10までなので絶対に入る
    def get_hane_left_rec(self,hidari_point,nemoto_point):
        width = 20
        left_x = hidari_point[0]-2
        right_x = nemoto_point[0]-2
        top_y = max(hidari_point[1],nemoto_point[1])-10
        bottom_y = top_y + width
        img_slice = self.img_masked[top_y:bottom_y,left_x:right_x,:]#トリミング
        img_fit = cv2.resize(img_slice,dsize = (20,20))
        return img_fit
    
    #右の払いようけっこうてきとうでoK
    def get_hane_right_rec(self,migi_point):
            width = 20
            right_x = migi_point[0]+2
            left_x  = right_x-width+2
            top_y = migi_point[1]-10
            bottom_y = migi_point[1]+10
            img_slice = self.img_masked[top_y:bottom_y,left_x:right_x,:]#トリミング
            return img_slice


#小のクラス 
class Sho(Char):
    
    #漢字の名前だけこっちで指定
    def __init__(self,points_ori,img_paper):
        super().__init__(points_ori,img_paper)
        self.kanji = "小"
        self.is_stock_rec = False
        

    #スコアクラスを生成してそれを返す # is_stock_rec　recをdbにストックするかどうか
    def scoreing(self,is_stock_rec):
        self.score = Score(self.img_char,self.img_exp,self.kanji)
        self.is_stock_rec = is_stock_rec
        #領域の数が足りないときに
        if len(self.basic_contours) == 1:
            # すべての画がつながっているパターン
            # そもそも一画しか書いていないパターン
            self.score.add_item_phase1("かんじのかたちがへんだよ",False)
        elif len(self.basic_contours) == 2:
            #どれかが足りていない or どれかがくっついているも判定したい
            self.score.add_item_phase1("かんじのかたちがへんだよ",False)
        
        #3つ領域があればとりあえずクリア
        elif len(self.basic_contours) == 3:
            self.score.add_item_phase1("きれいに３ぼんせんがひけてるよ",True)
            
            #一画目のチェック
            #項目１右上から左下に線が引かれているか？
            self._kaku1_check(self.basic_contours[0])
            self._kaku2_check(self.basic_contours[1])
            self._kaku3_check(self.basic_contours[2])
            #TODO 高さチェック
            #TODO 書き出しチェック
            #self.height_check()#かくの高さ
            #崩れていなければpoints_numを使ってええ感じにfeaturepointを切り出す
        #余計な線がある
        elif len(self.basic_contours) > 4:
            self.score.add_item_phase1("かんじのかたちがへんだよ",False)
        return self.score
        
    
    def _kaku1_check(self,contour):
        #右上の判定　左下の判定　各点の距離が近くなるはずと考えている
        if my_cv.distance(contour.max_x_point,contour.min_y_point) <20 and my_cv.distance(contour.min_x_point,contour.max_y_point) <20:
            
            self.score.add_item_phase2("１かくめがみぎうえからひだりしたにきれいにひけてるね",[contour],100,True)
            #ここに払い判定が入る
            rec = contour.get_left_bottom_rec()
            if self.is_stock_rec == True:#stockモードなら
                DB.add(rec,self.img_char,self.kanji)
            

        #左下部分がうまく切り取れなかったら
        elif my_cv.distance(contour.max_x_point,contour.min_y_point) <20:#右上だけ切り取れた
            self.score.add_item_phase2("１かくめのはらいのむきにきをつけよう",[contour],60,False)
        else:#右上さえ切り取れなかった
            self.score.add_item_phase1("１かくめのかたちがへんだよ",False)
    
    def _kaku2_check(self,contour):
        #近似輪郭を出力する
        flg,approx = contour.get_approx(5,10000,30,0.01)
        if flg == False:
            self.score.add_item_phase1("２かくめのかたちがへんだよ",False)
        #横幅が狭かったらハネれてない証拠
        elif abs(contour.min_x_point[0]-contour.max_x_point[0]) < 20:
            self.score.add_item_phase2("２かくめのさいごはしっかりはねよう",[contour],20,False)
        else:
            approx_contour = Contour(approx,self.img_char,None)
            #my_cv.display_approx(approx_contour.img_char,approx_contour.cnt)#debug
            #左端の点から隣接しているポイントを抜き出す =>Y軸の高さを見る
            hidari_idx=approx_contour.min_x_index
            hidari_point = approx_contour.min_x_point
            p1 = approx_contour.get_point(hidari_idx+1)#隣接（下側になる傾向)
            p2 = approx_contour.get_point(hidari_idx-1)#隣接点（上側になる傾向) 
            d1 =  my_cv.distance(hidari_point,p1)
            d2 =  my_cv.distance(hidari_point,p2)
            
            #TODO 10が検出
            if d1 >10 and d2 >10:
                #距離が遠すぎる
                self._kaku2_hane_hantei(hidari_point,p1,p2,d1,d2,contour)
            #両方近い(ハネが極端に短い)
            elif d1 <= 10 and d2 <= 10:
                self.score.add_item_phase3("２かくめはしっかりはねよう",hidari_point,60,False)
                
            #左の点にすごく近い点がある場合点Aと点Bを１つの点とみなして同様の処理を行う。
            #p2が近いのでp2の次のやつがp2になる
            elif  d2 <= 10:
                p2new = approx_contour.get_point(hidari_idx-2)
                d2new = my_cv.distance(p2,p2new)
                if d2new >20:
                    self._kaku2_hane_hantei(my_cv.mid_point(p2,hidari_point),p1,p2new,d1,d2new,contour)
                else:
                    self.score.add_item_phase1("２かくめのかたちをかくにんしよう",False)
                
            #p1が近いのでp1の次のやつがp1になる
            elif d1 <= 10:
                p1new = approx_contour.get_point(hidari_idx+2)
                d1new = my_cv.distance(p1,p1new)
                if d1new >20:#近くに次の点がない
                    self._kaku2_hane_hantei(my_cv.mid_point(p1,hidari_point),p1new,p2,d1new,d2,contour)
                else:#近くに次の点があるのは違和感
                    self.score.add_item_phase1("２かくめのかたちをかくにんしよう",False)
                
            else:#ここにはたどり着かんはず
                print("kaku2 error")
            
            #右上と右下のてんがまっすぐかこうかをチェックする
            if abs(approx_contour.right_bottom_point[0] - approx_contour.right_top_point[0])>40:
                self.score.add_item_phase2("２かくめのせんはまっすぐひこう",[contour],60,False)
            else :
                self.score.add_item_phase2("２かくめがまっすぐかけてるね",[contour],60,False)
    
    #２かくめのハネの判定の処理をまとめるためのメソッド
    def _kaku2_hane_hantei(self,hidari_point,p1,p2,d1,d2,contour):
        #print(hidari_point)
        #my_cv.display_point(self.img_char,hidari_point)
        #print(p1)
        #my_cv.display_point(self.img_char,p1)
        #print(p2)
        #my_cv.display_point(self.img_char,p2)
        if d1 > 150 or d2 > 150:
            if hidari_point[1]+10 > p2[1] and p2[1] > p1[1]:#隣接点が基準点よりも下に存在しているか
                self.score.add_item_phase3("２かくめのはねがおおきすぎるよ",hidari_point,80,False)
            else :
                self.score.add_item_phase2("２かくめのせんはまっすぐたてにかこう",[contour],60,True)
        #ちょうどいい長さ
        else:
            #my_cv.display_point(contour.img_char,hidari_point)#debug
            #my_cv.display_point(contour.img_char,p2)#debug
            #my_cv.display_point(contour.img_char,p1)#debug
            if hidari_point[1]-10 < p2[1] and p2[1] < p1[1]:#隣接点が基準点よりも下に存在しているか
                #ハネの向きはok!
                self.score.add_item_phase3("２かくめがしっかりはねれているね",hidari_point,100,False)
                
                #ハネ判定
                rec = contour.get_hane_left_rec(hidari_point,p2)#判定部分の矩形領域の切り出し
                if self.is_stock_rec == True:#stockモードなら
                    DB.add(rec,self.img_char,self.kanji)
            else :
                self.score.add_item_phase3("２かくめのはねのむきがおかしいよ",hidari_point,50,False)
                

    def _kaku3_check(self,contour):
        #左上、右下がひとつに定まるかのチェック
        if my_cv.distance(contour.min_x_point,contour.min_y_point) < 20 and my_cv.distance(contour.max_x_point,contour.max_y_point) <20:
            
            self.score.add_item_phase2("３かくめがきれいにかけてるね",[contour],100,True)

            #はねはらいはんてい
            rec = contour.get_right_bottom_rec()
            if self.is_stock_rec == True:#stockモードなら
                DB.add(rec,self.img_char,self.kanji)#stockする


        elif my_cv.distance(contour.min_x_point,contour.min_y_point) <20:#右上だけ切り取れた
            self.score.add_item_phase1("３かくめのかたちがへんだよ",False)
        else:
            self.score.add_item_phase1("３かくめのかたちがへんだよ",False)
            

class Mizu(Char):
    #漢字の名前だけこっちで指定
    def __init__(self,points_ori,img_paper):
        #print("===-")
        #print(points_ori)
        super().__init__(points_ori,img_paper)
        #print(self.points_ori)
        #my_cv.display_color(self.img_char)
        self.kanji = "水"

    #スコアクラスを生成してそれを返す
    def scoreing(self,is_stock_rec):
        self.score = Score(self.img_char,self.img_exp,self.kanji)
        self.is_stock_rec = is_stock_rec


        
        #とりあえず３つに分けれるかどうか

        #領域の数が足りないときに
        if len(self.basic_contours) == 1:
            # すべての画がつながっているパターン
            # そもそも一画しか書いていないパターン
            self.score.add_item_phase1("かんじのかたちがへんだよ",False)
        elif len(self.basic_contours) == 2:
            #どれかが足りていない or どれかがくっついているも判定したい
            self.score.add_item_phase1("ただしくせんがひけてるね",True)
            
            self._con1_check(self.basic_contours[0])
            self._con2_check(self.basic_contours[1])
            
             
        elif len(self.basic_contours) >3:
            self.score.add_item_phase1("かんじのかたちがへんだよ",False)

        return self.score

    def _con1_check(self,contour):
        flg, approx = contour.get_approx(4,10000,30,0.01)
        if flg == False:#approxがうまく切り取れなかった
            self.score.add_item_phase1("１かくめのかたちがへんだよ",False)
        else:
            kakidashi_point = contour.left_top_point #かきだし
            kakiowari_point = contour.left_bottom_point #書き終わり
            migiue = contour.right_top_point
            
            # 1領域目の評価基準
            # 左の払いと書き出しが離れすぎていたら左にあるから70点 

            # 右の点を切り出す。右の点から書き出しが上にあったら
            # 書き出しから右の点までの距離が短すぎたら
            if abs(contour.max_x_point[0]-contour.min_x_point[0]) < 20 or abs(contour.min_y_point[1]-contour.max_y_point[1])< 30:
                self.score.add_item_phase2("１かくめはもうすこしおおきくかこう",[contour],20,False)
            else:
            #一画目の横線について
                flg = True
                if abs(migiue[0] - kakidashi_point[0]) < 20:
                    self.score.add_item_phase2("１かくめのよこせんのながさがみじかいよ",[contour],60,False)
                    flg =False
                
                elif migiue[1] > kakidashi_point[1]-10:
                    self.score.add_item_phase2("１かくめのよこせんはややななめうえにむかってかこう",[contour],60,False)
                    flg =False
                
                #横線と斜め線について
                if abs(kakidashi_point[0] - kakiowari_point[0]) > 50:
                    self.score.add_item_phase2("１かくめのかきだしとかきおわりのいちがずれすぎてるよ",[contour],60,False)
                    flg =False
                #ここらへんはもっと微妙なのが狙える
                #斜め線について
                
                #右上点と比較して右上から左下に向かって線が引かれている。
                if migiue[0] < kakiowari_point[0] and migiue[1] > kakiowari_point[1]:
                    self.score.add_item_phase1("１かくめのかたちがへんだよ",False)
                    flg =False

                #縦横比で極端に横が長い 縦に比べて横が1.3倍以上
                if contour.width / contour.height >1.3:
                    self.score.add_item_phase2("ななめのせんはややしたむきにかこう",[contour],80,False)
                    flg =False
                if flg:
                    self.score.add_item_phase2("１かくめがきれいにかけてるね",[contour],100,True)
                    rec = contour.get_left_bottom_rec()
                    if self.is_stock_rec == True:#stockモードなら
                        DB.add(rec,self.img_char,self.kanji)
                        
                
        
    def _con2_check(self,contour):
        flg, approx = contour.get_approx(14,10000,30,0.01)
        if flg == False:#approxがうまく切り取れなかった
            self.score.add_item_phase1("かんじのかたちがへんだよ",False)
            return self.score
        #方針
        #とりあえず左のハネが存在してるか確認する（これは小のコードが使い回せる)
        approx_contour = Contour(approx,self.img_char,None)
        kaku2_bottom = self._hidarihane_check(approx_contour,contour)

        #self
        #右上から左下にかけての３かくめを見る(角度と長さ)
        #前提としてmin_y_pointは真ん中に来てる

        kaku2_kakidashi = approx_contour.min_y_point
        min_y = 255
        kaku3_kakidashi = None
        kaku3_kakidashi_idx = 0
        for i,point in enumerate(approx_contour.cnt):
            point = point[0]
            if kaku2_kakidashi[0]+20 < point[0] and min_y > point[1]: #つまりmin_y_pointよりも30px以上右にある点で一番上に存在している点が３かくめのかきだし
                min_y = point[1]
                kaku3_kakidashi = point
                kaku3_kakidashi_idx=i
        if kaku3_kakidashi is None:
            self.score.add_item_phase1("かんじのかたちがへんだよ",False)
            return self.score
        #my_cv.display_point(self.img_char,kaku2_kakidashi)
        #my_cv.display_point(self.img_char,kaku3_kakidashi)
        #書き出しのいちの妥当性
        if kaku3_kakidashi[1] > 127:
            self.score.add_item_phase2("３かくめのかきはじめのいちははんぶんよりうえにかこう",[contour],40,False)
        if kaku2_kakidashi[1] > kaku3_kakidashi[1]:
            self.score.add_item_phase2("３かくめのかきはじめは２かくめのかきだしよりしたになるようにしよう",[contour],80,False)
        if abs(kaku2_kakidashi[0] - kaku3_kakidashi[0]) <10:
            self.score.add_item_phase2("３かくめのかきはじめが２かくめにちかすぎるよ",[contour],80,False)
    

        #左下への払いを見る 
        # まず書き終わり点の抽出           
        kaku3_kakiowari1 = approx_contour.get_point(kaku3_kakidashi_idx+1)#隣接（上側になる傾向)
        kaku3_kakiowari2 = approx_contour.get_point(kaku3_kakidashi_idx-1)#隣接点（下側になる傾向)
        kaku3_kakiowari2_idx =  approx_contour.get_idx(kaku3_kakidashi_idx)
        d1 =  my_cv.distance(kaku3_kakidashi,kaku3_kakiowari1)
        d2 =  my_cv.distance(kaku3_kakidashi,kaku3_kakiowari2)
        if d1 >15 and d2 >15:
            #判定
            self._kaku3_check(kaku3_kakidashi,kaku3_kakiowari1,kaku3_kakiowari2,contour)
        #両方近い(ハネが極端に短い)
        elif d1 <= 15 and d2 <= 15:
            self.score.add_item_phase3("3かくめがみじかすぎるよ",[contour],80,False)
        #左の点にすごく近い点がある場合点Aと点Bを１つの点とみなして同様の処理を行う。
        #p2が近いのでp2の次のやつがp2になる
        elif  d2 <= 15:
            kaku3_kakiowari2 = approx_contour.get_point(kaku3_kakidashi_idx-2)
            d2new = my_cv.distance(kaku3_kakiowari2,approx_contour.get_point(kaku3_kakidashi_idx-1))
            if d2new < 20:#近くに次の点がない
                #print("a")
                self.score.add_item_phase2("３かくめのむきとながさをかくにんしよう",[contour],30,False)
            else:
                kaku3_kakiowari2_idx =  approx_contour.get_idx(kaku3_kakidashi_idx-20) 
                self._kaku3_check(kaku3_kakidashi,kaku3_kakiowari1,kaku3_kakiowari2,contour)

        #p1が近いのでp1の次のやつがp1になる
        elif d1 <= 15:
            kaku3_kakiowari1 = approx_contour.get_point(kaku3_kakidashi_idx+2)
            
            #my_cv.display_point(self.img_char,kaku3_kakiowari1)
            d1new = my_cv.distance(kaku3_kakiowari1,approx_contour.get_point(kaku3_kakidashi_idx+1))
            if d1new <20:#近くに次の点がない
                #print("b")
                self.score.add_item_phase2("３かくめのむきとながさをかくにんしよう",[contour],30,False)
            else:
                self._kaku3_check(kaku3_kakidashi,kaku3_kakiowari1,kaku3_kakiowari2,contour)
            

        #4角目払い判定
        kaku4_owari = None# ここを確定させる
        if contour.right_bottom_index == contour.max_y_index:
            self.score.add_item_phase2("４かくめのはらいおわりは２かくめよりもうえではらおう",[contour],30,False)
        elif contour.right_bottom_index == (kaku3_kakidashi_idx-1):#書き出しの次と一緒はおかしい
            self.score.add_item_phase1("２〜４かくめのかたちがへんだよ",False)
        else:
            kaku4_owari = contour.right_bottom_point #おわりがかくていする
            kaku4_owari_idx = contour.right_bottom_index
            kaku3_owari = kaku3_kakiowari2 #かく３の下側の終わり
            kaku3_owari_idx = kaku3_kakiowari2_idx
            kaku4_kakidashi = approx_contour.get_point(kaku3_owari_idx-1)#ここを確定させる
            
            #こっから払い判定
            #払いは途中で曲がっている可能性もあるしそうでない可能性もある
            #３角目のかきおわりから書き出しを特定する
            #かく３のおわりとかく４のかきだしが離れているパターン
            if abs(kaku4_kakidashi[0]-kaku3_owari[0])<10 and  abs(kaku4_kakidashi[0]-kaku3_owari[0])>10:
                self.score.add_item_phase2("３かくめのかきおわりと４かくめのかきだしがおなじいちにくるようにしよう",[contour],60,False)
        
            else :#近くに点がないのならいけそう
                kaku4_kakidashi = kaku3_owari#確定させる
                #途中でかくってなってるかどうかはとわない
                #distances
                if kaku4_owari[0] < kaku3_kakidashi[0]:
                    self.score.add_item_phase2("３かくめのかきはじめよりも４かくめはながくのばそう",[contour],60,False)
                elif my_cv.distance(kaku4_kakidashi,kaku4_owari)<40:
                    self.score.add_item_phase2("４かくめがみじかいよ",[contour],70,False)
                #書きおわりがmax_yとおなじになっちゃってたら
                elif kaku2_bottom[1] < kaku4_owari[1]:
                    self.score.add_item_phase2("４かくめがながすぎるよ",[contour],60,False)
                else:
                    #はらい判定
                    self.score.add_item_phase2("４かくめがしっかりかけてるね",[contour],100,True)
                    rec = contour.get_hane_right_rec(kaku4_owari)#判定部分の矩形領域の切り出し
                    if self.is_stock_rec == True:#stockモードなら
                        DB.add(rec,self.img_char,self.kanji)
            
                    self.score.add_item_phase3("４かくめがしっかりはらえているね",kaku4_owari,100,True)
        return self.score

    #２かくめようのひだりのはねがうまくいってるかをみるメソッド
    def _hidarihane_check(self,approx_contour,contour):
        hidari_idx=approx_contour.min_x_index
        hidari_point = approx_contour.min_x_point
        p1 = approx_contour.get_point(hidari_idx+1)#隣接（下側になる傾向)
        p2 = approx_contour.get_point(hidari_idx-1)#隣接点（上側になる傾向) 
        d1 =  my_cv.distance(hidari_point,p1)
        d2 =  my_cv.distance(hidari_point,p2)
        #TODO 10が検出
        if d1 >10 and d2 >10:
            #距離が遠すぎる
            self._kaku2_hane_hantei(hidari_point,p1,p2,d1,d2,contour)
        #両方近い(ハネが極端に短い)
        elif d1 <= 10 and d2 <= 10:
            self.score.add_item_phase3("２かくめはしっかりはねよう",hidari_point,60,False)
                        
        #左の点にすごく近い点がある場合点Aと点Bを１つの点とみなして同様の処理を行う。
        #p2が近いのでp2の次のやつがp2になる
        elif  d2 <= 10:
            p2new = approx_contour.get_point(hidari_idx-2)
            d2new = my_cv.distance(p2,p2new)
            if d2new >20:
                self._kaku2_hane_hantei(my_cv.mid_point(p2,hidari_point),p1,p2new,d1,d2new,contour)
            else:
                self.score.add_item_phase1("２かくめのかたちをかくにんしよう",False)
            
        #p1が近いのでp1の次のやつがp1になる
        elif d1 <= 10:
            p1new = approx_contour.get_point(hidari_idx+2)
            d1new = my_cv.distance(p1,p1new)
            if d1new >20:#近くに次の点がない
                self._kaku2_hane_hantei(my_cv.mid_point(p1,hidari_point),p1new,p2,d1new,d2,contour)
            else:#近くに次の点があるのは違和感
                self.score.add_item_phase1("２かくめのかたちをかくにんしよう",False)
                
        else:#ここにはたどり着かんはず
            print("kaku2 error")
        if abs( approx_contour.left_top_point[0] - p2[0]) >40:
            self.score.add_item_phase2("２かくめのせんはまっすぐひこう",[contour],60,False)
        else:
            self.score.add_item_phase2("２かくめのせんがまっすぐかけてるね",[contour],100,True)
        return p1#２角目の一番下の位置の座標を返す
        
        #２かくめのハネの判定の処理をまとめるためのメソッド
    def _kaku2_hane_hantei(self,hidari_point,p1,p2,d1,d2,contour):
        #print(hidari_point)
        #my_cv.display_point(self.img_char,hidari_point)
        #print(p1)
        #my_cv.display_point(self.img_char,p1)
        #print(p2)
        #my_cv.display_point(self.img_char,p2)
        if d1 > 150 or d2 > 150:
            if hidari_point[1]+10 > p2[1] and p2[1] > p1[1]:#隣接点が基準点よりも下に存在しているか
                self.score.add_item_phase3("２かくめのはねがおおきすぎるよ",hidari_point,80,False)
            else :#線が曲がりまくって払いよりも左に来てたパターン
                self.score.add_item_phase2("２かくめのせんはまっすぐたてにかこう",[contour],60,False)
        #ちょうどいい長さ
        else:
            #my_cv.display_point(contour.img_char,hidari_point)#debug
            #my_cv.display_point(contour.img_char,p2)#debug
            #my_cv.display_point(contour.img_char,p1)#debug
            if hidari_point[1]-10 < p2[1] and p2[1] < p1[1]:#隣接点が基準点よりも下に存在しているか
                #ハネの向きはok!
                self.score.add_item_phase3("２かくめがしっかりはねれているね",hidari_point,100,False)
                
                #ハネ判定
                rec = contour.get_hane_left_rec(hidari_point,p2)#判定部分の矩形領域の切り出し
                if self.is_stock_rec == True:#stockモードなら
                    DB.add(rec,self.img_char,self.kanji)
            else :
                self.score.add_item_phase3("２かくめのはねのむきがおかしいよ",hidari_point,50,False)
    
    #画３の判定 斜めかどうか
    def _kaku3_check(self,kaku3_kakidashi,kaku3_kakiowari1,kaku3_kakiowari2,contour):
        if abs(kaku3_kakidashi[0] - kaku3_kakiowari1[0]) > 30 and abs(kaku3_kakidashi[1] - kaku3_kakiowari1[1]) > 20 :
            self.score.add_item_phase2("３かくめがきれいにかけてるね",[contour],60,False)
        else:
            self.score.add_item_phase2("３かくめのむきとながさをかくにんしよう",[contour],30,False)  



class Char_Sample(Char):
    def __init__(self,points_ori,img_paper):
        super().__init__(points_ori,img_paper)
        self.kanji = "サンプル"

    #スコアクラスを生成してそれを返す
    def scoreing(self,is_stock_rec):
        self.score = Score(self.img_char,self.img_exp,self.kanji)      
        return self.score
