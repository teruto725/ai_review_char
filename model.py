import numpy as np
import cv2
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import my_cv



class Factory():
    def create_char(self,char_name,char):#charの名前とcharインスタンスを渡す
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
        #print(np.shape(self.chars))
        self.char1s = list() # 採点する文字１の配列
        self.char2s= list() # 採点する文字２の配列
        self.char1_exp = None # 文字１の見本
        self.char2_exp = None # 文字２の見本
        self._labeling()#ラベリングして上に振り分ける

    #label付けを行う
    def _labeling(self):
        fty = Factory()
        for char in self.chars:
            if 1200<char.get_lu_point()[0]<1700:
                self.char1_exp = char
            elif 900<char.get_lu_point()[0]<1100:
                self.char2_exp = char
            elif 400<char.get_lu_point()[0]<600:
                if char.get_lu_point()[1]< 650:
                    pass #なぞりの部分は評価しない
                else:
                    self.char1s.append(fty.create_char(self.char1_name,char))
            elif 100<char.get_lu_point()[0]<200:
                if char.get_lu_point()[1]<650:
                    pass #なぞりの部分は評価しない
                else:
                    self.char2s.append(fty.create_char(self.char2_name,char))
            else:
                print("labeling_error")
                char.display()
                print(char.get_lu_point())
                
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
            my_cv.display_con(img,contours)#debug
            #print(np.shape(contours))#ここまではok
            chars = list()
            for i, cnt in enumerate(contours):
                arclen = cv2.arcLength(cnt, True)# 輪郭の周囲を取得
                approx = cv2.approxPolyDP(cnt, arclen*0.05, True)# 輪郭の近似
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

class Char():
    THRESH_NUM = 160 #この値より大きい画素を白にする
    def __init__(self, points_ori,img_paper):# np.shape(points) = (4,1,2)
        self.points_ori = self._arrange_points(points_ori) #is_sameメソッドで使用する
        self.img_paper = img_paper
        self.img_sq = self._fit_image(img_paper,points_ori)# 255*255の正方形に変換した画像
        #my_cv.display_color(self.img_sq)
        #self.img_fltr = self._filter_image()#フィルターがかけられた2値の画像
        self.img_thresh = self._get_img_thresh()
        #my_cv.display_gray(self.img_thresh) #debug
        self.score = Score() # 採点結果を記録する辞書
        self.basic_contours = self._get_basic_contours()
        
    #points_oriを時計周りに修正する (0,0)(1,0)(1,1)(0,1)
    def _arrange_points(self,points):
        stack = list()
        m_s_idx = np.argsort([ p[0][0] + p[0][1] for p in points])#足し算でソートしたindex
        stack.append(points[m_s_idx[0]])
        if points[m_s_idx[1]][0][0] > points[m_s_idx[2]][0][0]:
            stack.append(points[m_s_idx[1]])
            stack.append(points[m_s_idx[3]])
            stack.append(points[m_s_idx[2]])
        else:
            stack.append(points[m_s_idx[2]])
            stack.append(points[m_s_idx[3]])
            stack.append(points[m_s_idx[1]])
        return  np.stack(stack)
    
    #四角に整形して結果を保持する
    def _fit_image(self,img,points_ori,x=255,y=255):
        epsilon = 0.1*cv2.arcLength(points_ori,True)#周辺長の取得
        paper_corners= cv2.approxPolyDP(points_ori,epsilon,True)#紙の頂点座標
        fix_con = np.array([[[0,0]],[[x,0]],[[x,y]],[[0,y]]], dtype="int32")#整形後のサイズ
        trans_arr = cv2.getPerspectiveTransform(np.float32(paper_corners),np.float32(fix_con))#変換行列の生成
        return  cv2.warpPerspective(img,trans_arr,(x,y))#変換

    #get_contorus()
    def _get_img_thresh(self):
        result = np.copy(self.img_sq)
        gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
        _, img_thresh = cv2.threshold(gray, Char.THRESH_NUM, 255, cv2.THRESH_BINARY)
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
        return [Contour(cnts_sorted[i],self.img_sq,hie) for i in range(len(cnts_sorted))]

    # ある程度の精度で直線近似したconotursを生成する
    def _get_approximation_contours(self):
        pass
    #img_transにたいしてfilterをかけることで文字部分のみを切り出してimg_filterとして格納する
    def _filter_image(self):
        img = self.img_sq
        result = np.copy(img)
        # HSV色空間に変換
        #hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # 青色のHSVの値域1
        """
        mi = np.array([140, 140, 140])#小さくするときつくなる
        ma = np.array([255,255,255])
        # 青色領域のマスク（255：赤色、0：赤色以外）    
        mask = cv2.inRange(img, mi, ma)#mi~maを黒色にしてしまう
        mask = cv2.bitwise_not(mask)
        for i in range(255):
            for j in range(255):
                if mask[i][j] == 0:#黒なら
                    for k in range(3):
                        result[i][j][k] = 255      
        result = cv2.fastNlMeansDenoisingColored(result,None,50,9,7,21)
        """
        img_gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)#
        ret, img_thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)#100以上の画素255
        contours, _ = cv2.findContours(img_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        img_del_min = np.copy(img_thresh)
        img_draw_con = np.copy(img_thresh)
        for con in contours:
            area = cv2.contourArea(con)
            if area<120 :
                img_del_min = cv2.drawContours(img_del_min, [con], 0, 255, -1)
        #my_cv.display_gray(img_del_min)
        return img_del_min
    
    
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
        cv2.imwrite("temp.png", self.img_sq)
        plt.imshow(plt.imread("temp.png"))
        plt.axis('off')
        plt.show()
        

    #　ポイントを出力する   
    def get_feature_points(self):
        thre_img = np.copy(self.img_thresh)
        contours, _ = cv2.findContours(thre_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        rate_parms = [800]
        for n, rate in enumerate(rate_parms):
            resimg = self.img_sq.copy() // 2 + 128
            for i, cnt in enumerate(contours):
                # 輪郭の周囲に比例する精度で輪郭を近似する
                size = cv2.contourArea(cnt)
                if size == 0:
                    continue
                #1.2をかけることで大きい図形に対してのバイアスを強くしている
                approx = cv2.approxPolyDP(cnt, rate/size*1.5, True) #第２引数が小さければ細かい近似大きければ大雑把な近似 1.2をかけること
                print(int(size))
                print(rate/size)
                print(np.shape(approx))
                if size>(255*255-5000):
                    continue
                cv2.polylines(resimg, [approx.reshape(-1,2)], True, 
                            (0,0,255), thickness=1, lineType=cv2.LINE_8)
                for app in approx:
                    cv2.circle(resimg, (app[0][0],app[0][1]), 2, (0, 255, 0), thickness=-1)
            my_cv.display_color(resimg)

    
    def check_stracture():
        pass
    
    def print_score():
        pass    

class Score():
    def __init__(self):
        self.score = list()
    
    #scoreクラス
    def add_score(self,message,point):
        pass

#領域クラス
class Contour():
    def __init__(self,cnt,img_char,hie):
        self.cnt = cnt # 領域アドレス
        self.img_char = img_char # 文字の画像
        self.hie = hie
        self._define_features()
    
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
        
        for point in self.cnt:
            point = point[0]
            if self.min_x_point is None:
                self.min_x_point = point
                self.min_y_point = point
                self.max_x_point = point
                self.max_y_point = point
                continue
            if self.min_x_point[0] > point[0]:
                self.min_x_point = point
            if self.min_y_point[1] > point[1]:
                self.min_y_point = point
            if self.max_x_point[0] < point[0]:
                self.max_x_point = point
            if self.max_y_point[1] < point[1]:
                self.max_y_point = point
    
        self.width = abs(self.max_x_point[0] - self.min_x_point[0]) #幅
        self.height = abs(self.max_y_point[1] - self.min_y_point[1]) #高さ
    
    #右上から左下にかけて引かれている線でx軸方面に飛び出していないかmin_x_point max_y_point 
    def is_migiue_to_hidarishita(self):
        #右上が点１つ 左下が点1つ 
        return my_cv.distance(self.max_x_point,self.min_y_point) <10 and my_cv.distance(self.min_x_point,self.max_y_point) <40

    def is_hidariue_to_migishita(self):
        return 
    #二値化画像をプロット
    def display_thresh(self):
        result = np.zeros([255,255])
        result = cv2.drawContours(result,[self.cnt],0,255,-1)
        my_cv.display_gray(result)

    #二値化画像をプロット
    def display(self):
        result = np.copy(self.img_char)
        result = cv2.drawContours(result,[self.cnt],0,(255,0,255),1)
        my_cv.display_gray(result)

    
#小のクラス 
class Sho (Char):
    points_num = [2,3,2] # これに合わせてfeaturepointを取得する
    centroid_x_scopes = [[0,110],[110,145],[145,155]] #3つのcentroidの基準
    width_scopes = [[20,80],[5,30],[20,80]]#widthの範囲
    height_scope = [[20,100],[130,230],[20,100]] # heightの範囲
    def check_stracture(self):
        pass
            #切り取り領域の数をチェックする
    
    #スコアを取得する
    def get_score(self):
        
        #領域の数が足りないときに
        if len(self.basic_contours) == 1:
            # すべての画がつながっているパターン
            # そもそも一画しか書いていないパターン
            # 
            self.score.add_score("かんじのかたちがへんだね、かきわすれやせんがつながっていないかかくにんしよう",None)
            return 0
        elif len(self.basic_contours) == 2:
            #どれかが足りていない or どれかがくっついているも判定したい
            self.score.add_score("かんじのかたちがへんだね、かきわすれやせんがつながっていないかかくにんしよう", None)
            return 0
        

        #3つ領域があればとりあえずクリア
        elif len(self.basic_contours) == 3:
            
            #一画目のチェック
            #項目１右上から左下に線が引かれているか？
            kaku1 = self.basic_contours[0]
            self._kaku1_check(kaku1)
            self._kaku2_check(kaku2)
            self._kaku3_check(kaku3)
            

            #崩れていなければpoints_numを使ってええ感じにfeaturepointを切り出す
            # はね、はらい部分の判定を行う

            #最後に更に細かくチェックを行う
             #一画目
            for i,contour in enumerate(self.basic_contours):
                if contou 

            pass
            
        elif ctns > 4:


        print(ctns,hie)
    
    def _kaku1_check(self,contour):
        #右上の判定　左下の判定　各点の距離が近くなるはずと考えている
        if my_cv.distance(contour.max_x_point,contour.min_y_point) <10 and my_cv.distance(contour.min_x_point,contour.max_y_point) <10:
            migiue = my_cv.mid_point()
        #左下部分がうまく切り取れなかったら
        elif my_cv.distance(contour.max_x_point,contour.min_y_point) <10:
            self.score.add_score("はらいのむきがただしいかかくにんしよう", contour)
        #払いができているか確認する
        elif True:
            pass 
        else:
            self.score.add_score("いっかくめのかたちがへんだね、みぎうえからひだりしたにむかってきれいにひこう",contour)
    
    def _kaku2_check(self,contour):
        #真っ直ぐに引けているかの判定


    def get_feature_points():
        pass


class Mizu(Char):
    pass


class 