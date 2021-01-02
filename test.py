class Factory():
    def create_char(self,char_name,char):#charの名前とcharインスタンスを渡す
        if char_name == "Mizu": 
            mizu = (Mizu)char
            return mizu
        elif char_name == "Sho":
            sho = (Sho)chars
            return sho


class Paper():
    def __init__(self,paper_img,char1_name,char2_name):#紙の領域が切り取られたimgを受け取る
        self.char1_name = char1_name#紙で言う右の文字(小)
        self.char2_name = char2_name#上で言う左の文字(水)
        self.chars = _find_chars(img)#文字の集合16個で固定
        self.char1s = list() # 採点する文字１の配列
        self.char2s= list() # 採点する文字２の配列
        self.char1_exp = None # 文字１の見本
        self.char2_exp = None # 文字２の見本
        self._labeling()#ラベリングして上に振り分ける

    #label付けを行う
    def _labeling(self):
        fty = Factory()
        for char in self.chars:
            if 1300<char.get_lu_point()[0]<1400:
                self.char1_exp = char
            elif 1000<char.get_lu_point()[0]<1100:
                self.char2_exp = char
            elif 400<char.get_lu_point()[0]<500:
                if char.get_lu_point()[1]< 650:
                    pass #なぞりの部分は評価しない
                else:
                    self.char1s.append(fty.create_char(self,char1_name,char))
            elif 100<char.get_lu_point()[0]<200:
                if char.get_lu_point()[1]<650:
                    pass #なぞりの部分は評価しない
                else:
                    self.char2s.append(fty.create_char(self,char2_name,char))
            else:
                print("error")
                
        
    def _find_chars(self,img, lower_thre=66000, upper_thre = 71000,sq_num = 16):#img
        gray0 = np.zeros(img.shape[:2], dtype=np.uint8)
        rows, cols, _channels = map(int, img.shape)        # down-scale and upscale the image to filter out the noise
        pyr = cv2.pyrDown(image, dstsize=(cols//2, rows//2))
        timg = cv2.pyrUp(pyr, dstsize=(cols, rows))
        cv2.mixChannels([timg], [gray0], (0, 0))# 画像のBGRの色平面で正方形を見つける ( 0で固定した)
        for l in range(0, 5):# いくつかのしきい値レベルを試す
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
            chars = list()
            for i, cnt in enumerate(contours):
                arclen = cv2.arcLength(cnt, True)# 輪郭の周囲を取得
                approx = cv2.approxPolyDP(cnt, arclen*0.02, True)# 輪郭の近似
                area = abs(cv2.contourArea(approx))# 面積
                #長方形の輪郭は、近似後に4つの角をもつ、
                #比較的広い領域
                #（ノイズの多い輪郭をフィルターで除去するため）
                #凸性(isContourConvex)になります。
                if approx.shape[0] == 4 and upper_thre > area > lower_thre and cv2.isContourConvex(approx) :
                    maxCosine = 0
                    #print(approx)
                    for j in range(2, 5):   # ジョイントエッジ間の角度の最大コサインを見つけます
                        cosine = abs(angle(approx[j%4], approx[j-2], approx[j-1]))
                        maxCosine = max(maxCosine, cosine)
                    if maxCosine < 0.1 :# すべての角度の余弦定理が小さい場合（すべての角度が約90度）、
                        char = Char(approx)
                        for ch in self.chars:
                            if ch.is_same(char):
                                break
                            self.chars.append(char)
            if sq_num == len(chars):
                return chars

class Char():
     def __init__(self, points_ori):# np.shape(points) = (4,1,2)
        self._arrange_points(points_ori)#座標を整理
        
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
        self.points_ori = np.stack(stack)

    #左上の座標が同じかどうか
    def is_same(self, other):
        op = other.points_ori[0][0]
        mp = self.points_ori[0][0]
        if abs(op[0] - mp[0]) + abs(op[1]-mp[1]) <=30:
            return True
        return False    
                               
    #四角に整形して結果を保持する
    def fit_image(self,img,x=255,y=255):
        epsilon = 0.1*cv2.arcLength(self.points_ori,True)
        paper_corners= cv2.approxPolyDP(self.points_ori,epsilon,True)#紙の頂点座標
        fix_con = np.array([[[0,0]],[[x,0]],[[x,y]],[[0,y]]], dtype="int32")#整形後のサイズ
        trans_arr = cv2.getPerspectiveTransform(np.float32(paper_corners),np.float32(fix_con))#変換行列の生成
        self.img_trans = cv2.warpPerspective(img,trans_arr,(x,y))#変換

    
    #左上のアドレスを(2)で返す
    def get_lu_point(self):
        return self.points_ori[0][0]
    
    #表示する
    def display(self):
        cv2.imwrite("temp.png", self.img_trans)
        plt.imshow(plt.imread("temp.png"))
        plt.axis('off')
        plt.show()
        
    #cntsを取得する
    def find_cnts(self):
        ctns, hie = cv2.findContours(sq.img_filter, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        return (ctns, hie)

    #　ポイントを出力する   
    def get_feature_points(self):
        thre_img = np.copy(sq.img_filter)
        contours, _ = cv2.findContours(thre_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        rate_parms = [800]
        for n, rate in enumerate(rate_parms):
            resimg = sq.img_trans.copy() // 2 + 128
            for i, cnt in enumerate(contours):
                # 輪郭の周囲に比例する精度で輪郭を近似する
                size = cv2.contourArea(cnt)
                approx = cv2.approxPolyDP(cnt, rate/size, True)#第２引数が小さければ細かい近似大きければ大雑把な近似
                print(int(size))
                print(rate/size)
                print(np.shape(approx))
                if size>(255*255-5000):
                    continue
                cv2.polylines(resimg, [approx.reshape(-1,2)], True, 
                            (0,0,255), thickness=1, lineType=cv2.LINE_8)
                for app in approx[0:3]:
                    cv2.circle(resimg, (app[0][0],app[0][1]), 2, (0, 255, 0), thickness=-1)
            my_cv.display_color(resimg)
       
    #img_transにたいしてfilterをかけることで文字部分のみを切り出してimg_filterとして格納する
    def filter_image(self):
        img = self.img_trans
        result = np.copy(img)
        # HSV色空間に変換
        #hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # 青色のHSVの値域1
        mi = np.array([60, 60, 30])#小さくするときつくなる
        ma = np.array([255,255,255])
        # 青色領域のマスク（255：赤色、0：赤色以外）    
        mask = cv2.inRange(img, mi, ma)
        mask = cv2.bitwise_not(mask)
        for i in range(255):
            for j in range(255):
                if mask[i][j] == 0:#黒なら
                    for k in range(3):
                        result[i][j][k] = 255      
        result = cv2.fastNlMeansDenoisingColored(result,None,50,9,7,21)
        img_gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
        ret, img_thresh = cv2.threshold(img_gray, 230, 255, cv2.THRESH_BINARY)#100以上の画素255
        #my_cv.display_gray(img_thresh)
        contours, _ = cv2.findContours(img_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        img_del_min = np.copy(img_thresh)
        img_draw_con = np.copy(img_thresh)
        for con in contours:
            area = cv2.contourArea(con)
            if area<120 :
                img_del_min = cv2.drawContours(img_del_min, [con], 0, 255, -1)
        #my_cv.display_gray(img_del_min)
        self.img_filter = img_del_min
    
    def check_stracture():
        pass

    
class Mizu (Char):
    def check_stracture(self):
        pass
    
class Sho(Char):
        
    #切り取り領域の数をチェックする
    def check_cnt_count(self):
        ctns, hie = self.find_cnts()
        print(ctns,hie)
    
    
    
    
