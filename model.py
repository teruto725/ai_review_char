import numpy as np
import cv2
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

class CVM():
#影を消す
    @staticmethod
    def deshade(img):
        rgb_planes = cv2.split(img)
        result_planes = []
        result_norm_planes = []
        for plane in rgb_planes:
            dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
            bg_img = cv2.medianBlur(dilated_img, 21)
            diff_img = 255 - cv2.absdiff(plane, bg_img)
            arr = np.array([])
            norm_img = cv2.normalize(diff_img,arr, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            result_planes.append(diff_img)
            result_norm_planes.append(norm_img)

        result = cv2.merge(result_planes)
        result_norm = cv2.merge(result_norm_planes)
        display(result)
        return result

    #白黒画像を表示する
    @staticmethod
    def display_gray(img,output_file_path="tmp.jpg"):
        cv2.imwrite(output_file_path, img)
        plt.imshow(plt.imread(output_file_path))
        plt.axis('off')
        plt.gray()
        plt.show()

    #カラー画像を表示する
    @staticmethod
    def display_color(img,output_file_path="tmp.jpg"):
        cv2.imwrite(output_file_path, img)
        plt.imshow(plt.imread(output_file_path))
        plt.axis('off')
        plt.show()
    @staticmethod
    def display_con(img,con,output_file_path="tmp.jpg"):
        im_con = img.copy()
        cv2.drawContours(im_con, con, -1, (0,255,0), 3)
        display_color(im_con)

    #threshold以上の画素値を白にする
    @staticmethod
    def threshold(img,threshold):
        ret, img_thresh = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)#100以上の画素255
        return img_thresh

    #グレースケール化
    @staticmethod
    def gray(img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img_gray    

    #境界値を取り出す
    @staticmethod
    def findContours(img):
        img_gray = gray(img)#グレイスケール
        #２値化
        ret,img_th1  = cv2.threshold(img_gray,220,255,cv2.THRESH_TOZERO_INV)
        img_not = cv2.bitwise_not(img_th1)
        ret,img_th2  = cv2.threshold(img_not,0,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU )
        #輪郭抽出
        contours, hierarchy = cv2.findContours(img_th2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return (contours, hierarchy)


    #カラー画像を受け取って紙を切り取って返す 紙の向きに注意
    @staticmethod
    def cutting_paper(img):
        x = 2000 #切り取り後のx座標
        y = int(x*1.415)

        # 境界抽出
        contours, _ = findContours(img)

        #２番目にでかい四角が紙の輪郭なのでpaper_tapを取り出す
        pic_tap = None #画像の輪郭
        paper_tap = None #紙の輪郭
        for i, con in enumerate(contours):
            size = cv2.contourArea(con)
            if pic_tap is None:
                pic_tap = (con,size)
                continue
            if pic_tap[1] <= size:
                pic_tap = (con,size)
                continue
            elif paper_tap is None:
                paper_tap = (con,size)
            elif paper_tap[1] <= size:
                paper_tap = (con,size)
            
        #直線近似して描画
        epsilon = 0.1*cv2.arcLength(paper_tap[0],True)
        paper_corners= cv2.approxPolyDP(paper_tap[0],epsilon,True)#紙の頂点座標
        fix_con = np.array([[[0,0]],[[x,0]],[[x,y]],[[0,y]]], dtype="int32")#整形後のサイズ

        M = cv2.getPerspectiveTransform(np.float32(paper_corners),np.float32(fix_con))#変換行列の生成
        img_trans = cv2.warpPerspective(img,M,(x,y))#変換
        img_rotate = cv2.rotate(img_trans, cv2.ROTATE_90_COUNTERCLOCKWISE)#反時計回りに回転
        return img_rotate


#四角の領域
class Ryoiki():
    def __init__(self,img,p_cnt):
        self.img = img
    


#####
class　Paper():
    def __init__(self):
        self.squares = list()

    def add(self,square):#スクエアを追加する
        for sq in self.squares:
            if sq.is_same(square):
                return 0
        self.squares.append(square)
    
    def count(self):
        return len(self.squares)
    
    #ndarrayを返す
    def arr_ori(self):
        return np.stack([sq.points_ori for sq in self.squares])
    
    #label付けを行う
    def set_label(self):
        for sq in self.squares:
            if 1300<sq.get_lu_point()[0]<1400:
                sq.set_label("example_1")
            elif 1000<sq.get_lu_point()[0]<1100:
                sq.set_label("example_2")
            elif 100<sq.get_lu_point()[0]<150:
                if sq.get_lu_point()[1]< 650:
                    sq.set_label("guide_1")
                else:
                    sq.set_label("draw_1")
            elif 400<sq.get_lu_point()[0]<500:
                if sq.get_lu_point()[1]<650:
                    sq.set_label("guide_1")
                else:
                    sq.set_label("draw_1")
            else:
                print("error")
    
    def get_draws(self):
        return [sq for sq in self.squares if "draw" in sq.get_label()]
            
    # 画像上で検出された一連の正方形を返します。
    def findSquares(image, lower_thre=66000, upper_thre = 71000,sq_num = 16):

        gray0 = np.zeros(image.shape[:2], dtype=np.uint8)

        # down-scale and upscale the image to filter out the noise
        rows, cols, _channels = map(int, image.shape)
        pyr = cv2.pyrDown(image, dstsize=(cols//2, rows//2))
        timg = cv2.pyrUp(pyr, dstsize=(cols, rows))
        # 画像のBGRの色平面で正方形を見つける ( 0で固定した)
        cv2.mixChannels([timg], [gray0], (0, 0))
        
        for l in range(0, 5): # いくつかのしきい値レベルを試す
            print("==== new l ======")
            sqs = ImageSquares()
            if l == 0:
                gray = cv2.Canny(gray0,50, 5)
                gray = cv2.dilate(gray, None)#Canny出力を拡張して、エッジセグメント間の潜在的な穴を削除します
            else:
                gray[gray0 >= (l+1)*255/5] = 0
                gray[gray0 < (l+1)*255/5] = 255
            contours, _ = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for i, cnt in enumerate(contours):
                arclen = cv2.arcLength(cnt, True)# 輪郭の周囲を取得
                approx = cv2.approxPolyDP(cnt, arclen*0.02, True)# 輪郭の近似
                area = abs(cv2.contourArea(approx))# 面積
                if approx.shape[0] == 4 and upper_thre > area > lower_thre and cv2.isContourConvex(approx) :#（ノイズの多い輪郭をフィルターで除去するため）凸性(isContourConvex)になります。
                    maxCosine = 0
                    for j in range(2, 5): # ジョイントエッジ間の角度の最大コサインを見つけます
                        cosine = abs(angle(approx[j%4], approx[j-2], approx[j-1]))
                        maxCosine = max(maxCosine, cosine)
                    if maxCosine < 0.1 :# すべての角度の余弦定理が小さい場合（すべての角度が約90度）、
                        sq = Square(approx)
                        sqs.add(sq)
            if sq_num == sqs.count():
                return sqs


class Char():
    def __init__(self,sq):
        self.sq = sq
    
    def 

class Mizu(Char):
    pass
class Sho(Char):
    pass
