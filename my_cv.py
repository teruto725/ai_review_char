import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

#影を消す
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
def display_gray(img,output_file_path="tmp.jpg"):
    cv2.imwrite(output_file_path, img)
    plt.imshow(plt.imread(output_file_path))
    plt.axis('off')
    plt.gray()
    plt.show()

#カラー画像を表示する
def display_color(img,output_file_path="tmp.jpg"):
    cv2.imwrite(output_file_path, img)
    plt.imshow(plt.imread(output_file_path))
    plt.axis('off')
    plt.show()

def display_con(img,con,output_file_path="tmp.jpg"):
    im_con = img.copy()
    cv2.drawContours(im_con, con, -1, (0,255,0), 3)
    display_color(im_con)

#threshold以上の画素値を白にする
def threshold(img,threshold):
    ret, img_thresh = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)#100以上の画素255
    return img_thresh

#グレースケール化
def gray(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img_gray    

#境界値を取り出す
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

#pt1とpt2の角度算出する
def angle(pt1, pt2, pt0) -> float:
    dx1 = float(pt1[0,0] - pt0[0,0])
    dy1 = float(pt1[0,1] - pt0[0,1])
    dx2 = float(pt2[0,0] - pt0[0,0])
    dy2 = float(pt2[0,1] - pt0[0,1])
    v = math.sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) )
    return (dx1*dx2 + dy1*dy2)/ v