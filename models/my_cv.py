import numpy as np
import matplotlib.pyplot as plt
import cv2
import math


#白黒画像を表示する
def display_gray(img,output_file_path="./tmp/tmp.png"):
    cv2.imwrite(output_file_path, img)
    plt.imshow(plt.imread(output_file_path))
    plt.axis('off')
    plt.gray()
    plt.show()

#カラー画像を表示する
def display_color(img,output_file_path="./tmp/tmp.png"):
    cv2.imwrite(output_file_path, img)
    plt.imshow(plt.imread(output_file_path))
    plt.axis('off')
    plt.show()

#displayのプロっと
def display_con(img,con,output_file_path="./tmp/tmp.png"):
    im_con = img.copy()
    im_con = cv2.drawContours(im_con, con, -1, (0,255,0), 30)
    display_color(im_con,output_file_path)

#座標のplot
def display_point(img,point,output_file_path="./tmp/tmp.png"):
    im_point = img.copy() // 2 + 128
    im_point = cv2.circle(im_point,tuple(point),3,(100,0,100),thickness=-1)
    display_color(im_point,output_file_path)

# approxの描画
def display_approx(img,approx,output_file_path="./tmp/tmp.png"):
    img_copy = img.copy()
    cv2.polylines(img_copy, [approx.reshape(-1,2)], True, (0,0,255), thickness=1, lineType=cv2.LINE_8)
    for i,app in enumerate(approx):
        cv2.circle(img_copy, (app[0][0],app[0][1]), 3, (0, 0+35*i, 0), thickness=-1)
    display_color(img_copy,output_file_path)


#threshold以上の画素値を白にする
def threshold(img,threshold):
    ret, img_thresh = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)#100以上の画素255
    return img_thresh

#グレースケール化
def gray(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img_gray    




#カラー画像を受け取って紙を切り取って返す 紙の向きに注意
def cutting_paper(img):
    x = 2000 #切り取り後のx座標
    y = int(x*1.415) #切り取り後のy座標
    img_copy = np.copy(img)
    img_gray = gray(img_copy)#グレイスケール
    #２値化
    img_th = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                 cv2.THRESH_BINARY, 39, 2)#影を考慮した二値化
    #display_gray(img_th)
    img_th = cv2.bitwise_not(img_th)
    #ノイズ除去
    img_noise = cv2.fastNlMeansDenoising(img_th,h=30)
    #display_gray(img_noise)

    #輪郭抽出
    contours, hierarchy = cv2.findContours(img_noise, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    print(np.shape(contours))
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
    print(np.shape(paper_tap[0]))
    #直線近似して描画
    #display_con(img,paper_tap[0])
    epsilon = 0.1*cv2.arcLength(paper_tap[0],True)
    paper_corners= cv2.approxPolyDP(paper_tap[0],epsilon,True)#紙の頂点座標
    paper_corners = arrange_approx_points(paper_corners)# ソートする
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


#重心を求める
def get_centroid(cnt):
    cnt = np.array(cnt)
    m = cv2.moments(cnt)
    cx = int(m['m10']/m['m00'])
    cy = int(m['m01']/m['m00'])
    return (cx,cy)

#point間の距離を出す
def distance(point1,point2):
    return abs(point1[0]-point2[0])+abs(point1[1]-point2[1])

#2点の中間距離を算出する
def mid_point(point1,point2):
    return (int((point1[0]+point2[0])/2),int((point1[1]+point2[1])/2))

#2値化画像を受け取ってフィルタ処理した2値画像を返す　線内のノイズを削除する
def mor_clear_filter(img_thresh):
    kernel = np.ones((3,3),np.uint8)
    closing = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel)
    return closing

#approxを時計回りに修正する #TODO ここを書き換える

def arrange_approx_points(points):
    stack = list()
    s_idx = np.argsort([ p[0][0] + p[0][1] for p in points])#左上から右下
    s2_idx = np.argsort([ p[0][0] - p[0][1] for p in points])#左下から右上
    stack.append(points[s_idx[0]])
    stack.append(points[s2_idx[3]])
    stack.append(points[s_idx[3]])
    stack.append(points[s2_idx[0]])
    #print(np.stack(stack))
    return  np.stack(stack)

#スコアの上限値と下限値, 値の上限値、下限値、値を入力してscoreを返す
def get_mm_score(s_max,s_min,v_max,v_min):
    pass