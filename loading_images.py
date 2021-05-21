import numpy as np
import os

IMAGE_SIZE = 64
SEGMENTATION_TARGET_LABEL = 255

# ディレクトリ内の画像を読み込む
# inputpath: ディレクトリ文字列, imagesize: 画像サイズ, type_color: Gray
def load_images(inputpath, imagesize, type_color):
    imglist = []
    exclude_prefixes = ('__', '.') 

    for root, dirs, files in os.walk(inputpath):
        for fn in sorted(files):
             dirs[:] = [dir for dir in dirs if not dir.startswith(exclude_prefixes)]
             files[:] = [file for file in files if not file.startswith(exclude_prefixes)]

        for fn in sorted(files):
            bn, ext = os.path.splitext(fn)
            if ext not in [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"]:
                continue

            filename = os.path.join(root, fn)
            
            if type_color == 'Gray':
                # グレースケール画像の場合
                testimage = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
                # サイズ変更
                height, width = testimage.shape[:2]
                testimage = cv2.resize(testimage, (imagesize, imagesize), interpolation = cv2.INTER_AREA)  
                # チャンネルの次元がないので1次元追加する
                testimage = np.asarray([testimage], dtype=np.float64)
                testimage = np.asarray(testimage, dtype=np.float64).reshape((1, imagesize, imagesize))
                # 高さ，幅，チャンネルに入れ替え．data_format="channels_last"を使うとき必要
                testimage = testimage.transpose(1, 2, 0)

            imglist.append(testimage)
    imgsdata = np.asarray(imglist, dtype=np.float32)

    return imgsdata # 画像リストとファイル名のリストを返す

# 指定ディレクトリに画像リストの画像を保存する
# savepath: ディレクトリ文字列, filenamelist: ファイル名リスト, imagelist: 画像リスト
def save_images(savepath, filenamelist, imagelist):
    for i, fn in enumerate(filenamelist):
        filename = os.path.join(savepath, fn)
        testimage = imagelist[i]
        testimage = np.delete(testimage, 2, 1) 
        cv2.imwrite(filename, testimage)

# 目的のラベル番号: orgnum のみを outnum にして，他はゼロにする関数
def convlabelnum2desire(label, orgnum, outnum, w, h):
    for case in range(0, len(label)):
        for i in range(0, h):
            for j in range(0, w):
                if(label[case][i][j][0] != orgnum):
                    label[case][i][j][0] = 0.0
                else:
                    label[case][i][j][0] = outnum

# データ準備
# 画像読み込み
# training用の原画像とラベル画像読み込み
image_train = load_images('', IMAGE_SIZE, 'Gray')
label_train = load_images('', IMAGE_SIZE, 'Gray')
# test用の原画像とラベル画像読み込み
image_test  = load_images('', IMAGE_SIZE, 'Gray')
label_test  = load_images('', IMAGE_SIZE, 'Gray')

# ラベル画像は指定した値 SEGMENTATION_TARGET_LABEL だけ残し他は0にする
convlabelnum2desire(label_train, SEGMENTATION_TARGET_LABEL, 255, IMAGE_SIZE, IMAGE_SIZE)
convlabelnum2desire(label_test, SEGMENTATION_TARGET_LABEL, 255, IMAGE_SIZE, IMAGE_SIZE)

# 画素値0-1正規化
image_train /= np.max(image_train)
label_train /= np.max(label_train)
image_test /= np.max(image_test)
label_test /= np.max(label_test)