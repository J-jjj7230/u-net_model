# 評価指標
def true_positive(y_true, y_pred):
    return K.sum(K.cast(K.equal(y_true * y_pred, 1), K.floatx()))

def true_negative(y_true, y_pred):
    return K.sum(K.cast(K.equal(y_true + y_pred, 0), K.floatx()))

def false_positive(y_true, y_pred):
    return K.sum(K.cast(K.less(y_true, y_pred), K.floatx()))

def false_negative(y_true, y_pred):
    return K.sum(K.cast(K.greater(y_true, y_pred), K.floatx()))

#DICE係数を計算する関数
def dice_coef(y_true, y_pred, smooth=1e-9):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

#ロス関数
def dice_coef_loss(y_true, y_pred):
    return (1.0 - dice_coef(y_true, y_pred))

# IoU（評価関数)
def iou_score(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    return (true_positive(y_true, y_pred) / (false_negative(y_true, y_pred)+true_positive(y_true, y_pred)+false_positive(y_true, y_pred)))

#ロス関数
def iou_score_loss(y_true, y_pred):
    return (1.0 - iou_score(y_true, y_pred))

