# 各モジュールのインポート
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, concatenate, Dropout
from keras.models import Model

# U-Net
def unet_model():
    input_img = Input(shape=(128, 128, 1))
    enc1 = Conv2D(64, kernel_size=3, strides=1, activation="relu", padding="same", kernel_initializer="he_normal")(input_img)
    enc1 = Conv2D(64, kernel_size=3, strides=1, activation="relu", padding="same", kernel_initializer="he_normal")(enc1)

    down1 = MaxPooling2D(pool_size=2, strides=2)(enc1) 
    enc2 = Conv2D(128, kernel_size=3, strides=1, activation="relu", padding="same", kernel_initializer="he_normal")(down1)
    enc2 = Conv2D(128, kernel_size=3, strides=1, activation="relu", padding="same", kernel_initializer="he_normal")(enc2)
    
    down2 = MaxPooling2D(pool_size=2, strides=2)(enc2)
    enc3 = Conv2D(256, kernel_size=3, strides=1, activation="relu", padding="same", kernel_initializer="he_normal")(down2)
    enc3 = Conv2D(256, kernel_size=3, strides=1, activation="relu", padding="same", kernel_initializer="he_normal")(enc3)
    enc3 = Dropout(0.5)(enc3)
    
    down3 = MaxPooling2D(pool_size=2, strides=2)(enc3)
    enc4 = Conv2D(512, kernel_size=3, strides=1, activation="relu", padding="same", kernel_initializer="he_normal")(down3)
    enc4 = Conv2D(512, kernel_size=3, strides=1, activation="relu", padding="same", kernel_initializer="he_normal")(enc4)
    enc4 = Dropout(0.5)(enc4)
    
    up3 = UpSampling2D(size=2)(enc4)
    dec3 = BatchNormalization()(up3)
    dec3 = concatenate([dec3, enc3], axis=-1)
    dec3 = Conv2D(256, kernel_size=3, strides=1, activation="relu", padding="same", kernel_initializer="he_normal")(dec3)
    dec3 = Conv2D(256, kernel_size=3, strides=1, activation="relu", padding="same", kernel_initializer="he_normal")(dec3)

    up2 = UpSampling2D(size=2)(dec3)
    dec2 = BatchNormalization()(up2)
    dec2 = concatenate([dec2, enc2], axis=-1)
    dec2 = Conv2D(128, kernel_size=3, strides=1, activation="relu", padding="same", kernel_initializer="he_normal")(dec2)
    dec2 = Conv2D(128, kernel_size=3, strides=1, activation="relu", padding="same", kernel_initializer="he_normal")(dec2)
    
    up1 = UpSampling2D(size=2)(dec2)
    dec1 = BatchNormalization()(up1)
    dec1 = concatenate([dec1, enc1], axis=-1)
    dec1 = Conv2D(64, kernel_size=3, strides=1, activation="relu", padding="same", kernel_initializer="he_normal")(dec1)
    dec1 = Conv2D(64, kernel_size=3, strides=1, activation="relu", padding="same", kernel_initializer="he_normal")(dec1)
    dec1 = Conv2D(2, kernel_size=3, strides=1, activation="relu", padding="same")(dec1)
    dec1 = Conv2D(1, kernel_size=1, strides=1, activation="sigmoid")(dec1)
    
    model = Model(input=input_img, output=dec1)
    
    return model

model = unet_model()

#ネットワークを表示
print(model.summary())