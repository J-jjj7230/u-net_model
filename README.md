# Segmentation using U-Net, a medical image segmentation model

"U-Net" is a model that was released in 2015 (https://arxiv.org/abs/1505.04597).

# overview

## Datasets and Image size
Tha image datasets(labled lung image of chest x-ray images) published by the japanese Society of Radiological Technology is used
(http://imgcom.jsrt.or.jp/minijsrtdb/). <br>
This data has been labeled by professional doctors. <br>
There are 237 images and image size is 128×128 grayscale images. The ratio of training data to evaluate data is  7:3.

## model
<img src="https://user-images.githubusercontent.com/77866776/115371705-2d17c200-a205-11eb-9af3-83445a64b4de.png" width="480px">

In the encoding section, a chest X-ray image is used as the input image, and convolution is performed twice in succession, followed by 2×2 Max Pooling. In the first convolution, the number of filters used is 64, and the kernel size of the filters is specified as 3×3. In other words, 64 different outputs are produced by applying the filter to 64 different 3×3 input images. The stride is set to 1, and the filter is applied and moved in 1 pixel steps. The activation function is specified as ReLU function.

### data conenection
<img src ="https://user-images.githubusercontent.com/77866776/115377476-9d751200-a20a-11eb-888c-89110219aaee.png" width="460px">

The third stage of the encoding section is connected to the third stage of the decoding section before upsampling. The size of the third stage of the encoding section is 32×32, and it holds the feature and position information before pooling. The third stage of the encoding section and the third stage of the upsampling section are concatenated to form a single data set, which enables recovery of the overall positional information while retaining the local features.

## evaluation index
In this study, I used accuracy, Iou, and F1score for segmentation of lung regions. <br>

Accuracy is a measure of the degree of agreement between the estimated result and the answer. <br>

![\begin{align*}
Accuracy=\frac{TP+TN}{TP+FP+TN+FN}
\end{align*}
](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cbegin%7Balign%2A%7D%0AAccuracy%3D%5Cfrac%7BTP%2BTN%7D%7BTP%2BFP%2BTN%2BFN%7D%0A%5Cend%7Balign%2A%7D%0A) <br>

IoU(intersection over union) is a method used in the field of object recognition to evaluate the degree of coincidence of regions, and it is used to evaluate contour leakage and slight overflow in a strict manner. <br>

![\begin{align*}
IoU=\frac{TP}{TP+FP+FN}
\end{align*}
](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cbegin%7Balign%2A%7D%0AIoU%3D%5Cfrac%7BTP%7D%7BTP%2BFP%2BFN%7D%0A%5Cend%7Balign%2A%7D%0A) <br>

The Dice coefficient is a measure of how similar the two sets are, and is used to evaluate whether or not something is being detected. <br>

![\begin{align*}
Dice=\ \frac{2TP}{2TP+FN+FP}
\end{align*}
](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cbegin%7Balign%2A%7D%0ADice%3D%5C+%5Cfrac%7B2TP%7D%7B2TP%2BFN%2BFP%7D%0A%5Cend%7Balign%2A%7D%0A) <br>

## model evaluation
The accuracy was 0.9544, the IoU was 0.7189, and the F1score was 0.9282.








