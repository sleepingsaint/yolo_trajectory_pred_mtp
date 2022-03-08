# Training Custom YOLOv4 Model

## Setup

### Create Custom config file

1. Create file `yolo-obj.cfg` with the same content as in `yolov4-custom.cfg` (or copy `yolov4-custom.cfg` to `yolo-obj.cfg)` and
   1. change line batch to [`batch=64`](https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L3)
   2. change line subdivisions to [`subdivisions=16`](https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L4)
   3. change line max\_batches to (`classes*2000`, but not less than number of training images and not less than `6000`), f.e. [`max_batches=6000`](https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L20) if you train for 3 classes
   4. change line steps to 80% and 90% of max\_batches, f.e. [`steps=4800,5400`](https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L22)
   5. set network size `width=416 height=416` or any value multiple of 32: https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L8-L9
   6. change line `classes=80` to your number of objects in each of 3 `[yolo]`-layers:
      1. https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L610
      2. https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L696
      3. https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L783
   7. change \[`filters=255`] to filters=(classes + 5)x3 in the 3 `[convolutional]` before each `[yolo]` layer, keep in mind that it only has to be the last `[convolutional]` before each of the `[yolo]` layers.
      * https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L603
      * https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L689
      * https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L776
   8. when using [`[Gaussian_yolo]`](https://github.com/AlexeyAB/darknet/blob/6e5bdf1282ad6b06ed0e962c3f5be67cf63d96dc/cfg/Gaussian\_yolov3\_BDD.cfg#L608) layers, change \[`filters=57`] filters=(classes + 9)x3 in the 3 `[convolutional]` before each `[Gaussian_yolo]` layer
      * https://github.com/AlexeyAB/darknet/blob/6e5bdf1282ad6b06ed0e962c3f5be67cf63d96dc/cfg/Gaussian\_yolov3\_BDD.cfg#L604
      * https://github.com/AlexeyAB/darknet/blob/6e5bdf1282ad6b06ed0e962c3f5be67cf63d96dc/cfg/Gaussian\_yolov3\_BDD.cfg#L696
      * https://github.com/AlexeyAB/darknet/blob/6e5bdf1282ad6b06ed0e962c3f5be67cf63d96dc/cfg/Gaussian\_yolov3\_BDD.cfg#L789
   9. So if `classes=1` then should be `filters=18`. If `classes=2` then write `filters=21`.&#x20;

{% hint style="danger" %}
Do not write in the cfg-file: filters=(classes + 5)x3
{% endhint %}

(Generally `filters` depends on the `classes`, `coords` and number of `mask`s, i.e. filters=`(classes + coords + 1)*<number of mask>`, where `mask` is indices of anchors. If `mask` is absence, then filters=`(classes + coords + 1)*num`)

So for example, for 2 objects, your file `yolo-obj.cfg` should differ from `yolov4-custom.cfg` in such lines in each of **3** \[yolo]-layers:

```ini
[convolutional]
filters=21

[region]
classes=2
```

### Creating obj.names file

This file holds all the objects names which YOLO will detect

Create file `obj.names` in the directory `build/darknet/data`, with objects names - each in new line

### Creating obj.data file

Create file `obj.data` in the directory `build/darknet/data`, containing (where **classes = number of objects**):

```ini
classes = 2
train  = data/train.txt
valid  = data/test.txt
names = data/obj.names
backup = backup/
```

### Arranging the dataset

1. Download the dataset from roboflow and unzip it.
2. Put image-files (.jpg) of your objects in the directory `build/darknet/data/obj`
3. Create file `train.txt` in directory `build/darknet/data/`, with filenames of your images, each filename in new line, with path relative to `darknet.exe`, for example containing:

```csv
data/obj/img1.jpg
data/obj/img2.jpg
data/obj/img3.jpg
```

### Download pretrained weights

1. Download pre-trained weights for the convolutional layers and put to the directory `build/darknet/`
   * for `yolov4.cfg`, `yolov4-custom.cfg` (162 MB): [yolov4.conv.137](https://github.com/AlexeyAB/darknet/releases/download/darknet\_yolo\_v3\_optimal/yolov4.conv.137) (Google drive mirror [yolov4.conv.137](https://drive.google.com/open?id=1JKF-bdIklxOOVy-2Cr5qdvjgGpmGfcbp) )
   * for `yolov4-tiny.cfg`, `yolov4-tiny-3l.cfg`, `yolov4-tiny-custom.cfg` (19 MB): [yolov4-tiny.conv.29](https://github.com/AlexeyAB/darknet/releases/download/darknet\_yolo\_v4\_pre/yolov4-tiny.conv.29)
   * for `csresnext50-panet-spp.cfg` (133 MB): [csresnext50-panet-spp.conv.112](https://drive.google.com/file/d/16yMYCLQTY\_oDlCIZPfn\_sab6KD3zgzGq/view?usp=sharing)
   * for `yolov3.cfg, yolov3-spp.cfg` (154 MB): [darknet53.conv.74](https://pjreddie.com/media/files/darknet53.conv.74)
   * for `yolov3-tiny-prn.cfg , yolov3-tiny.cfg` (6 MB): [yolov3-tiny.conv.11](https://drive.google.com/file/d/18v36esoXCh-PsOKwyP2GWrpYDptDY8Zf/view?usp=sharing)
   * for `enet-coco.cfg (EfficientNetB0-Yolov3)` (14 MB): [enetb0-coco.conv.132](https://drive.google.com/file/d/1uhh3D6RSn0ekgmsaTcl-ZW53WBaUDo6j/view?usp=sharing)

### Start training

Start training by using the command line: `darknet.exe detector train data/obj.data yolo-obj.cfg yolov4.conv.137`

To train on Linux use command: `./darknet detector train data/obj.data yolo-obj.cfg yolov4.conv.137` (just use `./darknet` instead of `darknet.exe`)

* (file `yolo-obj_last.weights` will be saved to the `build\darknet\x64\backup\` for each 100 iterations)
* (file `yolo-obj_xxxx.weights` will be saved to the `build\darknet\x64\backup\` for each 1000 iterations)
* (to disable Loss-Window use `darknet.exe detector train data/obj.data yolo-obj.cfg yolov4.conv.137 -dont_show`, if you train on computer without monitor like a cloud Amazon EC2)
* (to see the mAP & Loss-chart during training on remote server without GUI, use command `darknet.exe detector train data/obj.data yolo-obj.cfg yolov4.conv.137 -dont_show -mjpeg_port 8090 -map` then open URL `http://ip-address:8090` in Chrome/Firefox browser)

### Training with mAP

For training with mAP (mean average precisions) calculation for each 4 Epochs (set `valid=valid.txt` or `train.txt` in `obj.data` file) and run: `darknet.exe detector train data/obj.data yolo-obj.cfg yolov4.conv.137 -map`

### Extracting the trained weights

After training is complete - get result `yolo-obj_final.weights` from path `build\darknet\x64\backup\`

* After each 1000 iterations you can stop and later start training from this point. For example, after 2000 iterations you can stop training, and later just start training using: `darknet.exe detector train data/obj.data yolo-obj.cfg backup\yolo-obj_2000.weights`

(in the original repository https://github.com/pjreddie/darknet the weights-file is saved only once every 10 000 iterations `if(iterations > 1000)`)

* Also you can get result earlier than all 45000 iterations.

{% hint style="danger" %}
If during training you see `nan` values for `avg` (loss) field - then training goes wrong, but if `nan` is in some other lines - then training goes well.
{% endhint %}

{% hint style="warning" %}
If you changed width= or height= in your cfg-file, then new width and height must be divisible by 32.
{% endhint %}

{% hint style="info" %}
if error `Out of memory` occurs then in `.cfg`-file you should increase `subdivisions=16`, 32 or 64
{% endhint %}

## Testing the trained model

After training use such command for detection: `darknet.exe detector test data/obj.data yolo-obj.cfg yolo-obj_8000.weights`

## How to train tiny-yolo (to detect your custom objects)

Do all the same steps as for the full yolo model as described above. With the exception of:

* Download file with the first 29-convolutional layers of yolov4-tiny: https://github.com/AlexeyAB/darknet/releases/download/darknet\_yolo\_v4\_pre/yolov4-tiny.conv.29 (Or get this file from yolov4-tiny.weights file by using command: `darknet.exe partial cfg/yolov4-tiny-custom.cfg yolov4-tiny.weights yolov4-tiny.conv.29 29`
* Make your custom model `yolov4-tiny-obj.cfg` based on `cfg/yolov4-tiny-custom.cfg` instead of `yolov4.cfg`
* Start training: `darknet.exe detector train data/obj.data yolov4-tiny-obj.cfg yolov4-tiny.conv.29`
