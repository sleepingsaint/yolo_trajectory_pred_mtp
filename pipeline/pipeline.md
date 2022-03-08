# Pipeline



Pipeline contains the code with object detection, trajectory prediction and object collision modules coupled in single codebase.

## Downloading&#x20;

Clone this repo using the following command

```bash
git clone https://github.com/sleepingsaint/yolo_trajectory_pred_mtp.git
```

## Setting up the project

Run the download.py script and everything required will be downloaded and placed in corresponding directories

```
python download.py
```

## Manual Setup

* download objection detection weights from the following link and save it in detector/YOLOV3/weight/obj\_detection.weights
  * [https://drive.google.com/file/d/1C3Kqqu9gDXNNXr5WDpmhGTr-UaQ2ckqQ/view?usp=sharing](https://drive.google.com/file/d/1C3Kqqu9gDXNNXr5WDpmhGTr-UaQ2ckqQ/view?usp=sharing)
* download weights for trajectory prediction of arm from the following link and save it in Trajectory/models/Individual/traj\_arm.pth
  * [https://drive.google.com/file/d/1VfRVvc-7EowI540S0\_6FxJOMVg9XyXef/view?usp=sharing](https://drive.google.com/file/d/1VfRVvc-7EowI540S0\_6FxJOMVg9XyXef/view?usp=sharing)
* download weights for trajectory prediction of end effector from the following link and save it in Trajectory/models/Individual/traj\_endeffector.pth
  * [https://drive.google.com/file/d/1t\_qok3BNNHN6EK\_Uw3WiXrfGtLfQJpCs/view?usp=sharing](https://drive.google.com/file/d/1t\_qok3BNNHN6EK\_Uw3WiXrfGtLfQJpCs/view?usp=sharing)
* download weights for trajectory prediction of probe from the following link and save it in Trajectory/models/Individual/traj\_probe.pth
  * [https://drive.google.com/file/d/1SEZGUvLB2gfVwA-yGTBlF5k3YDiOzYWj/view?usp=sharing](https://drive.google.com/file/d/1SEZGUvLB2gfVwA-yGTBlF5k3YDiOzYWj/view?usp=sharing)
* download weights for trajectory prediction of person from the following link and save it in Trajectory/models/Individual/00013.pth
  * [https://drive.google.com/file/d/1p8vo9rig9Q0i0WLVudQBgzkdjtJXcDiS/view?usp=sharing](https://drive.google.com/file/d/1p8vo9rig9Q0i0WLVudQBgzkdjtJXcDiS/view?usp=sharing)
* Download the pretrained checkpoint file for deepsort and save it in deep\_sort/deep/checkpoint/
  * [https://drive.google.com/file/d/1MlXnCSjD5yOfxnnJMkruE0rCgLCMZJlB/view?usp=sharing](https://drive.google.com/file/d/1MlXnCSjD5yOfxnnJMkruE0rCgLCMZJlB/view?usp=sharing)
* download the testing video for running inference and save it in data\test\_video.mp4
  * [https://drive.google.com/file/d/13mVaJTsJ7rN-Bz5KtsS20dX1TZFAHLkU/view?usp=sharing](https://drive.google.com/file/d/13mVaJTsJ7rN-Bz5KtsS20dX1TZFAHLkU/view?usp=sharing)

## Adding the class names

Change the class names in data/names file if you used custom object for training

## Converting the Darknet model to onnx

Run the following command to convert the trained darknet model to ONNX format.

```
python darknet2onnx.py -c data/yolov3TATA608.cfg -w data/yolov3TATA608_final.weights -i data/test_image.jpg
```

* Here test image is for checking if the model is properly converted or not.
* It will create a converted ONNX model and predictions\_onnx.jpg image (predictions from the converted ONNX model) in root directory.

## Running the pipeline

Run the below command to run inference on a video file



```
python onnx_traj_every_frame.py -m <path to onnx model> -i <path to input video file> -o <path to save the output file> -f <number of frames to run inference on> -c <number of classes the model is trained on>
```

### Arguments Details

\-h, --help show this help message and exit

\-m MODEL, --model MODEL Path to the model file

\-i INPUT, --input INPUT Path to the video file

\-o OUTPUT, --output OUTPUT Path to the output video

\-f FRAME\_COUNT, --frame\_count FRAME\_COUNT Number of frames to run the video

\-q FRAME\_FREQ, --frame\_freq FRAME\_FREQ Frequency of frame to run the trajectory prediction on

\-v, --verbose Enable more details

\-c NUM\_CLASSES, --num\_classes NUM\_CLASSES Number of classes model trained on example:

#### Examples:

* Running for the first 100 frames

```
python onnx_traj_every_frame.py -m data/yolov_converted.onnx -i data/test_video.mp4 -f 100 -c 5 -o tmp_result.mp4
```

* Running for the entire video

```
python onnx_traj_every_frame.py -m data/yolo_converted.onnx -i data/test_video.mp4 -c 5 -o tmp_result.mp4
```

## Colab Link

Please refer to this [colab notebook](https://colab.research.google.com/drive/1OvyCCnw0asjzlrOP942BFT-XtbO9FUN9?usp=sharing) to understand the steps better.\
\
