# Trajectory

## Contents

* [Download the Project](#downloading-the-project)
* [Setting Up the project](#setting-up-the-project)
	* [Manual Setup](#manual-setup)
* [Running the pipeline](#running-the-pipeline)

## Downloading the project

Clone this repo using the following command 

```shell
git clone https://github.com/sleepingsaint/yolo_trajectory_pred_mtp.git
```

## Setting up the project

* Run the download.py script and everything required will be downloaded and placed in corresponding directories

```shell
	python download.py
```

## Manual Setup 

* download objection detection weights from the following link and save it in detector/YOLOV3/weight/obj_detection.weights

	* https://drive.google.com/file/d/1C3Kqqu9gDXNNXr5WDpmhGTr-UaQ2ckqQ/view?usp=sharing 

* download weights for trajectory prediction of arm from the following link and save it in Trajectory/models/Individual/traj_arm.pth
	* https://drive.google.com/file/d/1VfRVvc-7EowI540S0_6FxJOMVg9XyXef/view?usp=sharing 

* download weights for trajectory prediction of end effector from the following link and save it in Trajectory/models/Individual/traj_endeffector.pth 
	* https://drive.google.com/file/d/1t_qok3BNNHN6EK_Uw3WiXrfGtLfQJpCs/view?usp=sharing 

* download weights for trajectory prediction of probe from the following link and save it in Trajectory/models/Individual/traj_probe.pth 
	* https://drive.google.com/file/d/1SEZGUvLB2gfVwA-yGTBlF5k3YDiOzYWj/view?usp=sharing 

* download weights for trajectory prediction of person from the following link and save it in Trajectory/models/Individual/00013.pth 
	* https://drive.google.com/file/d/1p8vo9rig9Q0i0WLVudQBgzkdjtJXcDiS/view?usp=sharing 

* Download the pretrained checkpoint file for deepsort and save it in deep_sort/deep/checkpoint/
	* https://drive.google.com/file/d/1MlXnCSjD5yOfxnnJMkruE0rCgLCMZJlB/view?usp=sharing

* download the testing video for running inference and save it in data\test_video.mp4
	* https://drive.google.com/file/d/13mVaJTsJ7rN-Bz5KtsS20dX1TZFAHLkU/view?usp=sharing


## Adding the class names
Change the class names in data/names file if you used custom object for training

## Converting the Darknet model to onnx
Run the following command to convert the trained darknet model to ONNX format.

```bash
python darknet2onnx.py -c data/yolov3TATA608.cfg -w data/yolov3TATA608_final.weights -i data/test_image.jpg
```

* Here test image is for checking if the model is properly converted or not.
* It will create a converted ONNX model and predictions_onnx.jpg image (predictions from the converted ONNX model) in root directory.

## Running the pipeline

Run the below command to run inference on a video file

```shell
python onnx_traj_every_frame.py -m <path to onnx model> -i <path to input video file> -o <path to save the output file> -f <number of frames to run inference on> -c <number of classes the model is trained on>
```

### Arguments Details
  -h, --help            show this help message and exit

  -m MODEL, --model MODEL Path to the model file

  -i INPUT, --input INPUT Path to the video file
  
  -o OUTPUT, --output OUTPUT Path to the output video

  -f FRAME_COUNT, --frame_count FRAME_COUNT Number of frames to run the video

  -v, --verbose         Enable more details
  
  -c NUM_CLASSES, --num_classes NUM_CLASSES
                        Number of classes model trained on
example:

* Running for the first 100 frames

```python
python onnx_traj_every_frame.py -m data/yolov_converted.onnx -i data/test_video.mp4 -f 100 -c 5 -o tmp_result.mp4
```

* Running for the entire video
```shell
python onnx_traj_every_frame.py -m data/yolo_converted.onnx -i data/test_video.mp4 -c 5 -o tmp_result.mp4
```

### Colab Link

Please refer to this [colab notebook](https://colab.research.google.com/drive/1OvyCCnw0asjzlrOP942BFT-XtbO9FUN9?usp=sharing) to understand the steps better.


### Check out Previous Pipeline

The previous pipeline doesn't contain any optimizations and onnx format conversions. Run the following command to execute the script

```bash
python3 previous_pipeline.py -i <path to the video>
```