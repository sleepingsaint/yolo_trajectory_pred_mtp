# PyTorch Detector - ONNX Trajectory

This version of pipeline consists of object detector based on pytorch Yolo and trajectory prediction using ONNX runtime.

In this implementation the yolo model has been parsed into PyTorch primitives and used for detection.

## Setup

### Clone the repo

Clone or download the repository and extract it.

```
git clone https://github.com/sleepingsaint/yolo_trajectory_pred_mtp.git
```

### Install the dependecies

Run the below command to install all the dependecies and required libraries

```
pip3 install -r requirements.txt
```

### Setting up the project

Run the download.py script and everything required will be downloaded and placed in corresponding directories

```
python download.py
```

### Run the pipeline

```
python3 pytorch_detector_onnx_predictor.py -i <input video file> -f <frame count>                    
```

For more options run&#x20;

```
python3 pytorch_detector_onnx_predictor.py -h
```
