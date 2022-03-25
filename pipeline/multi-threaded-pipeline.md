# Multi Threaded Pipeline

In this implementation, the detector and predictor are both implemented using ONNX runtime. The trajectory prediction for each object is done on different threads using Thread Pool Executor in python.

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

* Run the download.py script and everything required will be downloaded and placed in corresponding directories

```
python download.py
```

### Run the pipeline

```
python3 threaded_pipeline.py -m <path to yolo onnx model> -i <input video file> -f <frame count>                    
```

For more options run

```
python3 threaded_pipeline.py -h
```
