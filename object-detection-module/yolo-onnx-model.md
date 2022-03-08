# YOLO ONNX Model

To implement any kind of performance optimizations we need a format which is readable by respective implementations.

One such implementation is ONNX specification.

Darknet doesn’t offer conversion modules to convert darknet trained models to ONNX format. I used an online library code which offers conversion scripts.&#x20;

## Downloading

Download the code with following command

```
git clone https://github.com/Tianxiaomo/pytorch-YOLOv4.git
```

## Installing Dependencies

Install ONNX runtime python library

```
pip install onnxruntime or pip install onnxruntime-gpu
```

## Convert and run the model

Run the following command to convert the model and test

```
python demo_darknet2onnx.py
```

This script will convert the darknet config file to pytorch neural network module and then convert to ONNX and run it using ONNX Runtime.
