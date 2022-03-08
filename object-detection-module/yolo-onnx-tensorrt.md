# YOLO ONNX TensorRT

This solution is to run the darknet trained model using TensorRT runtime.

{% hint style="info" %}
If you want to read more about TensorRT check [TensorRT](../notes/tensorrt.md) section in notes section.
{% endhint %}

{% hint style="info" %}
If you want to read more about ONNX check [ONNX](../notes/onnx.md) section in notes section.
{% endhint %}

## Downloading the code

Download the code using following command&#x20;

```
git clone https://github.com/sleepingsaint/yolov3_onnx_tensorrt
```

## Installing Dependencies

Install all the requirements&#x20;

```
pip install -r requirements.txt
```

## Convert the darknet model to ONNX&#x20;

```
python3 yolov3_to_onnx.py -c <path to config file> -w <path to weight file> -o <path to output onnx model>
```

## Building TensorRT engine

Build a TensorRT engine from the generated ONNX file and run inference on a sample image&#x20;

```
python3 onnx_to_tensorrt.py -o <path to onnx model> -e <path to the engine file> -i <path to the input video file> -f <number of frames to run the script> -s <path to save the result>
```
