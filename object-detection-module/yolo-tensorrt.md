# YOLO TensorRT

In this implementation, we don't convert the YOLO model to ONNX specification, but we directly tensorRT engine using its primitives by parsing the YOLO config file.

## Setup

We can install TensorRT by following the documentation provided by Nvidia or else we can use nvidia docker container which comes with TensorRT preinstalled.

```
docker pull nvcr.io/nvidia/tensorrt:21.07-py3
```

Clone the repo using the following command

```
git clone https://github.com/sleepingsaint/yolo-tensorrt-cpp.git
```

Start the container

```
nvidia-docker run --rm -it -v yolo-tensorrt:/workspace/yolo-tensorrt nvcr.io/nvidia/tensorrt:21.07-py3 bash
```

Compile the code

```
mkdir build && cd build && cmake .. && make && ./custom-detector -v <path to video file> -c <path to config file> -n <path to name files> -w <path to weights files>                    
```



Benefit with this implementation is that it is written in C++ and also there is no overhead of YOLO to ONNX conversion, hence gives faster inference speed compared to YOLO ONNX TensorRT implementation.
