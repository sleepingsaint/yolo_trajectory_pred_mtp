# TensorRT

## Introduction

TensorRT is a library developed by NVIDIA for faster inference on NVIDIA graphics processing units (GPUs). TensorRT is built on CUDA, NVIDIA's parallel programming model. It can give around 4 to 5 times faster inference on many real-time services and embedded applications.

TensorRT provides different optimization techniques to speed up the inference, like combining layers, optimized kernel selection, precision calibration etc.

## Working of TensorRT

![Source:  developer.nvida.com ](https://lh4.googleusercontent.com/1cHpJG5hKFPH5TexNlLlq030IERQczqSErR-quCN6OJE\_By31T6MvYKGA3PTz3Sw1N9y2YeGN\_2dEjiq5vSXJwl7mMMXA1dHY1tdIT9dYQdZXzt0zIT2KZKt9gEq79yIMN9Y9dvK)

TensorRT takes a trained model as an input (model can be trained on any supported TensorRT frameworks) and parses the model using the corresponding framework parser and applies optimizations to the model and generates a highly efficient inference engine file, which can be run using the tensorRT engine.&#x20;

The above image shows the working of TensorRT.&#x20;

1. In the first step we are training the model using our desired framework eg. PyTorch, Tensorflow, Keras etc.
2. After training we converted the model to ONNX format.&#x20;
3. Then TensorRT using the ONNX parser, parses the ONNX model
4. TensorRT applies suitable optimizations on the parsed model. Then it creates a serialized TensorRT file.&#x20;
5. Finally the generated TensorRT file can be run using the TensorRT Runtime engine.

## Optimizations of TensorRT

![Source: developer.nvida.com](https://lh4.googleusercontent.com/gpRVXEj97SV-0Dd69yHYSI61aLEWLqm5pMeOoOXs6\_XPNVaXBrLWP72zKKQhTR1QHVZeaN\_0r0l1y1e9s5TAZ2KbR0cmFHMCjYeQSixw6nCBFGrsSrp7W5Tw8xGo401rsWQaPf4B)

The above image represents the 3 rd step, i.e. applying optimizations to the parsed model.

### **Precision Calibration**

This step helps in reducing the precision point of the parameters and functions used in the model, which not only reduces the latency but also decreases the model file size.&#x20;

As shown in the below image It converts from 32 bit precision to 8 bit precision by scaling down to respective positions. The INT\_MIN becomes -127, and INT\_MAX becomes 127

![Source: medium.com/@abhaychaturvedi\_72055](https://lh3.googleusercontent.com/1KbyU80oGXp2TlCY5wn2GAaXi2M2ajsGkhtWEVt\_hdgu6wioNO9MH5a832NqAsRvJuT87veVdH9X0OVXpu0MCwGS5Sujog7Akjf0ZfxqJxlYhZZ\_6GtFgGInrX4AYIWunZqQFsLC)

### Layers and Tensor Fusion

While executing a graph by any deep learning framework a similar computation needs to be performed routinely. So, to overcome this TensorRT uses layer and tensor fusion to optimize the GPU memory and bandwidth by fusing nodes in a kernel vertically or horizontally(or both), which reduces the overhead and the cost of reading and writing the tensor data for each layer.

![Source: medium.com/@abhaychaturvedi\_72055](https://lh3.googleusercontent.com/4dOL50wHQD92tWDaKH3WjJBtuwIPQWrvZZkpwBa37lH173Il\_wV1cylUnlE3QNn2VLc2ZdyjFW4eUH7vhivC\_tG9EgV73O3f\_I4I61GDDEhBJzprkGtUtbRgdLYUw-Ql9psQoXaS)

In the above image 1x1 convolution blocks fuses horizontally to compute the value once instead of calculating multiple times, also fuses vertically and performs the CUDA optimized matrix operations on the fused layers.

### Kernel Auto Tuning

While optimizing models, there is some kernel specific optimization which can be performed during the process, like algorithms, batch sizes based on the target GPU platform.

### Dynamic Tensor Memory

TensorRT improves memory reuse by allocating memory to tensor only for the duration of its usage. It helps in reducing the memory footprints and avoiding allocation overhead for fast and efficient execution.

### Multiple Stream Execution

TensorRT is designed to process multiple input streams in parallel. This is basically Nvidia’s CUDA stream.
