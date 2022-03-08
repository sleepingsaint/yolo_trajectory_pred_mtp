# ONNX

## Introduction

ONNX (Open Neural Network Exchange) is an open format built to represent machine learning models. ONNX defines a common set of operators - the building blocks of machine learning and deep learning models - and a common file format to enable AI developers to use models with a variety of frameworks, tools, runtimes, and compilers.

## ONNX structure

![ONNX elements](https://lh6.googleusercontent.com/w8friyIHSAck4hZXMCjLoq7SnRwF0d14xnXiHj0Jrcb-rOUcdQxHdVnVfT\_qC4ax4tMEonJ3uDirR-OsbLEhKi8skUll67-StKgrcRKs5GSGXV8wD19huc7la\_LGze-ipdz3t7EO)

ONNX format represents the computation graph of the framework on which the neural network model is trained. The above block diagram shows the components of the ONNX specification.&#x20;

ONNX is an open specification that consists of the following components:&#x20;

* A definition of an extensible computation graph model.&#x20;
* Definitions of standard data types.&#x20;
* Definitions of built-in operators.&#x20;

ONNX operators are of two types:&#x20;

* Primitive Operators
* Functional Operators

## ONNX Runtime

Now we have a onnx model, but we need a way to execute this model and run inference using this and ONNX Runtime is the way to go. ONNX Runtime is a cross-platform inference and training machine-learning accelerator.&#x20;

ONNX Runtime is compatible with different hardware, drivers, and operating systems, and provides optimal performance by leveraging hardware accelerators where applicable alongside graph optimizations and transforms.&#x20;

ONNX Runtime provides execution providers which enables the hardware acceleration for the model like TensorRT, CUDA, Open Vino etc.

![Structure of ONNX working](https://lh5.googleusercontent.com/IviVrtlUBCJt-aYKgF\_CZ10SJUl9tbvUCJhjPo5xSDviIdp3hv8-z\_8xwsfOCLt3pJZxnOF7xWO2X22M7njoBfYn9-dZeHTHBDYf8B1\_BpB4u6bQHxE3K6\_f4EM97OtqBzgdRQlY)

Above diagram shows the flow of execution in ONNX runtime. First the ONNX computation graph is loaded into the memory and then it takes the input, runs it using the corresponding execution provider and gives the results.

In the above image the ONNX runtime is using GPU Execution Provider which may be CUDA, TensorRT etc.
