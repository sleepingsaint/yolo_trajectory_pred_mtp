# Pipeline

## Outline

![Pipeline block diagram](<.gitbook/assets/MTP Pipeline.drawio.png>)

### Object Detection Module

This module uses state-of-art Object Detection Neural Network called **YOLO.** We have trained our model using the custom dataset we prepared from multiple sources. We used the transfer learning approach to train the model.

We then optimized the inference time of the detection module using different solutions like ONNX runtime, TensorRT.

### Trajectory Prediction Module

This module predicts the future trajectory of the detected objects from the previous object detection module, given the past trajectory locations. For building this module we used state of art neural network model **Transformer**.&#x20;

We generated a dataset based on the ground truth values we got and trained this model. We also the optimized the inference time of this module using ONNX and TensorRT.&#x20;

### Collision Detection Module

Given the predicted future trajectory of objects and computed depth values we predict the possiblity of the collision. We used multiple time synchronised video streams to compute depth values for objects.
