# YOLO - You Only Look Once

Prior detection systems repurpose classifiers or localizers to perform detection. They apply the model to an image at multiple locations and scales. High scoring regions of the image are considered detections.

YOLO use a totally different approach. It apply a single neural network to the full image. This network divides the image into regions and predicts bounding boxes and probabilities for each region. These bounding boxes are weighted by the predicted probabilities.

YOLO has several advantages over classifier-based systems. It looks at the whole image at test time so its predictions are informed by global context in the image. It also makes predictions with a single network evaluation unlike systems like R-CNN which require thousands for a single image. This makes it extremely fast, more than 1000x faster than R-CNN and 100x faster than Fast R-CNN.&#x20;

{% hint style="info" %}
See [paper](https://pjreddie.com/media/files/papers/YOLOv3.pdf) for more details on the full system.
{% endhint %}

## Versions

There are multiple versions of YOLO available. YOLOv1, YOLOv2, YOLOv3 are from the original author **PJ Reddie**. YOLOv4 is next iteration of YOLO family model with improvements like data augmentation steps by **Alexey Bochkovskiy.**&#x20;

There is a unofficial community driven YOLO model called YOLOv5 is available. It is written in python using PyTorch framework.

For this project we have used YOLOv4 and trained it as per the documentation provided in the github repo by Alexey Bochkovskiy.
