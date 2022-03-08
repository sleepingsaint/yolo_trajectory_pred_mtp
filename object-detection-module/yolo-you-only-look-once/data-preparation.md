# Data Preparation

{% hint style="info" %}
If you already have a premade dataset please check the next sections.
{% endhint %}

This page describes about generating ground truths dataset, data augmentation, labelling and converting to YOLO understandable format.&#x20;



We used **Roboflow** to generate, label and preprocess data.

## Roboflow

![https://app.roboflow.com](https://assets.website-files.com/5f6bc60e665f54545a1e52a5/613f691a75c36d203344223d\_open-graph.png)

Roboflow is a web based application which provides multiple features

* Cloud space to store the dataset
* Labelling and annotating tools
* Preprocessing and Augmentation tools
* Generating datasets from videos

## How we generated our datasets?

We uploaded the video data we have to the roboflow cloud and split the video to individual frames by 25 frames per second since the frames in a second won't change much we can skip few frames. We suggest not to go over 25 frames, because the might get overfit to the dataset. We created 5 annotations / labels - End Effector, problance, probe, window, person.&#x20;

After labelling the frames, we generated the multiple sets of dataset with different preprocessing and data augmentation steps and randomly spllit the dataset into train, test and validation datasets.&#x20;

## Preparing your own dataset

* Create an account on Roboflow and choose the pricing plan as per your requirement.
* Create a project in Roboflow and add the annotations labels group and select **Object Detection (Bounding Box)** as your project type.
* Update your video and images dataset to the roboflow.
  * Split the video into individual frames by varying the frames per second.
* Start labelling the images by drawing the bounding boxes in the online editor of roboflow.
  * Roboflow also lets you use the Bounding boxes from the previous image or frame, which can be very useful and time saving especially when you are working with video data.
* Once you are done with labelling your dataset proceed to generating the dataset.
* Click on Versions tab in your project and click on generate new version button.
  * Select test, train, validate percentages
  * Apply different preprocessing steps as per your requirements like resize, crop, grayscale etc.
  * Once you are done with preprocessing, now proceed to the augmentation steps. Select different augmentation options ( both image level and bounding boxes options ) like changing hue, saturation, blur, shear, rotation etc.
  * Once you applied your augmentation options, click on generate dataset at the bottom of the page. This will generate a new dataset with all the options you selected earlier.
* After generating the dataset click on export and select **YOLO Darknet** option to export in the form suitable for yolo to train.