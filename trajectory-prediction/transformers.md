# Transformers

The Transformer Neural Network is a novel architecture that aims to solve sequence-to-sequence tasks while handling long-range dependencies with ease. It was proposed in the paper _“Attention Is All You Need”_ 2017. It is the current state-of-the-art technique in the field of NLP.

We experimented with transformers in the trajectory predictions, since it can contain the past history better than RNN and LSTM's and it can be implemented using GPU parellelism features.

This code is by the authors of “Transformer Networks for Trajectory Forecasting”, which helps in dataset preparation and training \[4].

### Downloading the code&#x20;

Download the repo from github with following command

```
git clone https://github.com/FGiuliari/Trajectory-Transformer.git
```

### Training the model

To train just run the train\_individual.py with different parameters

```
CUDA_VISIBLE_DEVICES=0 python train_individualTF.py --dataset_name eth --max_epoch 240 --batch_size 100 --name eth_train --factor 1
```

* name: final model name&#x20;
* dataset\_name: name of the dataset to train the transformer&#x20;
* max\_epoch: number of epochs to continue training&#x20;
* batch\_size: batch size of the training data

### More parameters

For accessing more parameters like embedding size, number of layers, heads, run the following command

```
python train_individualTF.py -h
```

* \-h, --help show this help message and exit&#x20;
* \--dataset\_folder DATASET\_FOLDER&#x20;
* \--dataset\_name DATASET\_NAME&#x20;
* \--obs OBS&#x20;
* \--preds PREDS&#x20;
* \--emb\_size EMB\_SIZE&#x20;
* \--heads HEADS&#x20;
* \--layers LAYERS&#x20;
* \--dropout DROPOUT&#x20;
* \--cpu&#x20;
* \--val\_size VAL\_SIZE&#x20;
* \--verbose&#x20;
* \--max\_epoch MAX\_EPOCH&#x20;
* \--batch\_size BATCH\_SIZE&#x20;
* \--validation\_epoch\_start VALIDATION\_EPOCH\_START&#x20;
* \--resume\_train&#x20;
* \--delim DELIM&#x20;
* \--name NAME&#x20;
* \--factor FACTOR&#x20;
* \--save\_step SAVE\_STEP&#x20;
* \--warmup WARMUP&#x20;
* \--evaluate EVALUATE&#x20;
* \--model\_pth MODEL\_PTH
