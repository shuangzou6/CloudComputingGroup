# pytorch-cifar10
Group practice on CIFAR10 with PyTorch <br>
Inspired by [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar) by [kuangliu](https://github.com/kuangliu). 

## Introduction
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. 
There are 50000 training images and 10000 test images. 

The dataset is divided into five training batches and one test batch, each with 10000 images. 
The test batch contains exactly 1000 randomly-selected images from each class. 
The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. 
Between them, the training batches contain exactly 5000 images from each class. 

## Requirement
- python3.6
- numpy
- pytorch 0.4.0
- torchvision 0.2.0

## Usage
```bash
python3 main.py
```
optional arguments:

    --lr                default=1e-3    learning rate
    --epoch             default=200     number of epochs tp train for
    --trainBatchSize    default=100     training batch size
    --testBatchSize     default=100     test batch size
## Configs
__200__ epochs for each run-through, <br>
__500__ batches for each training epoch, <br>
__100__ batches for each validating epoch, <br>
__100__ images for each training and validating batch

##### Learning Rate
__1e-3__ for [1,74] epochs <br>
__5e-4__ for [75,149] epochs <br>
__2.5e-4__ for [150,200) epochs <br>

## Result
Models | Accuracy | Comments
:---:|:---:|:---:
[Alexnet](https://github.com/IvoryCandy/pytorch-cifar10/blob/master/models/AlexNet.py) | 74.74% | Result is far away from my expectation (5%+). Reasons might be inappropriate modification to fit dataset(32x32 images). 
[VGG](https://github.com/IvoryCandy/pytorch-cifar10/blob/master/models/VGG.py) | TBD | - - - -
[ResNet](https://github.com/IvoryCandy/pytorch-cifar10/blob/master/models/ResNet.py) | TBD | - - - -

