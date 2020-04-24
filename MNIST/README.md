# pytorch-cifar10
group practice on MNIST with PyTorch <br>


## Introduction
The MNIST dataset consists of 60000 training samples and 10000 test samples
each of which is a 28 x 28 pixel grayscale handwritten digital images. 

These handwritten digital have been dimensioned, and be located in the center of the image. 
The entire dataset needs to be loaded into the numpy array for training and testing.

## Requirement
- python3.0
- numpy
- torchvision 0.2.0

## Usage
```bash
python3 Alexnet_MNIST.py
python3 Resnet_MNIST.py
python3 VGGNet_MNIST.py
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
[Alexnet](https://github.com/IvoryCandy/pytorch-cifar10/blob/master/models/AlexNet.py) | 97.82% | Result is far away from my expectation (5%+). Reasons might be inappropriate modification to fit dataset(32x32 images). 
[VGG16](https://github.com/IvoryCandy/pytorch-cifar10/blob/master/models/VGG.py) | 93.35% | - - - -
[ResNet](https://github.com/IvoryCandy/pytorch-cifar10/blob/master/models/ResNet.py) | 90.62% | - - - -
