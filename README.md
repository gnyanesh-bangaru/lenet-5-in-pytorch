# LeNet-5-Implementation-through-PyTorch
This is an architecture which was proposed in 1998 Proceedings of IEEE journal with about 37505	citations as of today.
Even though this is a paper from 1998, it provides basic fundamentals of a neural network. By the development of network architecture and error rate reduction, we can grasp the importance of Deep Learning components to be added on to the network.

This is a simple implementation of the MNIST dataset on LeNet-5 Architecture using PyTorch. (Handwritten Digits)


*NOTE: This code was explicitly coded by Object Oriented Approach including one class for each of the files.*

# Architecture
32×32 Input Image 

Six 28×28 feature maps convolutional layer (5×5 size) 

Average Pooling layers (2×2 size) 

Sixteen 10×10 feature maps convolutional layer (5×5 size) 

Average Pooling layers (2×2 size) 

Flattened to 16x5x5 which is

Fully connected to 120 neurons 

Fully connected to 84 neurons 

Fully connected to 10 outputs

# Requirements
Python >= *3.0*

PyTorch Version >= *0.4.0*

torchvision >= *0.2.1*

# Obtained Accuracy
**99 %**
