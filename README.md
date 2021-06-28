# LeNet-5-Implementation-through-PyTorch
This is an architecture which was proposed in 1998 Proceedings of IEEE journal with about 37505	citations as of today.
Even though this is a paper from 1998, it provides basic fundamentals of a neural network. By the development of network architecture and error rate reduction, we can grasp the importance of Deep Learning components to be added on to the network.

This is a simple implementation of the MNIST dataset on LeNet-5 Architecture using PyTorch. (Handwritten Digits)


*NOTE: This code was explicitly coded by Object Oriented Approach including one class for each of the files.*

# Requirements
Python >= *3.0*

PyTorch Version >= *0.4.0*

torchvision >= *0.2.1*

# Architecture
Number of Image Channels = 1

32×32 Input Image 

Six 28×28 feature maps convolutional layer (5×5 size) 

Average Pooling layers (2×2 size) 

Sixteen 10×10 feature maps convolutional layer (5×5 size) 

Average Pooling layers (2×2 size) 

Flattened to 16x5x5 which is

Fully connected to 120 neurons 

Fully connected to 84 neurons 

Fully connected to 10 outputs

# Obtained Output
![Figure 2021-06-29 001134](https://user-images.githubusercontent.com/67636257/123688088-42e1dd00-d86f-11eb-8d91-da060c5eb880.png)

![Screenshot (306)](https://user-images.githubusercontent.com/67636257/123688567-cac7e700-d86f-11eb-94fe-f588246cd7d2.png)

# Obtained Accuracy
**~99 %**
