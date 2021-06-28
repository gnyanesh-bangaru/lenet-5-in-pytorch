# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 17:09:08 2021

@author: cgnya
"""
import torch
from torch import nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
# import time
# import os
# import copy
from Model import LeNet

class TrainTest:
    
    def __init__(self):
        self.transform = transforms.Compose(
                            [transforms.ToTensor(),
                              transforms.Normalize(0.5, 0.5),
                              transforms.Resize([32,32])])
    
        self.train_data = torchvision.datasets.MNIST(
                            root='/data',
                            train=True,
                            download=True,
                            transform=self.transform)
    
        self.test_data = torchvision.datasets.MNIST(
                            root='/data',
                            train=False,
                            download=True,
                            transform=self.transform)
    
    
        self.trainloader = DataLoader(self.train_data, batch_size=256, shuffle=True)
        self.testloader = DataLoader(self.test_data, batch_size=1024, shuffle=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

    def train_test(self):
        
        model = LeNet().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(model.parameters(), lr = 1e-1, momentum=0.9)
        self.epochs = 30
        
        #Training Loop
        for epoch in range(self.epochs):
            running_loss = 0.0
            for i, (images, labels) in enumerate(self.trainloader, 0):
                images = images.to(self.device)       
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                
                outputs = model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                if i % 32==31:            
                    print('[%d / %5f] Loss: %.3f' % (epoch+1, i+1, running_loss / 32))
                    running_loss = 0.0
                    
        correct = 0
        total = 0
        #Testing Loop
        with torch.no_grad():
            for images, labels in self.testloader:
                images =  images.to(self.device)
                labels = labels.to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 'Accuracy of the network on the 10000 test images: %d %%' % (100.0 * correct / total)