# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 17:09:08 2021

@author: cgnya
"""
######## Generic Modules ########
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import numpy as np

######## User-Defined Module ########
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
        self.testloader = DataLoader(self.test_data, batch_size=256, shuffle=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = LeNet().to(self.device)

    classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

    
    def train_test(self):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr = 1e-1, momentum=0.9)
        self.epochs = 30
        
        #Training Loop
        for epoch in range(self.epochs):
            running_loss = 0.0
            for i, (images, labels) in enumerate(self.trainloader, 0):
                images = images.to(self.device)       
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                if i % 32==31:            
                    print('[%d / %5f] Loss: %.3f' % (epoch+1, i+1, running_loss / 32))
                    running_loss = 0.0
                    
        correct = 0
        total = 0 
        correct_pred = { classname:0 for classname in self.classes}
        total_pred = { classname:0 for classname in self.classes}
        #Testing Loop
        with torch.no_grad():
            for images, labels in self.testloader:
                images =  images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                _, predictions = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predictions == labels).sum().item()
                    
                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        correct_pred[self.classes[label]] += 1
                    total_pred[self.classes[label]] += 1
                    
            # print accuracy for each class
            for classname, correct_count in correct_pred.items():
                accuracy = 100 * float(correct_count) / total_pred[classname]
                print('Accuracy for class {} is : {:.1f}'.format(classname,accuracy))     
        
        return 'Accuracy of the network on the 10000 test images: %d %%' % (100.0 * correct / total)
    
    
    def visualize(self):    
        self.testloader = DataLoader(self.test_data, batch_size=4, shuffle=True)
        def imshow(img):
            img = img / 2 + 0.5 #unnomalize
            npimg = img.numpy()
            print(npimg.shape)
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
            plt.show()
    
        dataiter = iter(self.testloader)
        images, labels = dataiter.next()
        #print images
        imshow(torchvision.utils.make_grid(images))
        print('Ground Truth Label :',' '.join('%5s' % self.classes[labels[j]] for j in range(4)))
        outputs = self.model(images.to(self.device))
        _, predicted = torch.max(outputs,1) # axis or dim =1 representing column i.e classes
        print('Predicted Label    :', ' '.join('%5s' % self.classes[predicted[j]] for j in range(4)))  
        
