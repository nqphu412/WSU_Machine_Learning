#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Machine Learning 
Module 10: MLP 
"""

#%% Preamble: packages 
import numpy as np
from sklearn.datasets import load_iris 
from sklearn.linear_model import Perceptron

# Use pip install pytorch to install torch 
# May need to sort out ipywidgets: 
#   pip install ipywidgets
#   or with conda, do:
#   conda install -c conda-forge ipywidgets
#   jupyter nbextension enable --py widgetsnbextension
#   You many need to restart the ipython kernel after the above installation

import os
import torch
from torch import nn
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

#%% ########### Section 1: MLP using sklearn ###########
iris = load_iris()
X = iris.data[:, (2, 3)] # petal length, petal width 
y = (iris.target == 0).astype(np.int) # Iris setosa?

per_clf = Perceptron()
per_clf.fit(X, y)
y_pred = per_clf.predict([[2, 0.5]])

#%% ########### Section 2: MLP using pytorch ###########

""" 
We are going to use CIFAR-10 images for classification task. 
The CIFAR-10 dataset is a collection of images that are commonly used to train machine learning and computer vision 
algorithms. It is one of the most widely used datasets for machine learning research. The CIFAR-10 dataset contains 
60,000 32x32 color images in 10 different classes. The 10 different classes represent airplanes, cars, birds, cats, 
deer, dogs, frogs, horses, ships, and trucks. There are 6,000 images of each class.
See https://www.cs.toronto.edu/~kriz/cifar.html for details. 
"""

class MLP(nn.Module):
  # Multilayer Perceptron.
  def __init__(self): # Here we define the MLP structure. 
    super().__init__()
    self.layers = nn.Sequential(
      nn.Flatten(), # One image is 32x32 with 3 channels, ie R, G, B. So we flatten it first. 
      nn.Linear(32 * 32 * 3, 64), # Then the input is 32*32*3, and we decide to have 64 units in the first hidden layer
      nn.ReLU(), # Use ReLU as the activation function in the first hidden layer. 
      nn.Linear(64, 32), # Then second hidden layer with 32 units. 
      nn.ReLU(), # Again ReLU as the activatino function
      nn.Linear(32, 10) # Finally outlayer has 10 units as we have 10 classes in the data set as targets. Each 
                        # unit indicates the probability of the input being corresponding class.                  
    )


  def forward(self, x): # Here we define forward pass. Basically just run through all layers for given input
    # Forward pass
    return self.layers(x)
 
def calculate_accuracy(y_pred, y): # Get accuracy manually
# y_pred: the output from the model
# y: the true labels
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc,top_pred    
  
if __name__ == '__main__':
    
    # Set fixed random number seed
    torch.manual_seed(42)
    
    # Prepare CIFAR-10 dataset, total 50k images. 
    dataset = CIFAR10(os.getcwd(), download=True, transform=transforms.ToTensor())
    train_data,test_data = torch.utils.data.random_split(dataset,[40000,10000])
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=10, shuffle=True)
    
    # Initialize the MLP
    mlp = MLP()
    
    # Define the loss function and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)
    
    # Run the training loop
    for epoch in range(0, 5): # 5 epochs at maximum
      
      # Print epoch
      print(f'Starting epoch {epoch+1}')
      
      # Set current loss value
      current_loss = 0.0
      
      # Iterate over the DataLoader for training data
      for i, data in enumerate(trainloader, 0):
        # Get inputs
        inputs, targets = data
       
        # Zero the gradients
        optimizer.zero_grad()
        
        # Perform forward pass
        outputs = mlp(inputs)
        
        # Compute loss
        loss = loss_function(outputs, targets)
        
        # Perform backward pass
        loss.backward()
        
        # Perform optimization
        optimizer.step()
        
        # Print statistics
        current_loss += loss.item()
        if i % 500 == 499:
            print('Loss after mini-batch %5d: %.3f' %
                  (i + 1, current_loss / 500))
            current_loss = 0.0
    
    # Process is complete.
    print('Training process has finished.')
      
    # Test the trained model
    
    test_dataloader = DataLoader(test_data, batch_size=20, shuffle=False)
    
    for i, data in enumerate(test_dataloader, 0):
        inputs, targets = data
        acc,y = calculate_accuracy(mlp(inputs),targets)
        if i==0:
            predy = y
            truey = targets
        else: 
            predy =torch.cat((predy,y))
            truey = torch.cat((truey,targets))
            
    acc,_ = calculate_accuracy(predy,truey)  
    print("Total accuracy: ",acc.detach().numpy())
    """
    The accuracy is really disappointing. But it is ok. We will have much better 
    models called convolutional neural networks later on to improve it. Also the 
    number of epoches is quite small. You may increase it and train it for longer 
    to see if it improves. 
    """

#%% ########### Section 3: task to do ###########
# Task: build a simple autoencoder based on the pytorch code (and data) here. 
class MLPautoencoder(nn.Module):
  # Autoencoder based on Multilayer Perceptron.
  def __init__(self): # Here we define the MLP structure. 
    super().__init__()
    self.layers = nn.Sequential(
      nn.Flatten(), # One image is 32x32 with 3 channels, ie R, G, B. So we flatten it first. 
      nn.Linear(32 * 32 * 3, 64), # Then the input is 32*32*3, and we decide to have 64 units in the first hidden layer
      nn.ReLU(), # Use ReLU as the activation function in the first hidden layer. 
      nn.Linear(64, 32), # Then second hidden layer with 32 units. 
      nn.ReLU(), # Again ReLU as the activatino function
      nn.Linear(32, 32 * 32 * 3) # Finally outlayer has the same size as input 
    )


  def forward(self, x): # Here we define forward pass. Basically just run through all layers for given input
    # Forward pass
    return self.layers(x)

dataset = CIFAR10(os.getcwd(), download=True, transform=transforms.ToTensor())
train_data,test_data = torch.utils.data.random_split(dataset,[1000,49000])
trainloader = torch.utils.data.DataLoader(train_data, batch_size=10, shuffle=True)

# Initialize the MLP
model = MLPautoencoder()

# Define the loss function and optimizer
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Run the training loop
for epoch in range(0, 5): # 5 epochs at maximum
  
  # Print epoch
  print(f'Starting epoch {epoch+1}')
  
  # Set current loss value
  current_loss = 0.0
  
  # Iterate over the DataLoader for training data
  for i, data in enumerate(trainloader, 0):
    # Get inputs
    inputs, targets = data
  
    # Zero the gradients
    optimizer.zero_grad()
    
    # Perform forward pass
    outputs = model(inputs)
    
    # Compute loss
    loss = loss_function(outputs, inputs.flatten(1))
    
    # Perform backward pass
    loss.backward()
    
    # Perform optimization
    optimizer.step()
    
    # Print statistics
    current_loss += loss.item()
    
    if i % 5 == 4:
        print('Loss after mini-batch %5d: %.3f' %
              (i + 1, current_loss / 500))
        current_loss = 0.0

# Process is complete.
print('Training process has finished.')

#%%
import torchvision as tv
def imshow(img,ax):
  img = img / 2 + 0.5   # unnormalize
  npimg = img.numpy()   # convert from tensor
  ax.imshow(np.transpose(npimg, (1, 2, 0))) 
  ax.axis('off')
  # ax.show()
  
dataiter = iter(trainloader)
imgs, lbls = dataiter.next()
plt.figure()
fig, axes = plt.subplots(2,10)
for i in range(10):  # show just the frogs
    imshow(tv.utils.make_grid(imgs[i]),axes[0,i])
    output = model(torch.unsqueeze(imgs[0], dim=0)).reshape(3,32,32).detach()
    imshow(tv.utils.make_grid(output),axes[1,i])
