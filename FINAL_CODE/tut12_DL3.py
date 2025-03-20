#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Machine Learning 
Module 12: DNN 
"""

#%% Preamble: packages 
import numpy as np
from sklearn.datasets import load_iris 
from sklearn.linear_model import Perceptron

import os
import torch
from torch import nn
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from torchviz import make_dot
import hiddenlayer as hl

# use the following command to install missing package
# pip install packagename 
#%% ########### Section 1: CNN using pytorch ###########

""" 
We are going to use CIFAR-10 images for classification task. 
The CIFAR-10 dataset is a collection of images that are commonly used to train machine learning and computer vision 
algorithms. It is one of the most widely used datasets for machine learning research. The CIFAR-10 dataset contains 
60,000 32x32 color images in 10 different classes. The 10 different classes represent airplanes, cars, birds, cats, 
deer, dogs, frogs, horses, ships, and trucks. There are 6,000 images of each class.
See https://www.cs.toronto.edu/~kriz/cifar.html for details. 
"""

# define the CNN architecture
class mySmallCNN(nn.Module):
   
    def __init__(self):
        
        super(mySmallCNN, self).__init__()

        self.conv_layer = nn.Sequential(

            # Conv Layer block 
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096,128), # Q1: The input is 4096 dim. Why? 
            nn.ReLU(inplace=True),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(32, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        """Perform forward."""
        
        # conv layers
        x = self.conv_layer(x)
        
        # flatten
        x = x.view(x.size(0), -1)
        
        # fc layer
        x = self.fc_layer(x)

        return x

#%% Define the model training function
def train_model(optimiser, model, loss_function, trainloader, valloader, 
                n_epochs=10, fplotloss=True, fdraw=False, filename="", mode=0):
# =============================================================================
# n_epochs: number of epochs. Be careful with this number as a large one may end up a long time to train  
# mode: 0 - as classification, then target is the label. 
#       1 - as autoencoder, then target is the data itself. Need to check forward method because in AE mode
#           the forward method of the model can output multiple things in a tuple. Also the loss function 
#           has to be in line with the output-target pair. 
# =============================================================================
    # move tensors to GPU if CUDA is available
    train_on_gpu = torch.cuda.is_available()
    if train_on_gpu:
        print("GPU available! Train model on GPU.")
        model.cuda()
    
    train_losslist = []
    val_losslist = []
   
    
    # Track the best model so far (evaluated on validation set)
    val_loss_min = np.Inf 
    
    print("Entering training cycles.")
    for epoch in [*range(n_epochs)]:
    
        # keep track of training and validation loss
        train_loss = 0.0
        val_loss = 0.0
        
        # train the model, one training cycle
        model.train()
        for data, target in trainloader:
            # move tensors to GPU if CUDA is available
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
            # clear the gradients of all optimized variables
            optimiser.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            if mode==0:
                loss = loss_function(output, target)
            else: # Be careful with the AE mode training
                loss = loss_function(output[0], data)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimiser.step()
            # update training loss
            train_loss += loss.item()*data.size(0)
 
        # validate the model 
        model.eval()
        for data, target in valloader:
            # move tensors to GPU if CUDA is available
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            if mode==0:
                loss = loss_function(output, target)
            else: 
                loss = loss_function(output[0], data)
            # update average validation loss 
            val_loss += loss.item()*data.size(0)
        
        # calculate average losses
        train_loss = train_loss/len(trainloader.dataset)
        val_loss = val_loss/len(valloader.dataset)
    
        train_losslist.append(train_loss)
        val_losslist.append(val_loss)
            
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss, val_loss))
        
        # save model if validation loss has decreased
        if val_loss <= val_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            val_loss_min,
            val_loss))
            torch.save(model.state_dict(), 'bestmodel'+filename+'.pt')
            val_loss_min = val_loss
            
    # Plot training and validation loss if fplotloss=True
    if fplotloss:
        plt.plot(*range(n_epochs), train_losslist)
        plt.plot(*range(n_epochs), val_losslist)
        plt.ylim((min(train_losslist+val_losslist),max(train_losslist+val_losslist)))
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Performance of the model")
        plt.legend(["Training loss","Validation loss"])
        plt.show()    
    
    # Visualise the network structure and store it in myCNN_structure.png. 
    # Look for it under the working directory. Only when fdraw=True
    if fdraw:
        transforms = [ hl.transforms.Prune('Constant') ] # Removes Constant nodes from graph.
        graph = hl.build_graph(model, data, transforms=transforms)
        graph.theme = hl.graph.THEMES['blue'].copy()
        graph.save('myCNN_structure', format='png')
        make_dot(output, params=dict(list(model.named_parameters()))).render("myCNN_structure_moredetail", format="png")

    # Process is complete.
    print('Training process has finished.')
    return train_losslist, val_losslist

# Define accuracy calculation function
def calculate_accuracy(y_pred, y): # Get accuracy manually
# y_pred: the output from the model
# y: the true labels
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc,top_pred    

#%% Main program start from here. This program can run in terminal without section 2. 
if __name__ == '__main__':
    
    # Set fixed random number seed
    torch.manual_seed(42)
   
    # Prepare CIFAR-10 dataset, total 50k images. 
    dataset = CIFAR10(os.getcwd(), download=True, transform=transforms.ToTensor())
    train_data,val_data, test_data = torch.utils.data.random_split(dataset,[400,100,49500])
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=10, shuffle=True)
    valloader = torch.utils.data.DataLoader(val_data, batch_size=10, shuffle=True)
    
    model = mySmallCNN()
    print(model)
    
    optimiser = torch.optim.SGD(model.parameters(), lr=.01)
    # use CrossEntropyLoss as loss and mode=0 as we are doing cls here. 
    train_model(optimiser, model, nn.CrossEntropyLoss(), 
                trainloader, valloader, n_epochs=10, fplotloss=True, fdraw=True, filename='_small');
    
    # Test the trained model
    
    test_dataloader = DataLoader(test_data, batch_size=20, shuffle=False)
    
    for i, data in enumerate(test_dataloader, 0):
        inputs, targets = data
        acc,y = calculate_accuracy(model(inputs),targets)
        if i==0:
            predy = y
            truey = targets
        else: 
            predy =torch.cat((predy,y))
            truey = torch.cat((truey,targets))
            
    acc,_ = calculate_accuracy(predy,truey)  
    print("Total accuracy: ",acc.detach().numpy())
    
# =============================================================================
# The accuracy is not a concern. To improve it we can do the following: 
# 1. increase training data size;
# 2. increase number of epoches;
# 3. increase number of kernels; 
# 4, increase convolution layers and many more.
# Beware that the above will also lengthen the computation maybe significantly. 
# Your computer may even seem to freeze. It's better to do above on a powerful 
# computer, ideally the one with GPUs.
# =============================================================================
    
#%% ########### Section 2: VGG like CNN ###########
# =============================================================================
# Task: build a VGG like CNN as shown in the tutorial material
# Hints: 
# ▸ Fix the kernel size to 3x3. 
# ▸ The input images are of size 32x32x3, ie 3 channels.
# ▸ Assume after each convolution layer, use a ReLU layer as activation function.
# ▸ Carefully pick the parameters for max pooling in order to get the right size as required in the figure. 
# ▸ You can add BN, dropout etc. as long as they do not change the size. 
# ▸ There is a output layer after fc7 for classification task. 
# =============================================================================

class myVGG(nn.Module):
   
    def __init__(self):
        
        super(myVGG, self).__init__()

        self.conv_layer = nn.Sequential(

            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 3
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 4
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.LazyLinear(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1000),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(1000, 10),
            nn.Softmax(dim=1)         
        )

    def forward(self, x):
        """Perform forward."""
        
        # conv layers
        x = self.conv_layer(x)
        
        # flatten
        x = x.view(x.size(0), -1)
        
        # fc layer
        x = self.fc_layer(x)

        return x

torch.manual_seed(42)
   
# Prepare CIFAR-10 dataset, total 50k images. 
dataset = CIFAR10(os.getcwd(), download=True, transform=transforms.ToTensor())
train_data,val_data, test_data = torch.utils.data.random_split(dataset,[30000,10000,10000])
trainloader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True)
valloader = torch.utils.data.DataLoader(val_data, batch_size=100, shuffle=True)

model = myVGG()
print(model)

optimiser = torch.optim.SGD(model.parameters(), lr=.01)

train_model(optimiser, model, nn.CrossEntropyLoss(), 
            trainloader, valloader, n_epochs=5000, fplotloss=False, fdraw=False,filename='_myVGG')

#%% ########### Section 3: CNN based autoencoder ###########
# =============================================================================
# Section3 (optional): This is a challenge for advanced level. You can reuse 
# most of the code in tutorial - moduel12.py for example the training function. 
# Task: build an autoencoder (AE) for image reconstruction, similar to module10 
# tutorial but we use CNN instead. 
# Hints: 
# Use ConvTranspose2d to increase the size of feature map(s). 
# Be careful with the sizes/shape of each layer’s output; 
# Final output layer should have the output in the same shape as the input (images) 
# and accordingly the loss function has to change. 
# We show the results of reconstructed images by a small CNN based (AE) on 400 
# images after only 100 epochs and the shapes are already there!
# =============================================================================

# define autoencoder architecture
class mySmallAE(nn.Module):
   
    def __init__(self):
        
        super(mySmallAE, self).__init__()

        self.conv_layer1 = nn.Sequential(
            # Conv Layer block 
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        self.conv_layer2 = nn.Sequential(
            # Conv Layer block 
            nn.ConvTranspose2d(16, 32, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.ReLU(inplace=True),            
            )

    def forward(self, x):
        # conv layers
        x = self.conv_layer1(x)
        
        # flatten
        v = x.view(x.size(0), -1)
        
        # fc layer
        x = self.conv_layer2(x)

        return x, v

torch.manual_seed(42)
   
# Prepare CIFAR-10 dataset, total 50k images. 
dataset = CIFAR10(os.getcwd(), download=True, transform=transforms.ToTensor())
train_data,val_data, test_data = torch.utils.data.random_split(dataset,[400,100,49500])
trainloader = torch.utils.data.DataLoader(train_data, batch_size=10, shuffle=True)
valloader = torch.utils.data.DataLoader(val_data, batch_size=10, shuffle=True)

model = mySmallAE()
print(model)

optimiser = torch.optim.SGD(model.parameters(), lr=.01)

train_model(optimiser, model, nn.MSELoss(), 
            trainloader, valloader, n_epochs=100, 
            fplotloss=False, fdraw=False, filename='_smallAE', mode=1);

#%%
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']
import torchvision as tv
def imshow(img,ax):
  # img = img / 2 + 0.5   # unnormalize
  npimg = img.numpy()   # convert from tensor
  ax.imshow(np.transpose(npimg, (1, 2, 0))) 
  ax.axis('off')
  # ax.show()

n = 3
dataiter = iter(trainloader)
imgs, lbls = dataiter.next()
output,v = model(imgs)
plt.figure(figsize=(20,20))
fig, axes = plt.subplots(2,n)
for i in range(n):  # show just the frogs
    imshow(tv.utils.make_grid(imgs[i]),axes[0,i])
    axes[0,i].set_title(classes[int(lbls[i])])
    im = output[i].detach()
    imshow(tv.utils.make_grid(im),axes[1,i])
    if i==int(n/2):
        axes[1,i].set_title('Reconstructed images')
fig.tight_layout()
plt.show()

