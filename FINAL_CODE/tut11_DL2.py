#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Machine Learning COMP3032
Module 11 - sequence modelling

@author: yiguo
"""
#%% Preamble: packages 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#%% ###########Section 1: Comparison - naive model and LSTM ###########
# Seed
seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

n = 15
X = [float(x) for x in range(n)]
y = [1.6*x + 4 + np.random.normal(10, 1) for x in X]
X = torch.tensor(X).unsqueeze(-1)
y =torch.tensor(y).unsqueeze(-1)

plt.plot(X,y,'r*')
plt.title('Sequence data')
plt.show()

# Very naive solution for sequence modeling
X_train = y[:9]
y_train = y[1:10]
X_val = y[10:-1]
y_val = y[11:]

seq_model = nn.Sequential(
    nn.Linear(1, 100),
    nn.Tanh(),
    nn.Linear(100, 1))


def training_loop_seq(n_epochs, optimiser, model, loss_fn, X_train,  X_val, y_train, y_val):
    for epoch in range(1, n_epochs + 1):
        output_train = model(X_train) # forwards pass
        loss_train = loss_fn(output_train, y_train) # calculate loss
        output_val = model(X_val) 
        loss_val = loss_fn(output_val, y_val)
        
        optimiser.zero_grad() # set gradients to zero
        loss_train.backward() # backwards pass
        optimiser.step() # update model parameters
        if epoch == 1 or epoch % 100 == 0:
            print(f"Epoch {epoch}, Training loss {loss_train.item():.4f},"
                  f" Validation loss {loss_val.item():.4f}")

optimiser = optim.Adam(seq_model.parameters(), lr=1e-3)
training_loop_seq(
    n_epochs = 900, 
    optimiser = optimiser,
    model = seq_model,
    loss_fn = nn.MSELoss(),
    X_train = X_train,
    X_val = X_val, 
    y_train = y_train,
    y_val = y_val)


pred = seq_model(y[:-1])
predy = pred.detach().numpy()
plt.plot(X[1:],y[1:],'r*')
plt.plot(X[1:],predy,'b.')
plt.show()

# What is the above doing? 

#%% We utilise LSTM instead. 
class LSTM(nn.Module):
    def __init__(self, hidden_layers=16):
        super(LSTM, self).__init__()
        self.hidden_layers = hidden_layers
        # lstm1, lstm2, linear are all layers in the network
        self.lstm1 = nn.LSTMCell(1, self.hidden_layers)
        self.lstm2 = nn.LSTMCell(self.hidden_layers, self.hidden_layers)
        self.linear = nn.Linear(self.hidden_layers, 1)
        
    def forward(self, y):
        outputs, n_samples = [], y.size(0)
        h_t = torch.zeros(n_samples, self.hidden_layers, dtype=torch.float32)
        c_t = torch.zeros(n_samples, self.hidden_layers, dtype=torch.float32)
        h_t2 = torch.zeros(n_samples, self.hidden_layers, dtype=torch.float32)
        c_t2 = torch.zeros(n_samples, self.hidden_layers, dtype=torch.float32)
        
        for time_step in y.split(1, dim=1):
            # N, 1
            h_t, c_t = self.lstm1(time_step, (h_t, c_t)) # initial hidden and cell states
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2)) # new hidden and cell states
            output = self.linear(h_t2) # output from the last FC layer
            outputs.append(output)
            
        # transform list to tensor    
        outputs = torch.cat(outputs, dim=1)
        return outputs

#%%
def training_loop(n_epochs, model, optimiser, loss_fn, 
                  train_input, train_target, test_input, test_target):
    for i in range(n_epochs):
        def closure():
            optimiser.zero_grad()
            out = model(train_input)
            loss = loss_fn(out, train_target)
            loss.backward()
            return loss
        optimiser.step(closure)
        # print the loss
        out = model(train_input)
        loss_print = loss_fn(out, train_target)
        print("Step: {}, Loss: {}".format(i, loss_print))

lstmmodel = LSTM()
criterion = nn.MSELoss()
optimiser = optim.LBFGS(lstmmodel.parameters(), lr=0.08)    

training_loop(n_epochs = 50,
              model = lstmmodel,
              optimiser = optimiser,
              loss_fn = criterion,
              train_input = X_train,
              train_target = y_train,
              test_input = X_val,
              test_target = y_val)

pred = lstmmodel(y[:-1])
lstmpredy = pred.detach().numpy()
plt.plot(X[1:],y[1:],'r*')
plt.plot(X[1:],predy,'b.')
plt.plot(X[1:],lstmpredy,'g.')
plt.title("Performance comparison")
plt.legend(["Data","Naive sequential model","LSTM"])
plt.show()
# It shows fromt the figure that LSTM (very small) is better than the naive model (big). 
# Q1: Find out the performance metrics (numeric values) of these two methods. 
# Q2: How many parameters in each model? 

#%% ########### Section 2: Another simulation ###########
# Now we move on to different data. 
N = 10 # number of samples
L = 100 # length of each sample (number of values for each sine wave)
T = 2 # width of the wave
x = np.empty((N,L), np.float32) # instantiate empty array
x[:] = np.arange(L) + np.random.randint(-4*T, 4*T, N).reshape(N,1)
y = np.sin(x/1.0/T).astype(np.float32)

import matplotlib.pyplot as plt
plt.plot(np.arange(L),y[0,:],'r',linewidth=1.0)
plt.plot(np.arange(L),y[1,:],'g',linewidth=1.0)
plt.plot(np.arange(L),y[2,:],'b',linewidth=1.0)
plt.show()

train_input = torch.from_numpy(y[3:, :-1]) 
train_target = torch.from_numpy(y[3:, 1:])
test_input = torch.from_numpy(y[:3, :-1]) 
test_target = torch.from_numpy(y[:3, 1:])

def training_loop_full(n_epochs, model, optimiser, loss_fn, 
                  train_input, train_target, test_input, test_target):
    for i in range(n_epochs):
        def closure():
            optimiser.zero_grad()
            out = model(train_input)
            loss = loss_fn(out, train_target)
            loss.backward()
            return loss
        optimiser.step(closure)
        out = model(train_input)
        loss_print = loss_fn(out, train_target)
        print("Step: {}, Loss: {}".format(i, loss_print))


model = LSTM()

criterion = nn.MSELoss()
optimiser = optim.LBFGS(model.parameters(), lr=0.08)    

training_loop_full(n_epochs = 20,
              model = model,
              optimiser = optimiser,
              loss_fn = criterion,
              train_input = train_input,
              train_target = train_target,
              test_input = test_input,
              test_target = test_target)              

#%% Show the test results in figures
pred = model(test_input)
predy = pred.detach().numpy()
for i in range(test_target.size()[0]):
    plt.figure()
    plt.plot(test_target[i,:],'r*--')
    plt.plot(predy[i,:],'b.')
    plt.title("Test results on the {}-th test sample".format(i+1))
    plt.show()
    
#%% ########### Section 3: task to do ###########
# Task: Make prediction into the future by 100 steps. 
#       Build this predicting into future in foward method of LSTM model. 
#       Use the template below and complete the code for this task.
#       After you done, insert your new LSTM model and retrain it in previous 
#       section and test out using the code at the end.
# Do NOT run this part until you finish it. 
# You may need to refresh the environment by clearing all variables. If you do 
# so, make sure you run preamble and section 2. 

class LSTM_intothefuture(nn.Module):
    def __init__(self, hidden_layers=16):
        super(LSTM_intothefuture, self).__init__()
        self.hidden_layers = hidden_layers
        # lstm1, lstm2, linear are all layers in the network
        self.lstm1 = nn.LSTMCell(1, self.hidden_layers)
        self.lstm2 = nn.LSTMCell(self.hidden_layers, self.hidden_layers)
        self.linear = nn.Linear(self.hidden_layers, 1)
        
    def forward(self, y, future_preds=0): # future_preds indicates how many steps that the model predicts into the future. 
        outputs, n_samples = [], y.size(0)
        h_t = torch.zeros(n_samples, self.hidden_layers, dtype=torch.float32)
        c_t = torch.zeros(n_samples, self.hidden_layers, dtype=torch.float32)
        h_t2 = torch.zeros(n_samples, self.hidden_layers, dtype=torch.float32)
        c_t2 = torch.zeros(n_samples, self.hidden_layers, dtype=torch.float32)
        
        for time_step in y.split(1, dim=1):
            # N, 1
            h_t, c_t = self.lstm1(time_step, (h_t, c_t)) # initial hidden and cell states
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2)) # new hidden and cell states
            output = self.linear(h_t2) # output from the last FC layer
            outputs.append(output)
            
        for i in range(future_preds): # Rolling prediction 
            # this only generates future predictions if we pass in future_preds>0
            # mirrors the code above, using last output/prediction as input
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs.append(output)
        # transform list to tensor    
        outputs = torch.cat(outputs, dim=1)
        return outputs
    
    
    
#%% Use the following code to test your LSTM model which can predict into the 
#   future. 
model_future = LSTM_intothefuture()

criterion = nn.MSELoss()
optimiser = optim.LBFGS(model_future.parameters(), lr=0.08)    

training_loop_full(n_epochs = 20,
              model = model_future,
              optimiser = optimiser,
              loss_fn = criterion,
              train_input = train_input,
              train_target = train_target,
              test_input = test_input,
              test_target = test_target)              


future = 100 # Predict 100 steps into the future for all test samples.
pred = model_future(test_input, future)
loss = criterion(pred[:, :-future], test_target)
predy = pred.detach().numpy()

# Plot them 
for i in range(test_target.size()[0]):
    plt.figure()
    plt.plot(test_target[i,:],'r*-')
    plt.plot(predy[i,:],'b.--')
    plt.title("The {}-th test results and future prediction".format(i+1))
    plt.show()