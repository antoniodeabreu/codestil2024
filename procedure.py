import numpy as np
from sklearn.model_selection import train_test_split
from os import path

from torch.nn.functional import normalize
# from rbf_layer import RBFLayer

import matplotlib.pyplot as plt

# pytorch relates imports
import torch 
import torch.nn as nn
import torch.optim as optim


#csv to numpy
from numpy import genfromtxt

from captum.attr import IntegratedGradients, Occlusion, InputXGradient, FeatureAblation, DeepLift, GradientShap, ShapleyValueSampling
import seaborn as sns



n_inputs=1
n_outputs = 1

num_epochs=500 
batch_size = 5 
learning_rate = 0.001


# build custom module for logistic regression
class Regression(nn.Module):    
    # build the constructor
    def __init__(self, n_inputs, n_outputs):
        super(Regression, self).__init__()
        self.linear = torch.nn.Linear(n_inputs, n_outputs)
    # make predictions
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss


 
def train(model_inp, num_epochs = num_epochs):
    optimizer = torch.optim.RMSprop(model_inp.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for inputs, labels in train_iter:
            # forward pass
            outputs = model_inp(inputs)
            # defining loss
            loss = criterion(outputs, labels)
            # zero the parameter gradients
            optimizer.zero_grad()
            # computing gradients
            loss.backward()
            # accumulating running loss
            running_loss += loss.item()
            # updated weights based on computed gradients
            optimizer.step()
        if epoch % 10 == 0:    
            print('Epoch [%d]/[%d] running accumulative loss across all batches: %.3f' %
                  (epoch + 1, num_epochs, running_loss))
        running_loss = 0.0


if __name__ == '__main__':

# n_inputs=1
# n_outputs = 1

# num_epochs=500 
# batch_size = 4
# learning_rate = 0.001


 coefficients = [] 
 input_tensors_list = torch.load('input_tensor.pt')
 output_tensors_list = torch.load('output_tensor.pt') 
 for i in range(len(input_tensors_list)):  # Iterate over the length of the shorter list (ensures equal iteration count)
    X_train1 = input_tensors_list[i]
    X_train1 =X_train1.view(-1,1)
    y_train1 = output_tensors_list[i]
    y_train1 =y_train1.view(-1,1)
    datasets = torch.utils.data.TensorDataset(X_train1, y_train1)
    train_iter = torch.utils.data.DataLoader(datasets, batch_size, shuffle=True, num_workers=8)
    criterion = RMSELoss()
    model = Regression(n_inputs, n_outputs)
    train(model)
    coefficients.append(model.linear.weight)


 torch.save(coefficients, 'coefficients.pt')


 coefficients1=torch.load('coefficients.pt')


 coefficients1[2][0][0].item()


