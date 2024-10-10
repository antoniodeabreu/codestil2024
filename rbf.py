import numpy as np
from sklearn.model_selection import train_test_split
from os import path

from torch.nn.functional import normalize
from rbf_layer import RBFLayer

import matplotlib.pyplot as plt

# pytorch relates imports
import torch
import torch.nn as nn
import torch.optim as optim


#csv to numpy
from numpy import genfromtxt


import torch
import torch.nn as nn


input_size=13
output_size = 1

num_epochs=1800
batch_size = 256 
learning_rate = 0.001


def l_norm(x, p=2):
    return torch.norm(x, p=p, dim=-1)


# Gaussian RBF
def rbf_gaussian(x):
    return (-x.pow(2)).exp()

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


def train_load_save_model(model_obj, model_path):
    if path.isfile(model_path):
        # load model
        print('Loading pre-trained model from: {}'.format(model_path))
        model_obj.load_state_dict(torch.load(model_path))
    else:    
        # train model
        train(model_obj)
        print('Finished training the model. Saving the model to the path: {}'.format(model_path))
        torch.save(model_obj.state_dict(), model_path)



if __name__ == '__main__':
 
 X = genfromtxt("/home/antonio.batista/antonio/Projeto_Antonio_Luciano/dataset.csv", delimiter=",",  skip_header=1, usecols={0,1,2,3,4,5,6,7,8,9,10,11,12})
 y = genfromtxt("/home/antonio.batista/antonio/Projeto_Antonio_Luciano/dataset.csv", delimiter=",",skip_header=1, usecols={15})  
	
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)	

 # save array into csv file 
 np.savetxt("X_test_0.csv", X_test, delimiter = ",")
 np.savetxt("y_test_0.csv", y_test, delimiter = ",")
 np.savetxt("X_train_0.csv", X_train, delimiter = ",")
 np.savetxt("y_train_0.csv", y_train, delimiter = ",")
	  
 X_train = torch.tensor(X_train).float()
 X_train = normalize(X_train, p=2.0, dim = 1)

 y_train = torch.tensor(y_train).view(-1,1).float()
	

    
 datasets = torch.utils.data.TensorDataset(X_train, y_train)
 train_iter = torch.utils.data.DataLoader(datasets, batch_size, shuffle=True, num_workers=32)

  


 #criterion = nn.MSELoss(reduction='sum')

 criterion = RMSELoss()
#loss = criterion(yhat,y)

# Use a radial basis function with euclidean norm
 model = RBFLayer(in_features_dim=input_size, # input features dimensionality
               num_kernels=5,                 # number of kernels
               out_features_dim=output_size,            # output features dimensionality
               radial_function=rbf_gaussian,  # radial basis function used
               norm_function=l_norm)          # l_norm defines the \ell norm


 train_load_save_model(model, "/home/antonio.batista/antonio/Projeto_Antonio_Luciano/rbf_layer/rbf_0.pt")



 X_test = torch.tensor(X_test).float()
 X_test = normalize(X_test.t(), p=2.0, dim = 1)
 X_test = X_test.t()

 y_test = torch.tensor(y_test).view(-1,1).float()

 y_hat=model(X_test)

 print(criterion(y_hat,y_test))

