from os import path

import torch
import torch.nn as nn 
from torch.nn.functional import normalize 

from sklearn.model_selection import train_test_split
from numpy import genfromtxt

from rbf_layer import RBFLayer

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


def train(model_inp, num_epochs = 1):
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


def generate_dataset(data_tensor, num_samples, step_size=1):
  """
  Generates a dataset of input-output pairs for the 'model' function,
  increasing age_academic of the provided 'data_tensor' gradually.

  Args:
      data_tensor (torch.Tensor): The initial input tensor with age_academic and performance.
      num_samples (int): The number of data points to generate (including the original input).
      step_size (int, optional): The increment for age_academic in each sample (default: 1).

  Returns:
      tuple: A tuple containing (input_tensor, output_tensor) for training.
  """

  # Initialize empty lists to store input and output tensors
  input_tensors = []
  output_tensors = []


  # Loop for additional samples with increased age_academic
  for i in range(1, num_samples):
    # Extract the current age_academic value
    current_age_academic = data_tensor[0][8].item() + i * step_size

    # Clamp the value to stay within a reasonable range (adjust if needed)
    current_age_academic = min(max(current_age_academic, 0), 100)

    # Update the performance value if needed (modify if applicable)
    x0 = data_tensor[0][0].item()   
    x1 = data_tensor[0][1].item() 
    x2 = data_tensor[0][2].item()   
    x3 = data_tensor[0][3].item() 
    x4 = data_tensor[0][4].item()   
    x5 = data_tensor[0][5].item() 
    x6 = data_tensor[0][6].item()   
    x7 = data_tensor[0][7].item() 
    x9 = data_tensor[0][9].item()   
    x10 = data_tensor[0][10].item() 
    x11 = data_tensor[0][11].item()   
    x12 = data_tensor[0][12].item() 
 
    # Create a new tensor with the updated age_academic
    updated_data_tensor = torch.tensor([[x0, x1, x2, x3, x4, x5, x6, x7,current_age_academic, x9 , x10, x11, x12]],  dtype=torch.float)
    #updated_data_tensor.requires_grad = True
    # Calculate the corresponding indice_h using your model function
    X = normalize(updated_data_tensor, p=2.0, dim = 1)
    output_tensor = model(X)

    # Append the tensors to the lists
   # input_tensors.append(torch.tensor([X[0][8].item()],  dtype=torch.float))
    input_tensors.append(torch.tensor([i],  dtype=torch.float))
    output_tensors.append(torch.tensor([output_tensor],  dtype=torch.float))
   
  # Convert lists to tensors
  input_tensor = torch.cat(input_tensors)
  output_tensor = torch.cat(output_tensors)
 
 

  return input_tensor, output_tensor





if __name__ == '__main__':

 input_size=13
 output_size = 1

 criterion = RMSELoss()
 

# Use a radial basis function with euclidean norm
 model = RBFLayer(in_features_dim=input_size, # input features dimensionality
               num_kernels=5,                 # number of kernels
               out_features_dim=output_size,            # output features dimensionality
               radial_function=rbf_gaussian,  # radial basis function used
               norm_function=l_norm)          # l_norm defines the \ell norm


 train_load_save_model(model, "/home/antonio.batista/antonio/Projeto_Antonio_Luciano/rbf_layer/rbf_0.pt")


#X = genfromtxt("/home/antonio/Desktop/Scientometrics/data/dataset.csv", delimiter=",",  skip_header=1, usecols={0,1,2,3,4,5,6,7,8,9,10,11,12})
#y = genfromtxt("/home/antonio/Desktop/Scientometrics/data/dataset.csv", delimiter=",",skip_header=1, usecols={15})  
	
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)	

#X_test = torch.tensor(X_test).float() 


 X_test = genfromtxt("/home/antonio.batista/antonio/Projeto_Antonio_Luciano/rbf_layer/X_test_0.csv", delimiter=",",  skip_header=0, usecols={0,1,2,3,4,5,6,7,8,9,10,11,12})

 X_test = torch.tensor(X_test).float() 

 indices = torch.randperm(len(X_test))[:300]
 scientists=X_test[indices] 
 torch.save(scientists,'scientists.pt')
 
 num_samples = 20  
 step_size = 1  

 data_tensor = scientists

 output_tensors1 = [] 
 input_tensors1 = []
# Iterate through each row using a for loop
 for row_index in range(data_tensor.shape[0]):
    row_data = data_tensor[row_index, :] 
    row_data = row_data.reshape(1, 13) 
    input_tensor1, output_tensor1 = generate_dataset(row_data, num_samples, step_size)
    output_tensors1.append(output_tensor1)   
    input_tensors1.append(input_tensor1)
    #print(row_data) 

 

#input_tensor1 = torch.tensor([1,2,3,4,5,6,7,8,9], dtype=torch.float)
#output_tensor1 = torch.tensor([4,8,12,16,20,24,28,32,36], dtype=torch.float)
 torch.save(input_tensors1, 'input_tensor.pt')
 torch.save(output_tensors1, 'output_tensor.pt')



