import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision import datasets
import torch.nn.functional as functional
import torch.optim as optim
import numpy as np
from torchmetrics import Precision, Recall, F1Score, Accuracy

##############################

def load_data():
  #download training data from open datasets
  training_data = datasets.FashionMNIST(root = "data", train = True, download = True, transform = ToTensor())

  #download test data from open datasets
  test_data = datasets.FashionMNIST(root = "data", train = False, download = True, transform = ToTensor())

  return training_data, test_data

##############################

def create_dataloaders (training_data, test_data, batch_size = 64):
  training_dataloader = DataLoader(training_data, batch_size = batch_size)
  test_dataloader = DataLoader(test_data, batch_size = batch_size)

  return training_dataloader, test_dataloader

##############################

device = "cuda" if torch.cuda.is_available() else "CPU"
#print(f"Using {device} device")

##############################

class cs21m004_cnn (nn.Module):
  def __init__(self, n_input = 1, n_output = 10, n_channels = 64):
    super().__init__()
    self.conv_layer1 = nn.Conv2d(n_input, n_channels, kernel_size = 5)
    self.conv_layer2 = nn.Conv2d(n_channels, n_output, kernel_size = 3)
    self.flatten = nn.Flatten()
    self.fc = nn.Linear(4840, 10)
    self.softmax = nn.Softmax(dim = 1)

  def forward(self, x):
    x = self.conv_layer1(x)
    x = functional.relu(x)
    x = self.conv_layer2(x)
    x = self.flatten(x)
    x = self.fc(x)
    x = self.softmax(x)
    return x
  
##############################

class cs21m004_cnn_advanced (nn.Module):
  def __init__(self, config, pic_length, pic_breadth):
    super(cs21m004_cnn_advanced, self).__init__()
    self.conv_in_layer = nn.Conv2d(1, config[0][0], kernel_size = (2,2), stride = 1, padding = "same")
    self.conv_layer = []
    for item in config:
      self.conv_layer.append(nn.Conv2d(item[0], item[1], kernel_size = item[2], stride = 1, padding = "same"))
    self.conv_out_layer = nn.Conv2d(config[-1][1], 10, kernel_size = (3,3), stride = 1, padding = "same")
    self.flatten = nn.Flatten()
    self.fc = nn.Linear(10 * pic_length * pic_breadth, 10)
    self.softmax = nn.Softmax(dim = 1)

  def forward(self, x, config):
    x = self.conv_in_layer(x)
    i = 0
    for i in range(len(config)):
      x = self.conv_layer[i](x)
      i = i + 1
    x = self.conv_out_layer(x)
    x = self.flatten(x)
    x = self.fc(x)
    x = self.softmax(x)
    return x
  
##############################

def load_model():
  model = cs21m004_cnn()
  return model

##############################

def load_model_advanced(config, pic_length, pic_breadth):
  model = cs21m004_cnn_advanced(config, pic_length, pic_breadth)
  return model

##############################

def get_optimiser(model):
  optimiser = optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)
  return optimiser

##############################

def _train_network (train_loader, model, optimizer, loss_fn):
  num_batches = len(train_loader)
  
  training_loss = 0.0
  for i, data in enumerate(train_loader, start = 0):
    input, label = data
    #input = input.to(device)
    #label = label.to(device)

    #forward + backward + optimize
    output = model(input)
    label_1hot = functional.one_hot(label, 10)
    loss = loss_fn (output, label_1hot)
    training_loss += loss.item()

    #flush out parameter gradients
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  print(f"Average loss: {training_loss / num_batches}")
  
##############################

def _train_network_advanced (train_loader, config, model, optimizer, loss_fn):
  num_batches = len(train_loader)
  
  training_loss = 0.0
  for i, data in enumerate(train_loader, start = 0):
    input, label = data
    #input = input.to(device)
    #label = label.to(device)

    #forward + backward + optimize
    output = model(input, config)
    label_1hot = functional.one_hot(label, 10)
    loss = loss_fn (output, label_1hot)
    training_loss += loss.item()

    #flush out parameter gradients
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  print(f"Average loss: {training_loss / num_batches}")  

##############################

def train_model(train_data_loader, model, optimiser, loss_fn, epochs = 10):
  for e in range(epochs):
    print(f"Epoch {e+1}: ")
    _train_network(train_data_loader, model, optimiser, loss_fn)

  print("Finished training")
  
##############################

def train_model_advanced(train_data_loader, model, optimiser, loss_fn, epochs = 10):
  for e in range(epochs):
    print(f"Epoch {e+1}: ")
    _train_network_advanced(train_data_loader, config, model, optimiser, loss_fn)

  print("Finished training")
  
##############################

def _test_network(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            #X, y = X.to(device), y.to(device)
            tmp = torch.nn.functional.one_hot(y, num_classes= 10)
            pred = model(X)
            test_loss += loss_fn(pred, tmp).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")
    #precision_recall_fscore_support(y_ground, y_pred, average='macro')
    accuracy1 = Accuracy()
    print('Accuracy :', accuracy1(pred,y))
    precision = Precision(average = 'macro', num_classes = 10)
    print('precision :', precision(pred,y))

    recall = Recall(average = 'macro', num_classes = 10)
    print('recall :', recall(pred,y))
    f1_score = F1Score(average = 'macro', num_classes = 10)
    print('f1_score :', f1_score(pred,y), '\n')
    return accuracy1,precision, recall, f1_score
  
##############################

def test_model(dataloader, model, loss_fn, epochs = 10):
  for e in range(epochs):
    print(f"Epochs {e+1}: ")
    _test_network(dataloader, model, loss_fn)

  print("Finished Testing.")
  
##############################

