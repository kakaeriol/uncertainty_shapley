import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.markers as mmarkers
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.ticker as plticker
import matplotlib.ticker
import random
#
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
class LinearRegression(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(in_features))
        self.bias = bias
        if bias:
            self.bias_term = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        x = x @ self.weights
        if self.bias:
            x += self.bias_term
        return x

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1,784, kernel_size=5)
        self.conv2 = nn.Conv1d(784, 32, kernel_size=5)
        self.conv3 = nn.Conv1d(32,64, kernel_size=5)
        self.fc1 = nn.Linear(3*3*64, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        #x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv3(x),2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.view(-1,3*3*64 )
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    def evaluation(self, test_loader, BATCH_SIZE=32,device='cpu'):
        correct = 0
        for test_imgs, test_labels in test_loader:
            test_imgs = Variable(test_imgs).float().to(device)
            output = self.forward(test_imgs).detach().cpu()
            test_labels = test_labels.detach().cpu().numpy()
            predicted = torch.max(output,1)[1].numpy()
            correct += (predicted == test_labels).sum()

        return float(correct)/(len(test_loader)*BATCH_SIZE)
        

class My_CNN(CNN):
    def __init__(self,train_loader, test_loader,EPOCHS=5, device='cpu'):
        super(My_CNN, self).__init__()
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.EPOCHS = EPOCHS
        BATCH_SIZE = 32
        self.model = CNN().to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        
        error = nn.CrossEntropyLoss()
        for epoch in range(self.EPOCHS):
            correct = 0
            for batch_idx, (X_batch, y_batch) in enumerate(self.train_loader):
                var_X_batch = Variable(X_batch).float().to(device)
                var_y_batch = Variable(y_batch).to(device)
                self.optimizer.zero_grad()
                output = self.model(var_X_batch)
                loss = error(output, var_y_batch)
                loss.backward()
                self.optimizer.step()
    
                # Total correct predictions
                predicted = torch.max(output.data, 1)[1] 
                correct += (predicted == var_y_batch).sum()
                #print(correct)
                if batch_idx % 50 == 0:
                    print('Epoch : {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Accuracy:{:.3f}%'.format(
                        epoch, batch_idx*len(X_batch), len(train_loader.dataset), 100.*batch_idx / len(train_loader), loss.data, float(correct*100) / float(BATCH_SIZE*(batch_idx+1))))
        self.acc = self.model.evaluation(test_loader,device=device)