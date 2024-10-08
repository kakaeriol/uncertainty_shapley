import os
import torch
import torch.nn as nn
import torch.nn.functional as F
# import pdb
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torcheval.metrics import R2Score
from sklearn.metrics import r2_score
from torch.nn import L1Loss

MODELS_DIR = "/external1/nguyenpham/model" 
MNIST_FLAT_DIM = 28 * 28
class LeNet(nn.Module):
    def __init__(self, pretrained=False, num_classes = 10, input_size=28, **kwargs):
        super(LeNet, self).__init__()
        suffix = f'dim{input_size}_nc{num_classes}'
        self.model_path = os.path.join(MODELS_DIR, f'lenet_mnist_{suffix}.pt')
        assert input_size in [28,32], "Can only do LeNet on 28x28 or 32x32 for now."

        feat_dim = 16*5*5 if input_size == 32 else 16*4*4
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        if input_size == 32:
            self.conv1 = nn.Conv2d(1, 6, 3)
            self.conv2 = nn.Conv2d(6, 16, 3)
        elif input_size == 28:
            self.conv1 = nn.Conv2d(1, 6, 5)
            self.conv2 = nn.Conv2d(6, 16, 5)
        else:
            raise ValueError()

        self._init_classifier()

        if pretrained:
            state_dict = torch.load(self.model_path)
            self.load_state_dict(state_dict)

    def _init_classifier(self, num_classes=None):
        """ Useful for fine-tuning """
        num_classes = self.num_classes if num_classes is None else num_classes
        self.classifier = nn.Sequential(
            nn.Linear(self.feat_dim, 120),  # 6*6 from image dimension
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        return self.classifier(x)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def save(self):
        state_dict = self.state_dict()
        torch.save(state_dict, self.model_path)

class MNIST_MLP(nn.Module):
    def __init__(
            self,
            input_dim=MNIST_FLAT_DIM,
            hidden_dim=98,
            output_dim=10,
            dropout=0.5,
    ):
        super(ClassifierModule, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.hidden  = nn.Linear(input_dim, hidden_dim)
        self.output  = nn.Linear(hidden_dim, output_dim)

    def forward(self, X, **kwargs):
        X = X.reshape(-1, self.hidden.in_features)
        X = F.relu(self.hidden(X))
        X = self.dropout(X)
        X = F.softmax(self.output(X), dim=-1)
        return X

class MNIST_CNN(nn.Module):
    def __init__(self, input_size=28, dropout=0.3, nclasses=10, pretrained=False):
        super(MNIST_CNN, self).__init__()
        self.nclasses = nclasses
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv2_drop = nn.Dropout2d(p=dropout)
        self.fc1 = nn.Linear(1600, 100) # 1600 = number channels * width * height
        self.logit = nn.Linear(100, self.nclasses)
        self.fc1_drop = nn.Dropout(p=dropout)
        suffix = f'dim{input_size}_nc{nclasses}'
        self.model_path = os.path.join(MODELS_DIR, f'cnn_mnist_{suffix}.pt')
        if pretrained:
            state_dict = torch.load(self.model_path)
            self.load_state_dict(state_dict)

    def forward(self, x):
        x = torch.relu(F.max_pool2d(self.conv1(x), 2))
        x = torch.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3))
        x = torch.relu(self.fc1_drop(self.fc1(x)))
        x = self.logit(x)
        x = F.log_softmax(x, dim=-1)
        return x

    def save(self):
        state_dict = self.state_dict()
        torch.save(state_dict, self.model_path)
        
class Net(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=100, pretrained=False):
        # super(Net, self).__init__()
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        suffix = f'dim{input_size}_nc{output_size}'
        self.model_path = os.path.join(MODELS_DIR, f'net_{suffix}.pt')
      
    def count_params(self):
        return sum([p.view(-1).shape[0] for p in self.parameters()])
    
    def forward(self, x):
        h = self.linear1(x).relu()
        h = h + self.linear2(h).relu()
        hx= self.linear3(h)
        return F.log_softmax(hx, dim=1)

    def train_model(self, train_loader, val_loader=None, num_epochs=10, learning_rate=0.001, device = None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            self.train()
            running_loss = 0.0
            running_corrects = 0
            total = 0

            for inputs, labels in train_loader:
                inputs = inputs.to(device).float()
                labels = labels.type(torch.LongTensor)   
                labels = labels.to(device)
                optimizer.zero_grad()

                outputs = self(inputs)
                loss = criterion(outputs, labels)
                

                loss.backward()
                optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total += labels.size(0)

            epoch_loss = running_loss / total
            epoch_acc = running_corrects.double() / total

            # print(f'Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}')

            if val_loader:
                return self.evaluate(val_loader, device)

    def evaluate(self, data_loader, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.eval()
        running_loss = 0.0
        running_corrects = 0
        total = 0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.to(device).float()
                labels = labels.to(device)
                outputs = self(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = running_corrects.double() / total
        # print(f'Validation - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}')
        return epoch_acc, epoch_loss
      
    def save(self):
        state_dict = self.state_dict()
        torch.save(state_dict, self.model_path)


class ResNet_18_Classifier(nn.Module): ## use for cifar
    def __init__(self, num_classes=10):
        super(ResNet_18_Classifier, self).__init__()
        self.model = torchvision.models.resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model_path = os.path.join(MODELS_DIR, f'resnes_18_classifier.pt')
        

    def forward(self, x):
        return self.model(x)

    def train_model(self, train_loader, val_loader=None, num_epochs=10, learning_rate=0.001, device = None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            self.train()
            running_loss = 0.0
            running_corrects = 0
            total = 0

            for inputs, labels in train_loader:
                inputs = inputs.to(device).float()
                inputs = inputs.permute(0, 3, 1, 2)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total += labels.size(0)

            epoch_loss = running_loss / total
            epoch_acc = running_corrects.double() / total

            # print(f'Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}')

            if val_loader:
                return self.evaluate(val_loader)

    def evaluate(self, data_loader, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.eval()
        running_loss = 0.0
        running_corrects = 0
        total = 0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.to(device).float()
                inputs = inputs.permute(0, 3, 1, 2)
                labels = labels.to(device)

                outputs = self(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = running_corrects.double() / total
        # print(f'Validation - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}')
        return epoch_acc, epoch_loss

    def predict(self, data_loader):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.eval()
        predictions = []

        with torch.no_grad():
            for inputs in data_loader:
                if isinstance(inputs, (list, tuple)):
                    inputs = inputs[0]
                # inputs = inputs.to(device)
                inputs = inputs.permute(0, 3, 1, 2).float()

                outputs = self(inputs)
                _, preds = torch.max(outputs, 1)
                predictions.extend(preds.cpu().numpy())

        return predictions

    def save(self):
        state_dict = self.state_dict()
        torch.save(state_dict, self.model_path)


class CNNRegressor(nn.Module): # use for calihouse
    def __init__(self, input_channels, num_outputs=1):
        super(CNNRegressor, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=1),
            nn.ReLU(),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 1, 128),
            nn.ReLU(),
            nn.Linear(128, num_outputs)
        )
        suffix = f'dim{input_channels}_nc{num_outputs}'
        self.model_path = os.path.join(MODELS_DIR, f'CNN_{suffix}.pt')

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x

    def train_model(self, train_loader, val_loader=None, num_epochs=50, learning_rate=0.001, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
    
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
    
        for epoch in range(num_epochs):
            self.train()
            running_loss = 0.0
            total_samples = 0
    
            for inputs, targets in train_loader:
                inputs = inputs.to(device).float()
                targets = targets.to(device).float()
                if inputs.dim() == 2:
                    inputs = inputs.unsqueeze(2)
                optimizer.zero_grad()
    
                outputs = self(inputs).squeeze()
    
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
    
                running_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)
    
            epoch_loss = running_loss / total_samples
            # print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
    
            if val_loader:
                return self.evaluate(val_loader, device)

    def evaluate(self, data_loader, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.eval()
        criterion = R2Score().to(device)
    
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(device).float()
                targets = targets.to(device).float()
                if inputs.dim() == 2:
                    inputs = inputs.unsqueeze(2)
                outputs = self(inputs).squeeze()
                criterion.update(outputs, targets)
    
        epoch_r2 = criterion.compute().item()
        return None, epoch_r2

    def predict(self, data_loader):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.eval()
        predictions = []

        with torch.no_grad():
            for inputs in data_loader:
                if isinstance(inputs, (list, tuple)):
                    inputs = inputs[0]
                inputs = inputs.to(device)

                outputs = self(inputs).squeeze()
                predictions.extend(outputs.cpu().numpy())

        return predictions
    def save(self):
        state_dict = self.state_dict()
        torch.save(state_dict, self.model_path)


import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import R2Score

class MLPRegressor(nn.Module):
    def __init__(self, input_size, num_outputs=1):
        super(MLPRegressor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, num_outputs)
        )

    def forward(self, x):
        return self.model(x)
    def train_model(self, train_loader, val_loader=None, num_epochs=50, learning_rate=0.001, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
    
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
    
        for epoch in range(num_epochs):
            self.train()
            running_loss = 0.0
            total_samples = 0
    
            for inputs, targets in train_loader:
                inputs = inputs.to(device).float()
                inputs = inputs.unsqueeze(1)
                targets = targets.to(device).float()
                optimizer.zero_grad()
    
                outputs = self(inputs)
                
                
                outputs = outputs.squeeze(-1).squeeze(-1)  # Adjust outputs shape
    
                # Ensure outputs and targets have the same shape
    
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
    
                running_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)

    
            epoch_loss = running_loss / total_samples
            # print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

            # if val_loader:
            #     _, val_r2 = self.evaluate(val_loader, device)
            #     print(f'Validation R2 Score: {val_r2:.4f}')
    
            if val_loader:
                return self.evaluate(val_loader, device)

    def evaluate(self, data_loader, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.eval()
        criterion = R2Score().to(device)

        inputs, targets = data_loader.dataset.data,  data_loader.dataset.targets.cpu()
        inputs = inputs.unsqueeze(1).to(device).float()
        output = self(inputs).squeeze(-1).squeeze(-1).cpu()
        # print(output.shape, targets.shape)
        epoch_r2=r2_score(output.detach().numpy(),targets)
        # with torch.no_grad():
        #     for inputs, targets in data_loader:
        #         inputs = inputs.to(device).float()
        #         inputs = inputs.unsqueeze(1)
        #         targets = targets.to(device).float()
        #         outputs = outputs.squeeze(-1)
        #         criterion.update(outputs, targets)
    
        # epoch_r2 = criterion.compute().item()
        return None, epoch_r2
        
