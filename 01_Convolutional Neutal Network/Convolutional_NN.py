# Imports
import re
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# 1. Create CNN
class ConvolutionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride = (1,1), padding=(1,1)) # same convolution
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride = (2,2)) 
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride = (1,1), padding=(1,1)) # same convolution
        self.fc1 = nn.Linear(16*7*7, 10)
    
    def forward(self, x):       
        x = F.relu(self.conv1(x))   # 1,  8, 28, 28
        x = self.pool(x)            # 1,  8, 14, 14
        x = F.relu(self.conv2(x))   # 1, 16, 14, 14
        x = self.pool(x)            # 1, 16,  7,  7

        # Fully connected
        x = x.reshape(x.shape[0],-1)
        x = self.fc1(x)
        return x

# 2. Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 3. Hyperparameters
learning_rate = 0.001
batch_size = 1

# 4. Initialize NN -> send to device
model = ConvolutionNet().to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Load data
train_dataset = datasets.MNIST(root = 'dataset/', train = True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle=True)

test_dataset = datasets.MNIST(root = 'dataset/', train = False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = True)

# Train NN
num_epochs = 2
for epoch in range(num_epochs):
    print(f'TRAINING : {epoch}/{num_epochs}')
    for batch,(data, target) in enumerate(train_loader):
        
        #find data and target
        data = data.to(device)
        target = target.to(device)
        
        # forward
        score = model(data)

        # loss
        loss = criterion(score, target)

        # backward
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

# Check accuracy on training & test to see how good our model
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        print('Checking accuracy:')
        for x,y in loader:
            x = x.to(device)
            y = y.to(device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        return(f'Accuracy {(num_correct/num_samples)*100}')

print(f'Test Data: {check_accuracy(test_loader, model)}')
print(f'Train Data: {check_accuracy(train_loader, model)}')