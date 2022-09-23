#1. imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transform
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

#2. create Model
class NeuralNetwork(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_shape, 50)
        self.fc2 = nn.Linear(50, num_classes)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#3. create device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#4. set Hyperparameters
learning_rate = 0.002
input_shape = 784
num_classes = 10
num_epochs = 10
batch_size = 64

# create NN
model = NeuralNetwork(input_shape, num_classes).to(device)

# create Functions to save and load model
def save_checkpoint(checkpoint, path = 'checkpoint.pth.tar'):
    print('Saving Checkpoint')
    torch.save(checkpoint, path)
    
def load_checkpoint(checkpoint):
    print('Loading Checkpoint')
    model.load_state_dict(checkpoint['parameters'])
    optimizer.load_state_dict(checkpoint['optimizer'])

#5. Optimizer and loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#6. Load Dataset
train_set = datasets.MNIST('dataset/', train = True, transform = transform.ToTensor(), download = True)
test_set = datasets.MNIST('dataset/', train = False, transform = transform.ToTensor(), download = True)
train_loader = DataLoader(train_set, batch_size, shuffle = True )
test_loader = DataLoader(test_set, batch_size, shuffle = True )

#7. training
load = False
if load:
    load_checkpoint(torch.load('checkpoint.pth.tar'))

for epoch in range(num_epochs):
    print(epoch+1,'/',num_epochs)

    if epoch+1 == num_epochs:
        checkpoint = {'parameters' : model.state_dict(), 'optimizer' : optimizer.state_dict()}
        save_checkpoint(checkpoint)

    for batch, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)

        # reshape to make same size as input size
        data = data.reshape(data.shape[0], -1)

        # forward
        score = model(data)
        loss = criterion(score, target)

        # backward
        optimizer.zero_grad()
        loss.backward()

        #
        optimizer.step()

#8. check accuracy
def check_accuracy(loader, model):
    num_correct = 0 
    num_samples = 0

    with torch.no_grad():
        print('Checking Accuracy')
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            x = x.reshape(x.shape[0], -1)
            # forward
            score = model(x)
            _, predictions = score.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        return f'Accuracy of model: {(num_correct / num_samples)*100} %'

print('Training Set',check_accuracy(train_loader, model))
print('Test Set',check_accuracy(test_loader, model))