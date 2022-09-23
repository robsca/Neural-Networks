#1. imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transform

#2. create Model
class Bidirectional_RNN(nn.Module):
    def __init__(self, input_shape, hidden_size, n_layers, num_classes) -> None:
        super(Bidirectional_RNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        self.LSTM = nn.LSTM(input_shape, hidden_size, n_layers, batch_first = True,
                             bidirectional = True)

        self.fc1 = nn.Linear(hidden_size*2, num_classes) #x2 because of the bidirection
        
    def forward(self, x):
        hidden_state = torch.zeros(self.n_layers*2, x.size(0), self.hidden_size).to(device)
        cell_state = torch.zeros(self.n_layers*2, x.size(0), self.hidden_size).to(device)

        out, _ = self.LSTM(x, (hidden_state, cell_state))
        out = self.fc1(out[:, -1,:])
        return out

#3. create device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#4. set Hyperparameters
input_shape = 28
sequence_length = 28
n_layers = 2
hidden_size = 256
num_classes = 10

num_epochs = 1
batch_size = 64
learning_rate = 0.001

# create NN
model = Bidirectional_RNN(input_shape, hidden_size, n_layers, num_classes).to(device)

#5. Optimizer and loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#6. Load Dataset
train_set = datasets.MNIST('dataset/', train = True, transform = transform.ToTensor(), download = True)
test_set = datasets.MNIST('dataset/', train = False, transform = transform.ToTensor(), download = True)
train_loader = DataLoader(train_set, batch_size, shuffle = True )
test_loader = DataLoader(test_set, batch_size, shuffle = True )

#7. training
for epoch in range(num_epochs):
    print(epoch,'/',num_epochs)
    for batch, (data, target) in enumerate(train_loader):
        data = data.to(device).squeeze(1)
        target = target.to(device)

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
            x = x.to(device).squeeze(1)
            y = y.to(device)

            # forward
            score = model(x)
            _, predictions = score.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        return f'Accuracy of model: {(num_correct / num_samples)*100} %'

print('Training Set',check_accuracy(train_loader, model))
print('Test Set',check_accuracy(test_loader, model))