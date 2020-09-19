import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

# Define Hyper parameter
sequence_length = 28
input_size = 28
hidden_size = 128 #how many features we want
num_layers = 2
num_classes = 10
batch_size =100
epochs = 2
learning_rate = 0.01

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root = '/c/Users/asus/Desktop/Deep_Learning/MNIST_RNN', train=True, transform=transforms.ToTensor(), download = True)
test_dataset = torchvision.datasets.MNIST(root = '/c/Users/asus/Desktop/Deep_Learning/MNIST_RNN', train=False, transform=transforms.ToTensor(), download = True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)

#print(train_dataset.train_data.size()) #torch.Size([60000, 28, 28])
#plt.imshow(train_dataset.train_data[0].numpy(),cmap='gray')
#plt.title('%i' % train_dataset.train_labels[0])
#plt.show()
#print(train_dataset[0][0].shape) #1 * 28 * 28
#print(train_dataset[0][1]) a integer


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
        
    def forward(self, x):
        # Set inital hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        #print("input ", x.shape) torch.Size([100, 28, 28])
        # LSTM INPUT
        #   input (batch, seq_len, input_size)
        #   h_0 (num_layers * num_directions, batch, hidden_size)
        #   c_0 (num_layers * num_directions, batch, hidden_size)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        
        
        # LSTM OUTPUT
        #   output (batch, seq_len, num_directions * hidden_size)
        #   h_n (num_layers * num_directions, batch, hidden_size)
        #   c_n (num_layers * num_directions, batch, hidden_size)
        #print("output " ,out.shape) torch.Size([100, 28, 128])
        #print("out[:, -1, :] ", out[:, -1, :].shape) torch.Size([100, 128]
        out = self.fc(out[:, -1, :])
        return out
    
model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



# Train the model
total_step = len(train_loader)
for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):
        #print("before ", images.shape) torch.Size([100, 1, 28, 28])
        images = images.reshape(-1, sequence_length, input_size).to(device)
        #print("after ", images.shape) torch.Size([100, 28, 28])
        labels = labels.to(device)
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1,  epochs, i+1, total_step, loss.item()))
        
        
# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
       
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total)) 

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')      

        
        
