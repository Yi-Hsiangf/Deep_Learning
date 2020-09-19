import numpy as np
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt




def read_bci_data():
    S4b_train = np.load('S4b_train.npz')
    X11b_train = np.load('X11b_train.npz')
    S4b_test = np.load('S4b_test.npz')
    X11b_test = np.load('X11b_test.npz')

    train_data = np.concatenate((S4b_train['signal'], X11b_train['signal']), axis=0)
    train_label = np.concatenate((S4b_train['label'], X11b_train['label']), axis=0)
    test_data = np.concatenate((S4b_test['signal'], X11b_test['signal']), axis=0)
    test_label = np.concatenate((S4b_test['label'], X11b_test['label']), axis=0)

    train_label = train_label - 1
    test_label = test_label -1
    train_data = np.transpose(np.expand_dims(train_data, axis=1), (0, 1, 3, 2))
    test_data = np.transpose(np.expand_dims(test_data, axis=1), (0, 1, 3, 2))

    mask = np.where(np.isnan(train_data))
    train_data[mask] = np.nanmean(train_data)

    mask = np.where(np.isnan(test_data))
    test_data[mask] = np.nanmean(test_data)

    print(train_data.shape, train_label.shape, test_data.shape, test_label.shape)

    return train_data, train_label, test_data, test_label

class EEGnet(torch.nn.Module):
    def __init__(self, activation_type):
        super(EEGnet, self).__init__()
        if activation_type == "ELU":
            activation_func = nn.ELU(alpha = 1.0)
        elif activation_type == "RELU":
            activation_func = nn.ReLU()
        elif activation_type == "L_RELU":
            activation_func = nn.LeakyReLU()
        self.first_conv = nn.Sequential(
            nn.Conv2d(
                in_channels = 1,  #RGB:channel數 
                out_channels = 16,  #Feature map的數量
                kernel_size = (1, 51), #Convolution filter大小
                stride = (1, 1),
                padding = (0, 25), #小訣竅：如果想讓輸入和輸出的size相同就把padding設成（kernel size - 1） /2
                bias = False
            ),
            nn.BatchNorm2d(
                16, #number of feature
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True
            )
        )
        
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(
                in_channels = 16, 
                out_channels = 32,
                kernel_size = (2, 1),
                stride = (1, 1),
                groups = 16,
                bias = False
            ),
            nn.BatchNorm2d(
                32,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True
            ),
            
            activation_func,
            
            nn.AvgPool2d(
                kernel_size = (1, 4),
                stride = (1, 4),
                padding = 0
            ),
            nn.Dropout(
                p = 0.25
            )
        )
        
        self.separableConv = nn.Sequential(
            nn.Conv2d(
                in_channels = 32,
                out_channels = 32,
                kernel_size = (1, 15),
                stride = (1, 1),
                padding=(0, 7),
                bias = False
            ),
            nn.BatchNorm2d(
                32,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True
            ),
            nn.ELU(
                alpha = 1.0
            ),
            nn.AvgPool2d(
                kernel_size=(1, 8),
                stride=(1, 8),
                padding=0
            ),
            nn.Dropout(p=0.25)
        )
        
        self.classify = nn.Sequential(
            nn.Linear(
                in_features=736,
                out_features=2,
                bias=True
            )
        )
        
    def forward(self, x):
        x = self.first_conv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = x.view(-1, 736)
        x = self.classify(x)

        return x

# Declare Loss Function
loss_func = nn.CrossEntropyLoss()
LR = 0.01
BATCH_SIZE = 64
EPOCH = 5

#ELU
# Declare Model
ELU_model = EEGnet("ELU")
ELU_model.cuda() #Use GPU
ELU_model = ELU_model.float()
# Declare Optimizer
ELU_optimizer =  torch.optim.Adam(ELU_model.parameters(), lr=LR)

#RELU
RELU_model = EEGnet("RELU")
RELU_model.cuda() 
RELU_model = RELU_model.float()
RELU_optimizer =  torch.optim.Adam(RELU_model.parameters(), lr=LR)

#L RELU
L_RELU_model = EEGnet("L_RELU") 
L_RELU_model.cuda()
L_RELU_model = L_RELU_model.float()
L_RELU_optimizer =  torch.optim.Adam(L_RELU_model.parameters(), lr=LR)

def train(model, train_loader, test_loader, optimizer):
    train_acccuracy_list = []
    test_accuracy_list = []
    for epoch in range(EPOCH):
        train_total = 0
        train_correct = 0
        for batch_idx, (data, label) in enumerate(train_loader):
            #print("batch_idx\n", batch_idx)
            #print("\ndata, taget\n", (data, target))
            data = data.cuda()
            label = label.cuda()
            output = model(data.float())
            loss = loss_func(output, label.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            _, predicted = torch.max(output, dim=1) #predicted代表類別的位置,         
            train_total += label.size(0)
            train_correct += (predicted == label).sum().item()
            #print(train_correct)
            
        
        accuracy = 100 * train_correct / train_total
        print("**********************")
        print("Epoch", epoch, " Train accuracy: %.2f %%" % accuracy)
        print("Epoch", epoch, " Train loss: %.4f" % loss.item())
        train_acccuracy_list.append(accuracy)
        
        #Test 
        test_total = 0
        test_correct = 0
        for batch_idx, (data, label) in enumerate(test_loader):
            #print("batch_idx\n", batch_idx)
            #print("\ndata, taget\n", (data, target))
            data = data.cuda()
            label = label.cuda()
            output = model(data.float())
    
            _, predicted = torch.max(output, dim=1) #predicted代表類別的位置,         
            test_total += label.size(0)
            test_correct += (predicted == label).sum().item()
            #print(train_correct)
            
        
        accuracy = 100 * test_correct / test_total
        test_accuracy_list.append(accuracy)
        print("Epoch", epoch, " Test accuracy: %.2f %%\n" % accuracy)
        print("**********************")       
        
    return train_acccuracy_list, test_accuracy_list
        

def accuracy_plot(ELU_train_acc, ELU_test_acc, RELU_train_acc, RELU_test_acc, L_RELU_train_acc, L_RELU_test_acc):
    epoch = [i for i in range(EPOCH)]
    fig, ax = plt.subplots()
    
    ax.plot(epoch, ELU_train_acc, label='ELU_train')
    ax.plot(epoch, ELU_test_acc, label='ELU_test')

    ax.plot(epoch, RELU_train_acc, label='ReLU_train')
    ax.plot(epoch, RELU_test_acc, label='ReLU_test')

    ax.plot(epoch, L_RELU_train_acc, label='Leaky_ReLU_train')
    ax.plot(epoch, L_RELU_test_acc, label='Leaky_ReLU_test')

    ax.set_xlabel('Epoch')  # Add an x-label to the axes.
    ax.set_ylabel('Accuracy%')  # Add a y-label to the axes.
    ax.set_title("EGGnet")  # Add a title to the axes.
    ax.legend()  # Add a legend.
    #plt.savefig("accuracy.png")
    

if __name__ == '__main__':   
    train_data, train_label, test_data, test_label = read_bci_data()
    
    train_data = torch.from_numpy(train_data)
    train_label = torch.from_numpy(train_label)
    train_dataset = torch.utils.data.TensorDataset(train_data, train_label)
    
    test_data = torch.from_numpy(test_data)
    test_label = torch.from_numpy(test_label)
    test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
    #Dataloader
    train_loader = torch.utils.data.DataLoader(
        dataset = train_dataset,
        batch_size = BATCH_SIZE,
        shuffle = True,
        num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        dataset = test_dataset,
        batch_size = BATCH_SIZE,
        shuffle = True,
        num_workers=2
    )
    
    #print(ELU_model)
    print("Training with ELU")
    ELU_train_acc, ELU_test_acc = train(ELU_model, train_loader, test_loader, ELU_optimizer)
    
    print("Training with RELU")
    RELU_train_acc, RELU_test_acc = train(RELU_model, train_loader, test_loader, RELU_optimizer)
    
    print("Training with Leaky RELU")
    L_RELU_train_acc, L_RELU_test_acc = train(L_RELU_model, train_loader, test_loader, L_RELU_optimizer)
    
    accuracy_plot(ELU_train_acc, ELU_test_acc, RELU_train_acc, RELU_test_acc, L_RELU_train_acc, L_RELU_test_acc)