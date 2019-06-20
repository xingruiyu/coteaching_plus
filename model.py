from __future__ import print_function
import math
import torch
import torch.nn as nn
import torch.nn.init as init 
import torch.nn.functional as F
import torch.optim as optim

class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        
class CNN_small(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN_small, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def call_bn(bn, x):
    return bn(x)

class CNN(nn.Module):
    def __init__(self, input_channel=3, n_outputs=10, dropout_rate=0.25, momentum=0.1):
        self.dropout_rate = dropout_rate
        self.momentum = momentum 
        super(CNN, self).__init__()
        self.c1=nn.Conv2d(input_channel, 64,kernel_size=3,stride=1, padding=1)        
        self.c2=nn.Conv2d(64,64,kernel_size=3,stride=1, padding=1)        
        self.c3=nn.Conv2d(64,128,kernel_size=3,stride=1, padding=1)        
        self.c4=nn.Conv2d(128,128,kernel_size=3,stride=1, padding=1)        
        self.c5=nn.Conv2d(128,196,kernel_size=3,stride=1, padding=1)        
        self.c6=nn.Conv2d(196,16,kernel_size=3,stride=1, padding=1)        
        self.linear1=nn.Linear(256, n_outputs)
        self.bn1=nn.BatchNorm2d(64, momentum=self.momentum)
        self.bn2=nn.BatchNorm2d(64, momentum=self.momentum)
        self.bn3=nn.BatchNorm2d(128, momentum=self.momentum)
        self.bn4=nn.BatchNorm2d(128, momentum=self.momentum)
        self.bn5=nn.BatchNorm2d(196, momentum=self.momentum)
        self.bn6=nn.BatchNorm2d(16, momentum=self.momentum)

    def forward(self, x,):
        h=x
        h=self.c1(h)
        h=F.relu(call_bn(self.bn1, h))
        h=self.c2(h)
        h=F.relu(call_bn(self.bn2, h))
        h=F.max_pool2d(h, kernel_size=2, stride=2)

        h=self.c3(h)
        h=F.relu(call_bn(self.bn3, h))
        h=self.c4(h)
        h=F.relu(call_bn(self.bn4, h))
        h=F.max_pool2d(h, kernel_size=2, stride=2)

        h=self.c5(h)
        h=F.relu(call_bn(self.bn5, h))
        h=self.c6(h)
        h=F.relu(call_bn(self.bn6, h))
        h=F.max_pool2d(h, kernel_size=2, stride=2)

        h = h.view(h.size(0), -1)
        logit=self.linear1(h)
        return logit

class NewsNet(nn.Module):
    def __init__(self, weights_matrix, context_size=1000, hidden_size=300, num_classes=7):
        super(NewsNet, self).__init__()
        n_embed, d_embed = weights_matrix.shape
        self.embedding = nn.Embedding(n_embed, d_embed)
        self.embedding.weight.data.copy_(torch.Tensor(weights_matrix))
        self.avgpool=nn.AdaptiveAvgPool1d(16*hidden_size)
        self.fc1 = nn.Linear(16*hidden_size, 4*hidden_size)
        self.bn1=nn.BatchNorm1d(4*hidden_size)
        self.ac = nn.Softsign()
        self.fc2 = nn.Linear(4*hidden_size, hidden_size)
        self.bn2=nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):  
        embed = self.embedding(x) # input (128, 1000)
        embed = embed.detach()    # embed (128, 1000, 300)
        out = embed.view((1, embed.size()[0], -1)) # (1, 128, 300 000)
        out = self.avgpool(out)
        out = out.squeeze(0)
        out = self.fc1(out)
        out = self.bn1(out)
        out = self.ac(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.ac(out)
        out = self.fc3(out)
        return out

