import torch.nn as nn
import torch.nn.functional as F

class ann_net(nn.Module):
    def __init__(self):
        super(ann_net, self).__init__()
        self.fc1 = nn.Linear(784, 300)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(300, 128)
        self.relu2 = nn.ReLU()
        self.dout = nn.Dropout(0.2)
        self.fc3 = nn.Linear(128, 2)
        
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dout(x)
        x = self.fc3(x)
        
        return F.log_softmax(x, dim=1)
        

'''import torch.nn as nn
import torch.nn.functional as F

class ann_net(nn.Module):
    def __init__(self):
        super(ann_net, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)
        
    

import torch.nn as nn
import torch.nn.functional as F

class ann_net(nn.Module):
    def __init__(self):
        super(ann_net, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)'''
            