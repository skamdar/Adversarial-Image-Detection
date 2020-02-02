import torch.nn as nn
import torch.nn.functional as F

class ann_net(nn.Module):
    def __init__(self):
        super(ann_net, self).__init__()
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)
       
    

'''import torch.nn as nn
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
            