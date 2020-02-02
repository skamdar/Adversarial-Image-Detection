import torch.nn as nn
import torch.nn.functional as F

class adv_ann_net(nn.Module):
    def __init__(self):
        super(adv_ann_net, self).__init__()
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        
        x