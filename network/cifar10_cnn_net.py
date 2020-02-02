import torch.nn as nn
import torch.nn.functional as F

class cifar10_cnn_net(nn.Module):
    def __init__(self):
        super(cifar10_cnn_net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1)
        self.conv2 = nn.Conv2d(64, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1)
        self.drop_out = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(128*5*5, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)
        #print(x.size())
        x = x.view(-1, 3200)
        x = self.drop_out(x)
        x = F.relu(self.fc1(x))
        x = self.drop_out(x)
        x = F.relu(self.fc2(x))
        
        return F.log_softmax(x, dim=1)
