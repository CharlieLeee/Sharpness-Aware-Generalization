import torch.nn as nn
import torch.nn.functional as F



class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3072, 100)
        self.fc2 = nn.Linear(100, 10)
    
    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class SimpleConv(nn.Module):
    def __init__(self):
        super().__init__()
        nb_hidden = 200
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=2)
        self.fc1 = nn.Linear(4 * 4 * 64, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(-1, 4 * 4 * 64)))
        x = self.fc2(x)
        return x