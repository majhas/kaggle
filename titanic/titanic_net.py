import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, input_channel=None):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_channel, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
#
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.output(x)
        x = self.sigmoid(x)
        return x
