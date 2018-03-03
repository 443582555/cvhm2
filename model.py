from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3,padding=1)
        self.conv3 = nn.Conv2d(16,120,kernel_size=7)

        self.conv21 = nn.Conv2d(3, 12, kernel_size=5)
        self.conv22 = nn.Conv2d(12, 128, kernel_size=5)
        self.conv23 = nn.Conv2d(128,256,kernel_size=5)


        self.mp = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120,84)
        self.fc2 = nn.Linear(84,10)
        self.fc3 = nn.Linear(256,120)
        
    def forward(self,x):
        in_size = x.size(0)
        out1 = self.relu(self.mp(self.conv1(x)))
        #print(out)
        out1 = self.relu(self.mp(self.conv2(out1)))
        #print(out)
        out1 = self.relu(self.conv3(out1))
        #print(out)
        out1 = out1.view(in_size, -1)
        #print(out)
        # out1 = self.relu(self.fc1(out1))
        # #print(out)
        # out1 = self.fc2(out1)

        out2 = self.relu(self.mp(self.conv21(x)))
        #print(out)
        out2 = self.relu(self.mp(self.conv22(out2)))
        #print(out)
        out2 = self.relu(self.conv23(out2))
        out2 = out2.view(in_size, -1)
        #print(out)
        out2 = self.relu(self.fc3(out2))
        out2 = out2+out1
        #print(out)
        out2 = self.relu(self.fc1(out2))
        out = self.fc2(out2)
        #print(out)
        return out