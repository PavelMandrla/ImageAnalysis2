import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from utils import imshow, train_net
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(32)
])

net = nn.Sequential(
    nn.Conv2d(3, 6, kernel_size=5, padding=0, stride=1),
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5, padding=0, stride=1),
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), # je třeba ho změnit, na něco jiného
    #nn.LazyLinear(120),
    nn.ReLU(),
    nn.Linear(120, 84),
    #nn.LazyLinear(84),
    nn.ReLU(),
    nn.Linear(84, 2)
    #nn.LazyLinear(2)
)

train_net(net, transform, epochs=5)