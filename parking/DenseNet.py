import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import models as models
import matplotlib.pyplot as plt
import numpy as np
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(96)
])

batch_size = 8

data_dir = 'train_images'
image_datasets = datasets.ImageFolder(data_dir, transform=transform)
data_loader = torch.utils.data.DataLoader(image_datasets, batch_size=batch_size, shuffle = True, num_workers=4)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(image_datasets)
classes = ('free', 'full')


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


dataiter = iter(data_loader)
images, labels = dataiter.next()
print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

#imshow(torchvision.utils.make_grid(images))

resnet18 = models.resnet18(pretrained=True)

for name, child in resnet18.named_children():
    print("name: %s" % name)

#for param in resnet18.parameters():
    #param.requires_grad = False

num_ftrs = resnet18.fc.in_features
resnet18.fc = nn.Linear(num_ftrs, 2)


for name, param in resnet18.named_parameters():
    print(" %s" % name)
    param.requires_grad = False
    if ("fc" in name) or ("layer4" in name):
        param.requires_grad = True
params_to_update = [param for param in resnet18.parameters() if param.requires_grad==True]
print(len(params_to_update))

resnet18 = resnet18.to(device)


criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(resnet18.fc.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
for epoch in range(10):  # loop over the dataset multiple times
    #print('epoch %d' % epoch)
    running_loss = 0.0
    for i, data in enumerate(data_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = resnet18(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 20 == 19:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 20))
            running_loss = 0.0

print('Finished Training')
PATH = './my_ResNet_pretrained.pth'
torch.save(resnet18, PATH)
