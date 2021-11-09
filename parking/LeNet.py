import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(32)
])

batch_size = 8

data_dir = 'train_images'
image_datasets = datasets.ImageFolder(data_dir, transform=transform)
data_loader = torch.utils.data.DataLoader(image_datasets, batch_size=batch_size, shuffle = True, num_workers=4)

print(image_datasets)
classes = ('free', 'full')


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


dataiter = iter(data_loader)
images, labels = dataiter.next()
print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

imshow(torchvision.utils.make_grid(images))

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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

print(net)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(5):  # loop over the dataset multiple times
    print('epoch %d' % epoch)
    running_loss = 0.0
    for i, data in enumerate(data_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 20 == 19:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 20))
            running_loss = 0.0

print('Finished Training')
PATH = './my_LeNet.pth'
torch.save(net, PATH)
# je treba udelat konverzi, ktera bude fungovat peyi pytorchem