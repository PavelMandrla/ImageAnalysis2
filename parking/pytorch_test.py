import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from PIL import Image
from utils import four_point_transform, load_parking_map, load_truth, draw_spot, load_train_images
import glob
from stats import Results
from torch.nn import functional as F

cv2.namedWindow("detection", 0)

class Inception(nn.Module):
    # `c1`--`c4` are the number of output channels for each path
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # Path 1 is a single 1 x 1 convolutional layer
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        self.p1_1_norm = nn.BatchNorm2d(c1)
        # Path 2 is a 1 x 1 convolutional layer followed by a 3 x 3
        # convolutional layer
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        self.p2_2_norm = nn.BatchNorm2d(c2[1])
        # Path 3 is a 1 x 1 convolutional layer followed by a 5 x 5
        # convolutional layer
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        self.p3_2_norm = nn.BatchNorm2d(c3[1])
        # Path 4 is a 3 x 3 maximum pooling layer followed by a 1 x 1
        # convolutional layer
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)
        self.p4_2_norm = nn.BatchNorm2d(c4)


    def forward(self, x):
        p1 = F.relu(self.p1_1_norm(self.p1_1(x)))
        p2 = F.relu(self.p2_2_norm(self.p2_2(F.relu(self.p2_1(x)))))
        p3 = F.relu(self.p3_2_norm(self.p3_2(F.relu(self.p3_1(x)))))
        p4 = F.relu(self.p4_2_norm(self.p4_2(self.p4_1(x))))
        # Concatenate the outputs on the channel dimension
        return torch.cat((p1, p2, p3, p4), dim=1)


#IMG_SIZE = 32   # LeNet
IMG_SIZE = 224  # AlexNet
#IMG_SIZE = 96 # GoogLeNet

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(IMG_SIZE),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


net = torch.load("my_AlexNet.pth")
net.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pkm_coordinates = load_parking_map()
truth = load_truth()
test_images = [img for img in glob.glob("test_images/*.jpg")]
test_images.sort()
result_list = []
results = Results()
img_i = 0

for img in test_images:
    one_park_image = cv2.imread(img)
    one_park_image_show = one_park_image.copy()
    for one_c in pkm_coordinates:
        pts = [((float(one_c[0])), float(one_c[1])),
               ((float(one_c[2])), float(one_c[3])),
               ((float(one_c[4])), float(one_c[5])),
               ((float(one_c[6])), float(one_c[7]))]

        warped_image = four_point_transform(one_park_image, np.array(pts))
        res_image = cv2.resize(warped_image, (IMG_SIZE, IMG_SIZE))
        one_img_rgb = cv2.cvtColor(res_image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(one_img_rgb)
        image_pytorch = transform(img_pil).to(device)
        image_pytorch = image_pytorch.unsqueeze(0)
        output_pytorch = net(image_pytorch)

        _, predicted = torch.max(output_pytorch, 1)
        #print(predicted)

        spot_class = predicted[0]
        draw_spot(one_park_image_show, pts, spot_class)

        results.eval(truth[img_i], spot_class)
        img_i += 1

    cv2.imshow('one_park_image', one_park_image_show)
    key = cv2.waitKey(0)
    if key == 27:  # exit on ESC
        break

print("accuracy: %f" % results.get_accuracy())
print("F1 score: %f" % results.get_f1())
print("MCC: %f" % results.get_mcc())
print("fp: %f, fn: %f" % (results.fp, results.fn))