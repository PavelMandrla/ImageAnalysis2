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

cv2.namedWindow("detection", 0)

#IMG_SIZE = 32   # LeNet
IMG_SIZE = 224  # AlexNet

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(IMG_SIZE)
])


#net = torch.load("my_LeNet.pth")
net = torch.load("my_AlexNet.pth")
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
        print(predicted)

        spot_class = predicted[0]
        draw_spot(one_park_image_show, pts, spot_class)

        results.eval(truth[img_i], spot_class)
        img_i += 1

    cv2.imshow('one_park_image', one_park_image_show)
    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break

print("accuracy: %f" % results.get_accuracy())
print("F1 score: %f" % results.get_f1())
print("MCC: %f" % results.get_mcc())