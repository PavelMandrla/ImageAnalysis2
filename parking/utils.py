import numpy as np
import cv2
import glob

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import time


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped


def load_parking_map():
    pkm_coordinates = []
    with open('parking_map_python.txt', 'r') as pkm_file:
        pkm_lines = pkm_file.readlines()

        for line in pkm_lines:
            st_line = line.strip()
            sp_line = list(st_line.split(" "))
            pkm_coordinates.append(sp_line)
    return pkm_coordinates


def load_truth():
    with open("groundtruth.txt") as truth_file:
        truth = [int(x) for x in truth_file.read().splitlines()]
    return truth


def load_train_images(w, h):
    train_labels_list = []
    train_images_list = []
    train_images_full = [img for img in glob.glob("train_images/full/*.png")]
    train_images_free = [img for img in glob.glob("train_images/free/*.png")]
    for i in range(len(train_images_full)):
        one_park_image = cv2.imread(train_images_full[i], 0)
        res_image = cv2.resize(one_park_image, (w, h))
        train_images_list.append(res_image)
        train_labels_list.append(1)

    for i in range(len(train_images_free)):
        one_park_image = cv2.imread(train_images_free[i], 0)
        res_image = cv2.resize(one_park_image, (w, h))
        train_images_list.append(res_image)
        train_labels_list.append(0)

    return train_images_list, train_labels_list


def draw_spot(img, pts, cls):
    spot_color = (0, 255, 0) if cls == 1 else (255, 0, 0)

    int_points = [(int(a), int(b)) for a, b in pts]

    cv2.line(img, int_points[0], int_points[1], spot_color, 5)
    cv2.line(img, int_points[1], int_points[2], spot_color, 5)
    cv2.line(img, int_points[2], int_points[3], spot_color, 5)
    cv2.line(img, int_points[3], int_points[0], spot_color, 5)
    cv2.line(img, int_points[0], int_points[2], spot_color, 5)
    cv2.line(img, int_points[1], int_points[3], spot_color, 5)


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def train_net(net, transform, net_name, epochs=5, batch_size=8):
    data_dir = 'train_images'
    image_datasets = datasets.ImageFolder(data_dir, transform=transform)
    data_loader = torch.utils.data.DataLoader(image_datasets, batch_size=batch_size, shuffle=True, num_workers=4)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    print(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    start = time.time()
    for epoch in range(epochs):  # loop over the dataset multiple times
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
            if i % 20 == 19:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 20))

                running_loss = 0.0
    end = time.time()
    print('Finished Training')
    print(end - start)
    PATH = './my_%s.pth' % net_name # './my_LeNet.pth'
    torch.save(net, PATH)
