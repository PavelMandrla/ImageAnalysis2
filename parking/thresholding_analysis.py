import cv2
import numpy as np
import glob
from utils import four_point_transform, load_parking_map, load_truth, draw_spot
from stats import Results


counts = {
    "free": {},
    "full": {}
}

free_images = [img for img in glob.glob("./train_images/free/*.png")]
for img_path in free_images:
    img = cv2.imread(img_path)

    blur_image = cv2.GaussianBlur(img, (3, 3), 0)
    gray_image = cv2.cvtColor(blur_image, cv2.COLOR_BGR2GRAY)
    edge_image = cv2.Canny(gray_image, 40, 120)

    edge_count = cv2.countNonZero(edge_image)
    if edge_count not in counts['free'].keys():
        counts['free'][edge_count] = 0
    counts['free'][edge_count] += 1

full_images = [img for img in glob.glob("./train_images/full/*.png")]
for img_path in full_images:
    img = cv2.imread(img_path)

    blur_image = cv2.GaussianBlur(img, (3, 3), 0)
    gray_image = cv2.cvtColor(blur_image, cv2.COLOR_BGR2GRAY)
    edge_image = cv2.Canny(gray_image, 40, 120)

    edge_count = cv2.countNonZero(edge_image)
    if edge_count not in counts['full'].keys():
        counts['full'][edge_count] = 0
    counts['full'][edge_count] += 1

with open("threshold_counts.dat", 'w+') as file:
    for i in range(max(counts['free'].keys())):
        if i not in counts['free']:
            file.write("%s %d\n" % (i, 0))
        else:
            file.write("%s %d\n" % (i, counts['free'][i]))

    file.write('\n\n')

    for i in range(max(counts['full'].keys())):
        if i not in counts['full']:
            file.write("%s %d\n" % (i, 0))
        else:
            file.write("%s %d\n" % (i, counts['full'][i]))
