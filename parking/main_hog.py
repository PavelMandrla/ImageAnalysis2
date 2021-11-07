import sys
import cv2
import numpy as np
import math
import struct
from datetime import datetime
import glob
from utils import four_point_transform, load_parking_map, load_truth, draw_spot, load_train_images
from stats import Results

results = Results()
img_i = 0

IMG_SIZE = 96
#parametry bloku - velikost bloku, velikost buněk,...
hog = cv2.HOGDescriptor((IMG_SIZE, IMG_SIZE), (32, 32), (16, 16), (8, 8), 9, 1, -1, 0, 0.2, 1, 64, True)
svm = cv2.ml.SVM_create()  # můžeme použít třeba i k-nn
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setC(100.0)

svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 1000, 1e-6))

train_images, train_labels = load_train_images(IMG_SIZE, IMG_SIZE)
train_images = [hog.compute(res_image) for res_image in train_images]

print("train all: %d" % len(train_images))

svm.train(np.array(train_images), cv2.ml.ROW_SAMPLE, np.array(train_labels))
print("HOG training done")
svm.save("my_HOG_det.xml")

test_images = [img for img in glob.glob("test_images/*.jpg")]
test_images.sort()

pkm_coordinates = load_parking_map()
truth = load_truth()

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
        gray_image = cv2.cvtColor(res_image, cv2.COLOR_BGR2GRAY)

        hog_feature = hog.compute(gray_image)
        predict_label = svm.predict(np.array(hog_feature).reshape(1, -1))

        spot_class = 1 if predict_label[1] == 1 else 0
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





