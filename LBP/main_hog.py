#!/usr/bin/python

import sys
import cv2
import numpy as np
import math
import struct
from datetime import datetime
import glob


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


def get_F1_score(results):
    precision = float(results["tp"]) / float(results["tp"] + results["fp"])
    recall = float(results["tp"]) / float(results["fn"] + results["tn"])

    return 2.0 * float(precision * recall) / float(precision + recall)

def get_MCC(results):
    top = float(results["tp"] * results["tn"] - results["fp"] + results["fn"])
    bottom = float((results["tp"] + results["fp"]) * (results["tp"] + result["fn"]) * (results["tn"] + results["fp"]) * (results["tn"] + results["fn"]))
    return top / math.sqrt(bottom)

def main(argv):
    with open("groundtruth.txt") as truth_file:
        truth = [int(x) for x in truth_file.read().splitlines()]
    img_i = 0

    results = {
        "fp": 0,
        "fn": 0,
        "tp": 0,
        "tn": 0
    }


    cv2.namedWindow("detection", 0)

    pkm_file = open('parking_map_python.txt', 'r')
    pkm_lines = pkm_file.readlines()
    pkm_coordinates = []

    for line in pkm_lines:
        st_line = line.strip()
        sp_line = list(st_line.split(" "))
        pkm_coordinates.append(sp_line)

    test_images = [img for img in glob.glob("test_images/*.jpg")]
    test_images.sort()
    for img in test_images:
        one_park_image = cv2.imread(img)
        for one_c in pkm_coordinates:
            pts = [((float(one_c[0])), float(one_c[1])),
                   ((float(one_c[2])), float(one_c[3])),
                   ((float(one_c[4])), float(one_c[5])),
                   ((float(one_c[6])), float(one_c[7]))]
            # print(pts)
            # https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
            warped_image = four_point_transform(one_park_image, np.array(pts))
            res_image = cv2.resize(warped_image, (80, 80))

            blur_image = cv2.GaussianBlur(res_image, (3, 3), 0)
            gray_image = cv2.cvtColor(blur_image, cv2.COLOR_BGR2GRAY)
            edge_image = cv2.Canny(gray_image, 40, 120)

            #print(cv2.countNonZero(edge_image))
            spot_class = 1 if cv2.countNonZero(edge_image) > 350 else 0
            spot_color = (0, 255, 0) if spot_class == 1 else (255, 0, 0)

            int_points = [(int(a), int(b)) for a, b in pts]
            cv2.line(one_park_image, int_points[0], int_points[1], spot_color, 5)
            cv2.line(one_park_image, int_points[1], int_points[2], spot_color, 5)
            cv2.line(one_park_image, int_points[2], int_points[3], spot_color, 5)
            cv2.line(one_park_image, int_points[3], int_points[0], spot_color, 5)
            cv2.line(one_park_image, int_points[0], int_points[2], spot_color, 5)
            cv2.line(one_park_image, int_points[1], int_points[3], spot_color, 5)

            cv2.imshow('blur_image', blur_image)
            cv2.imshow('res_image', res_image)
            cv2.imshow('edge_image', edge_image)
            #cv2.waitKey(200)
            #roi = img[y:y+h, x:x+w]
            cv2.imshow('one_park_image', one_park_image)

            if truth[img_i]:
                if spot_class:
                    results["tp"] += 1
                else:
                    results["fn"] += 1
            else:
                if spot_class:
                    results["fp"] += 1
                else:
                    results["tn"] += 1

            img_i += 1
        print(results)
        key = cv2.waitKey(0)
        if key == 27:  # exit on ESC
            break

    return results


if __name__ == "__main__":
    train_images_full = [img for img in glob.glob("train_images/full/*.png")]
    train_images_free = [img for img in glob.glob("train_images/free/*.png")]

    IMG_SIZE = 96
    #parametry bloku - velikost bloku, velikost buněk,...
    hog = cv2.HOGDescriptor((IMG_SIZE, IMG_SIZE), (32, 32), (16, 16), (8, 8), 9, 1, -1, 0, 0.2, 1, 64, True)
    svm = cv2.ml.SVM_create()  # můžeme použít třeba i k-nn
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setC(100.0)

    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 1000, 1e-6))

    train_labels_list = []
    train_images_list = []
    for i in range(len(train_images_full)):
        one_park_image = cv2.imread(train_images_full[i], 0)
        res_image = cv2.resize(one_park_image, (IMG_SIZE, IMG_SIZE))
        hog_feature = hog.compute(res_image)
        train_images_list.append(hog_feature)
        train_labels_list.append(1)

    for i in range(len(train_images_free)):
        one_park_image = cv2.imread(train_images_free[i], 0)
        res_image = cv2.resize(one_park_image, (IMG_SIZE, IMG_SIZE))
        hog_feature = hog.compute(res_image)
        train_images_list.append(hog_feature)
        train_labels_list.append(0)

    print("train all: %d" % len(train_images_list))

    svm.train(np.array(train_images_list), cv2.ml.ROW_SAMPLE, np.array(train_labels_list))
    print("HOG training done")
    svm.save("my_det.xml")

    test_images = [img for img in glob.glob("test_images/*.jpg")]
    test_images.sort()
    result_list = []

    pkm_file = open('parking_map_python.txt', 'r')
    pkm_lines = pkm_file.readlines()
    pkm_coordinates = []

    for line in pkm_lines:
        st_line = line.strip()
        sp_line = list(st_line.split(" "))
        pkm_coordinates.append(sp_line)

    for img in test_images:
        one_park_image = cv2.imread(img)
        for one_c in pkm_coordinates:
            pts = [((float(one_c[0])), float(one_c[1])),
                   ((float(one_c[2])), float(one_c[3])),
                   ((float(one_c[4])), float(one_c[5])),
                   ((float(one_c[6])), float(one_c[7]))]
            # print(pts)
            # https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
            warped_image = four_point_transform(one_park_image, np.array(pts))
            res_image = cv2.resize(warped_image, (IMG_SIZE, IMG_SIZE))
            gray_image = cv2.cvtColor(res_image, cv2.COLOR_BGR2GRAY)

            hog_feature = hog.compute(gray_image)
            predict_label = svm.predict(np.array(hog_feature).reshape(1, -1))
            print(predict_label)

            # print(cv2.countNonZero(edge_image))
            spot_class = 1 if predict_label[1] == 1 else 0
            spot_color = (0, 255, 0) if spot_class == 1 else (255, 0, 0)

            int_points = [(int(a), int(b)) for a, b in pts]
            cv2.line(one_park_image, int_points[0], int_points[1], spot_color, 5)
            cv2.line(one_park_image, int_points[1], int_points[2], spot_color, 5)
            cv2.line(one_park_image, int_points[2], int_points[3], spot_color, 5)
            cv2.line(one_park_image, int_points[3], int_points[0], spot_color, 5)
            cv2.line(one_park_image, int_points[0], int_points[2], spot_color, 5)
            cv2.line(one_park_image, int_points[1], int_points[3], spot_color, 5)

            cv2.imshow('one_park_image', one_park_image)

            key = cv2.waitKey(0)
            if key == 27:  # exit on ESC
                break
            #if predict_label[1] == 1:




