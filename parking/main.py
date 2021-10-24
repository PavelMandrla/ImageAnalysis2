#!/usr/bin/python

import sys
import cv2
import numpy as np
import math
import struct
from datetime import datetime
import glob
from data_loader import four_point_transform


def main(argv):
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

            cv2.imshow('blur_image', blur_image)
            cv2.imshow('res_image', res_image)
            cv2.imshow('edge_image', edge_image)
            cv2.waitKey(200)
            # roi = img[y:y+h, x:x+w]
        cv2.imshow('one_park_image', one_park_image)
        key = cv2.waitKey(0)
        if key == 27:  # exit on ESC
            break


if __name__ == "__main__":
    main(sys.argv[1:])
