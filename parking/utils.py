import numpy as np
import cv2
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