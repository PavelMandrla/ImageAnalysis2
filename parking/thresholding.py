import cv2
import numpy as np
import glob
from utils import four_point_transform, load_parking_map, load_truth, draw_spot
from stats import Results


results = Results()
img_i = 0
pkm_coordinates = load_parking_map()
truth = load_truth()

cv2.namedWindow("detection", 0)
test_images = [img for img in glob.glob("test_images/*.jpg")]
test_images.sort()
for img in test_images:
    one_park_image = cv2.imread(img)
    one_park_image_show = one_park_image.copy()
    for one_c in pkm_coordinates:
        pts = [((float(one_c[0])), float(one_c[1])),
               ((float(one_c[2])), float(one_c[3])),
               ((float(one_c[4])), float(one_c[5])),
               ((float(one_c[6])), float(one_c[7]))]

        warped_image = four_point_transform(one_park_image, np.array(pts))
        res_image = cv2.resize(warped_image, (80, 80))

        blur_image = cv2.GaussianBlur(res_image, (3, 3), 0)
        gray_image = cv2.cvtColor(blur_image, cv2.COLOR_BGR2GRAY)
        edge_image = cv2.Canny(gray_image, 40, 120)

        spot_class = 1 if cv2.countNonZero(edge_image) > 350 else 0
        draw_spot(one_park_image_show, pts, spot_class)

        cv2.imshow('one_park_image', one_park_image_show)

        results.eval(truth[img_i], spot_class)
        img_i += 1

    key = cv2.waitKey(0)
    if key == 27:  # exit on ESC
        break

print("accuracy: %f" % results.get_accuracy())
print("F1 score: %f" % results.get_f1())
print("MCC: %f" % results.get_mcc())



