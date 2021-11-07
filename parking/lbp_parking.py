import cv2
import numpy as np
import glob
from utils import four_point_transform, load_parking_map, load_truth, draw_spot, load_train_images
from stats import Results

IMG_SIZE = 96
results = Results()
img_i = 0

pkm_coordinates = load_parking_map()
truth = load_truth()
train_images, train_labels = load_train_images(IMG_SIZE, IMG_SIZE)

LBP_recognizer = cv2.face.LBPHFaceRecognizer_create()
LBP_recognizer.train(train_images, np.array(train_labels))

print("TRAINING DONE")

test_images = [img for img in glob.glob("test_images/*.jpg")]
test_images.sort()
result_list = []

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

        predict_label, predict_confidence_lbp = LBP_recognizer.predict(gray_image)
        spot_class = 1 if predict_label == 1 else 0
        draw_spot(one_park_image_show, pts, spot_class)

        results.eval(truth[img_i], spot_class)
        img_i += 1

    key = cv2.waitKey(0)
    if key == 27:  # exit on ESC
        break

print("accuracy: %f" % results.get_accuracy())
print("F1 score: %f" % results.get_f1())
print("MCC: %f" % results.get_mcc())
