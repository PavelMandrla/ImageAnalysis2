import cv2
import cv2 as cv
import time

cap = cv.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

face_cascade = cv.CascadeClassifier('./haarcascade_frontalface_default.xml')

counter = 0
nFrame = 10
FPS = 0.0
start_time = time.time()

while(True):
    ret, opencv_frame = cap.read()
    faces_rects = face_cascade.detectMultiScale(opencv_frame, 1.1, 5)

    for rect in faces_rects:
        cv.rectangle(opencv_frame, rect, (0,0,255), 2)

    if counter == nFrame:
        end_time = time.time()
        allTime = float(end_time - start_time)
        FPS = float(counter)/allTime
        counter = 0
        start_time = time.time()
    counter += 1

    cv.putText(opencv_frame, 'FPS: %f' % FPS, (20, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
    cv.putText(opencv_frame, 'Detections: %d' % len(faces_rects), (20, 75), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

    cv.imshow("frame", opencv_frame)

    cv.waitKey(2)