import cv2
import cv2 as cv
import time

cap = cv.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv.HOGDescriptor.getDefaultPeopleDetector())

counter = 0
nFrame = 10
FPS = 0.0
start_time = time.time()

while(True):
    ret, opencv_frame = cap.read()

    rects, weights = hog.detectMultiScale(opencv_frame, winStride=(4, 4), scale=1.25, hitThreshold=1.0)
    print(weights)  #confidence detekce (jak moc si to je jisté)



    for rect in rects:
        cv.rectangle(opencv_frame, rect, (0,0,255), 2)


    if counter == nFrame:
        end_time = time.time()
        allTime = float(end_time - start_time)
        FPS = float(counter)/allTime
        counter = 0
        start_time = time.time()
    counter += 1

    cv.putText(opencv_frame, 'FPS: %f' % FPS, (20, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
    cv.putText(opencv_frame, 'Detections: %d' % len(rects), (20, 75), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

    cv.imshow("frame", opencv_frame)

    cv.waitKey(2)