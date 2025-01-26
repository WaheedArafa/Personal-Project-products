import handTrackingModule as htm
import cv2
import time
import numpy as np
import math
import os

wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0

detector = htm.handDetector(detectionCon=0.7)

minVol = 0
maxVol = 100

vol = 0
volBar = 400
volPer = 0

control = False

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    key = cv2.waitKey(1)
    img = detector.findHands(img, False)
    lmList = detector.findPosition(img, draw=False)
    if control:
        if len(lmList) != 0:
            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 150), 3)
            cv2.circle(img, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (cx, cy), 7, (0, 255, 255), cv2.FILLED)

            length = math.hypot(x2 - x1, y2 - y1)

            x_wrist, y_wrist = lmList[0][1], lmList[0][2]
            x_mid, y_mid = lmList[12][1], lmList[12][2]
            hand_size = math.hypot(x_mid - x_wrist, y_mid - y_wrist)
            length_normalized = length / hand_size

            vol = np.interp(length_normalized, [0.2, 0.8], [minVol, maxVol])
            volBar = np.interp(length_normalized, [0.2, 0.8], [400, 150])
            volPer = np.interp(length_normalized, [0.2, 0.8], [0, 100])

            os.system(f"osascript -e 'set volume output volume {int(vol)}'")

            if volPer == 0:
                cv2.circle(img, (cx, cy), 9, (0, 0, 255), cv2.FILLED)
            key = cv2.waitKey(1)
            if key == 113:
                control = False
    
    if control == False:
        if key == 32:
            control = True
            
    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 2)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
    cv2.putText(img, f'{int(volPer)}%', (40, 450), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    cv2.imshow('Img', img)
    if key == 27:
        break
