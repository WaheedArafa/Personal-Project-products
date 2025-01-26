import handTrackingModule as htm
import cv2
import numpy as np
import time
import os

folderPath = 'drawing bar'
myList = os.listdir(folderPath)
# print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
# print(len(overlayList))

drawColor = (0,0,0)
brushThickness = 15
eraserThickness = 100
xp, yp = 0, 0

imgCanvas = np.zeros((720, 1280, 3), np.uint8)
pTime = 0

header = overlayList[0]

detector = htm.handDetector(detectionCon=0.85)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

while True:
    seccess, img = cap.read()
    img = cv2.flip(img, 1)

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList)!=0:
        # print(lmList)

        x1,y1 = lmList[8][1:]
        x2,y2 = lmList[12][1:]

        fingers = detector.fingersUp()
        # print(fingers)

        if fingers[2] and fingers[1]:
            xp, yp = 0, 0
            # print('selection mode')
            cv2.circle(img, (x2,y2),15, drawColor, cv2.FILLED)
            if y2 < 140:
                if 200 < x2 < 350:
                    header = overlayList[3]
                    drawColor = (255,209,76)
                elif 450 < x2 < 650:
                    header = overlayList[1]
                    drawColor = (0,0,255)
                elif 750 < x2 < 900:
                    header = overlayList[2]
                    drawColor = (0,255,0)
                elif 950 < x2 < 1200:
                    header = overlayList[0]
                    drawColor = (0,0,0)
                cv2.circle(img, (x2,y2),15, drawColor, cv2.FILLED)


        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1,y1),15, drawColor, cv2.FILLED)
            # print('drawing mode')
            if xp==0 and yp==0:
                xp,yp = x1,y1

            if drawColor == (0,0,0):
                cv2.line(img, (xp,yp),(x1,y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp,yp),(x1,y1), drawColor, eraserThickness)
            cv2.line(img, (xp,yp),(x1,y1), drawColor, brushThickness)
            cv2.line(imgCanvas, (xp,yp),(x1,y1), drawColor, brushThickness)

            xp, yp = x1, y1

    imgGray = cv2.cvtColor(imgCanvas,cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray,50,255,cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img,imgInv)
    img = cv2.bitwise_or(img,imgCanvas)
    

    img[0:140, 0:1280] = header

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    
    # img = cv2.addWeighted(img,0.5,imgCanvas,0.5,0)
    cv2.imshow('Image', img)
    # cv2.imshow('Canvas', imgCanvas)
    key = cv2.waitKey(1)

    if key == 27:
        break
