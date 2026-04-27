import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class handDetector():
    def __init__(self, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.results = None
        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=maxHands,
            min_hand_detection_confidence=float(detectionCon),
            min_tracking_confidence=float(trackCon)
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
        self.results = self.detector.detect(mp_image)

        if draw and self.results.hand_landmarks:
            for hand_landmarks in self.results.hand_landmarks:
                h, w, _ = img.shape
                points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
                connections = [
                    (0,1),(1,2),(2,3),(3,4),
                    (0,5),(5,6),(6,7),(7,8),
                    (0,9),(9,10),(10,11),(11,12),
                    (0,13),(13,14),(14,15),(15,16),
                    (0,17),(17,18),(18,19),(19,20),
                    (5,9),(9,13),(13,17)
                ]
                for a, b in connections:
                    cv2.line(img, points[a], points[b], (0, 255, 0), 2)
                for pt in points:
                    cv2.circle(img, pt, 5, (255, 0, 0), cv2.FILLED)
        return img

    def findPosition(self, img, handNo=0, draw=False):
        lmList = []
        if self.results and self.results.hand_landmarks:
            if handNo < len(self.results.hand_landmarks):
                h, w, _ = img.shape
                for id, lm in enumerate(self.results.hand_landmarks[handNo]):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 10, (255, 255, 255), cv2.FILLED)
        return lmList

    def fingersUp(self, handNo=0):
        fingers = []
        if self.results and self.results.hand_landmarks:
            if handNo < len(self.results.hand_landmarks):
                hand = self.results.hand_landmarks[handNo]

                # Tip and pip landmark IDs for each finger
                # Thumb: 4, 3 | Index: 8,6 | Middle: 12,10 | Ring: 16,14 | Pinky: 20,18
                tipIds = [4, 8, 12, 16, 20]

                # Thumb — compare x instead of y since it's horizontal
                if hand[tipIds[0]].x < hand[tipIds[0] - 1].x:
                    fingers.append(1)
                else:
                    fingers.append(0)

                # Four fingers — compare y (lower y = higher on screen = finger up)
                for id in range(1, 5):
                    if hand[tipIds[id]].y < hand[tipIds[id] - 2].y:
                        fingers.append(1)
                    else:
                        fingers.append(0)

        return fingers
