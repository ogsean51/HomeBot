'''from PyQt5.QtWidgets import *
from PyQt5 import QtCore
from PyQt5 import QtGui'''
import sys
import mediapipe as mp
import cv2
import time
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import tensorflow as tf
import pandas as pd


cap = cv2.imread("Dataset/1.png")

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.75,
                      min_tracking_confidence=0.75)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0


img = cv2.imread("Dataset/50.png")
imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
results = hands.process(imgRGB)
data = ["WRIST", "THUMB", "THUMB_1", "THUMB_2", "THUMB_3", "THUMB_TIP", "INDEX", "INDEX_1", "INDEX_2", "INDEX_TIP", "MIDDLE", "MIDDLE_1", "MIDDLE_2", "MIDDLE_TIP", "RING", "RING_1", "RING_2", "RING_TIP", "PINKY", "PINKY_1", "PINKY_2", "PINKY_TIP"]


#print(results.multi_hand_landmarks)
#while True:
if results.multi_hand_landmarks:
    for handLms in results.multi_hand_landmarks:
        for id, lm in enumerate(handLms.landmark):
            print(id,lm)
            h, w, c = img.shape
            cx, cy = int(lm.x *w), int(lm.y*h)
            plt.scatter(cx, cy)
            #if id ==0:

            cv2.circle(img, (cx,cy), 3, (255,0,255), cv2.FILLED)

        mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)


cTime = time.time()
fps = 1/(cTime-pTime)
pTime = cTime

cv2.putText(img,str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

cv2.imshow("Image", img)
#if cv2.waitKey(33) == ord('a'):
    #cap.release()
    #cv2.destroyAllWindows()
plt.show()


