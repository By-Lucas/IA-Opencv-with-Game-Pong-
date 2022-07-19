# import libs
import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import mediapipe as mp

# Camera
cap = cv2.VideoCapture(0)


#Resolução
cap.set(3, 1280)
cap.set(4, 720)

# importa todas as imagens
imgBackground = cv2.imread("Resources/Background.png")
imageGameOver= cv2.imread("Resources/gameOver.png")
imageBall = cv2.imread("Resources/Ball.png", cv2.IMREAD_UNCHANGED)
imageBat1 = cv2.imread("Resources/bat1.png", cv2.IMREAD_UNCHANGED)
imageBat2 = cv2.imread("Resources/bat2.png", cv2.IMREAD_UNCHANGED)

# Detector de mãos
detector = HandDetector(detectionCon=0.8, maxHands=2)


while True:
    _, img = cap.read()
    img = cv2.flip(img, 1)

    # Find the hand and this landmarks
    hands, img = detector.findHands(img) #with draw

    # Deverlaying the backgound image
    img = cv2.addWeighted(img, 0.2, imgBackground, 0.8, 0)

    cv2.imshow("Image", img)
    cv2.waitKey(1)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break