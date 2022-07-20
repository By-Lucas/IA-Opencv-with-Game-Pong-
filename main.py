# import libs
# Instalar todas
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

# Variaveis
ballPos = [100, 100] # Posição da bola
speedX = 20 # velocidade da bola
speedY = 20 # velocidade da bola
gameOver = False
score = [0, 0]


while True:
    _, img = cap.read()
    img = cv2.flip(img, 1)
    imgRaw = img.copy()

    # Encontre a mão e esses marcos
    hands, img = detector.findHands(img, flipType=False) #retirar o

    # Deverlaying a imagem backgound
    img = cv2.addWeighted(img, 0.2, imgBackground, 0.8, 0)

    # Verifique se há mãos
    if hands:
        for hand in hands:
            x, y, w, h = hand['bbox']
            h1, w1, _ = imageBat1.shape
            y1 = y - h1//2
            y1 = np.clip(y1, 20, 415)
            # Se estiver uma mão na esquerda, vai aparecer a barra lateral esquerda
            if hand['type'] == "Left":
                img = cvzone.overlayPNG(img, imageBat1,(50, y1))
                if 59 < ballPos[0] < 59 + w1 and y1 < ballPos[1] < y1 + h1:
                    speedX = - speedX
                    ballPos[0] += 30
                    score[0] += 1

            # Se estiver uma mão na direita, vai aparecer a barra lateral direita
            if hand['type'] == "Right":
                img = cvzone.overlayPNG(img, imageBat2,(1195, y1))
                if 1195 - 55 < ballPos[0] < 1195 - 30 + w1 and y1 < ballPos[1] < y1 + h1:
                    speedX = - speedX
                    ballPos[0] -= 30
                    score[1] += 1

    #Game Over
    if ballPos[0] < 40 or ballPos[0] > 1200:
        gameOver = True
    
    if gameOver:
        img = imageGameOver
        # Resultado geral
        cv2.putText(img, str(score[1] + score[0]).zfill(2), (585, 360), cv2.FONT_HERSHEY_COMPLEX, 
                2.5,(200, 0, 20), 5)

    # Se nao fo Game Over, mova a bola
    else:
        # Mover posição da bola e limitar topo e fundo para ela
        if ballPos[1] >= 500 or ballPos[1] <= 10:
            speedY = -speedY

        ballPos[0] += speedX
        ballPos[1] += speedY

        # Desenho da bola
        img = cvzone.overlayPNG(img, imageBall, ballPos)

        # Mostrar Score na tela
        cv2.putText(img, str(score[0]), (300,650), cv2.FONT_HERSHEY_COMPLEX, 3,(255,255,255),5)
        cv2.putText(img, str(score[1]), (900,650), cv2.FONT_HERSHEY_COMPLEX, 3,(255,255,255),5)

    # Exibir webcam no canto inferior esquerdo
    img[580:700, 20:233] = cv2.resize(imgRaw, (213, 120))

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

    # Parar jogo
    if key == ord('q'):
        break

    # Reiniciar jogo
    if key == ord('r'):
        ballPos = [100, 100] # Posição da bola
        speedX = 20 # velocidade da bola
        speedY = 20 # velocidade da bola
        gameOver = False
        score = [0, 0]
        imageGameOver= cv2.imread("Resources/gameOver.png")
