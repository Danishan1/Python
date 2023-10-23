import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm


pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)
detector = htm.HandTracker()
skyBlue = (235, 206, 135)

while True:
    success, img = cap.read()
    detector.findHand(img)
    lmList = detector.findlmk()
    
    detector.drawSkl(img, lmList)
    

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps))+' FPS', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, skyBlue, 2)

    cv2.imshow("Image", img)
    
    if cv2.waitKey(1) & 0xFF == 27:
            break