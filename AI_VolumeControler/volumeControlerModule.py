import time
import cv2
import mediapipe as mp
import math
import HandTrackingModule as htm
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

##########################
hFrame,  wFrame = 480, 640
##########################

# Capturing Real-time Video
cap = cv2.VideoCapture(0)
cap.set(3, wFrame)
cap.set(4, hFrame)

# capturing Hand
hand = htm.handDetector(detectionCon=0.7)

# Details of Volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 400
volPer = 0
pTime = 0


# Loop that work
while True:
    sucess, img = cap.read()
    img = hand.findHands(img)
    lmList = hand.findPosition(img) # index, x, y
    
    if len(lmList) != 0:
        x1, y1 = lmList[4][1], lmList[4][2] 
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1+x2)//2, (y1+y2)//2
        volLen = math.hypot(x1-x2, y1-y2)
        
        #  Converting one range to other : 25 - 200
        vol = np.interp(volLen, [25, 200], [minVol, maxVol])
        volBar = np.interp(volLen, [25, 200], [400, 150])
        volPer = np.interp(volLen, [25, 200], [0, 100])
        volume.SetMasterVolumeLevel(vol, None)
        
        cv2.circle(img, (x1, y1), 5, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 5, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        
        cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 1)
        cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
        cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX,
                1, (255, 0, 0), 2)
        
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX,
                1, (255, 0, 0), 3)

    cv2.imshow("AI_VolumeControler", img)
    cv2.waitKey(1)
