import cv2 as cv
import mediapipe as mp


class HandTracker:

    def __init__(self, draw=True,
                 static_image_mode=False,
                 max_num_hands=2,
                 model_complexity=1,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):

        self.mphand = mp.solutions.hands
        self.mpdraw = mp.solutions.drawing_utils

        self.hand = self.mphand.Hands(static_image_mode, max_num_hands,
                                      model_complexity, min_detection_confidence, min_tracking_confidence)

    # Find Hand get shape  and number of detected hans  ##########################################
    def findHand(self, img):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.result = self.hand.process(imgRGB)
        self.shape = img.shape
        self.numOfHands = 0

        if self.result.multi_hand_landmarks:
            tlmk = self.result.multi_hand_landmarks
            self.numOfHands = len(tlmk)
            self.handInfo = self.result.multi_handedness

    # find list of all HandMarks
    def findlmk(self):
        ''' Before calling this call "findHand()" '''

        list = []
        hei, wid, cha = self.shape
        if self.result.multi_hand_landmarks:
            for handNo in range(0, self.numOfHands):
                l1 = []
                tlmk = self.result.multi_hand_landmarks[handNo].landmark
                for lmk in tlmk:
                    l1.append([int(lmk.x * wid), int(lmk.y * hei)])  # [x, y]
                list.append(l1)

        return list

    #  find Bounding Box ################################################################
    def findBoundingBox(self, list):
        bbox = []
        if len(list) > 0:
            for handNo in range(0, self.numOfHands):
                bbox.append(self.__findBoundingBox1(list[handNo]))

            return bbox

    def __findBoundingBox1(self, list):
        minHei, minWid, maxHei, maxWid = 1e9, 1e9, -1e9, -1e9
        if len(list) > 0:
            for li in list:
                minHei = min(li[1], minHei)
                maxHei = max(li[1], maxHei)
                minWid = min(li[0], minWid)
                maxWid = max(li[0], maxWid)

        return [minHei, maxHei, minWid, maxWid]

    #  landmark TIP ################################################################
    def landmarkTip(self, list):
        tip = []
        if len(list) > 0:
            for handNo in range(0, self.numOfHands):
                tip.append(self.__landmarkTip1(list[handNo]))

        return tip

    def __landmarkTip1(self, list):
        tip = []

        if len(list) > 0:
            for indx in range(4, 21, 4):
                tip.append(list[indx])
        return tip

    #  landmark DIP ################################################################
    def landmarkDip(self, list):
        dip = []
        if len(list) > 0:
            for handNo in range(0, self.numOfHands):
                dip.append(self.__landmarkDip1(list[handNo]))

        return dip

    def __landmarkDip1(self, list):
        dip = []

        if len(list) > 0:
            for indx in range(3, 21, 4):
                if indx != 0:
                    dip.append(list[indx])
        return dip

    #  landmark PIP ################################################################
    def landmarkPip(self, list):
        pip = []
        if len(list) > 0:
            for handNo in range(0, self.numOfHands):
                pip.append(self.__landmarkPip1(list[handNo]))

        return pip

    def __landmarkPip1(self, list):
        pip = []

        if len(list) > 0:
            for indx in range(2, 21, 4):
                if indx != 0:
                    pip.append(list[indx])
        return pip

    #  landmark MCP ################################################################
    def landmarkMcp(self, list):
        mcp = []
        if len(list) > 0:
            for handNo in range(0, self.numOfHands):
                mcp.append(self.__landmarkMcp1(list[handNo]))

        return mcp

    def __landmarkMcp1(self, list):
        mcp = []

        if len(list) > 0:
            for indx in range(1, 21, 4):
                if indx != 0:
                    mcp.append(list[indx])
        return mcp

    #  landmark MCP ################################################################
    def landmarkWrist(self, list):
        wrist = []
        if len(list) > 0:
            for handNo in range(0, self.numOfHands):
                wrist.append(self.__landmarkWrist1(list[handNo]))

        return wrist

    def __landmarkWrist1(self, list):
        return list[0]

    #  draw Skeleton ####################################################################
    def drawSkl(self, img, list, lineCol=(255, 255, 255), dotCol=(235, 206, 135)):

        tlmk = self.result.multi_hand_landmarks
        if tlmk:
            for hlm in tlmk:
                self.mpdraw.draw_landmarks(img, hlm, self.mphand.HAND_CONNECTIONS,
                                           self.mpdraw.DrawingSpec(
                                               color=dotCol),
                                           self.mpdraw.DrawingSpec(color=lineCol))

                TIP = self.landmarkTip(list)
                bbox = self.findBoundingBox(list)
                for handNo in range(0, self.numOfHands):
                    for point in TIP[handNo]:
                        cv.circle(img, point, 7, dotCol, cv.FILLED)
                    
                    xWrist, yWrist = list[handNo][0]
                    cv.circle(img, (xWrist, yWrist), 7, self.mpdraw.RED_COLOR, cv.FILLED)
                    yMin, yMax, xMin, xMax = bbox[handNo]
                    pad, pad1 = 20, 10
                    textHei = 25 
                    cv.rectangle(img, (xMin-pad, yMin-pad), (xMax+pad, yMax+pad), dotCol, 1) 
                    cv.rectangle(img, (xMin-pad, yMin-pad - textHei), (xMax+pad, yMin-pad), dotCol, cv.FILLED) 
                    cv.putText(img, str(handNo), (xWrist+pad1, yWrist+pad1), cv.FONT_HERSHEY_COMPLEX,  0.5, lineCol, 1)
                    
                    # Handedness
                    hand = self.handInfo[handNo].classification[0]
                    
                    # print(hand)
                    
                    text = f"{hand.label} | {hand.score*100:.2f}%"
                    cv.putText(img, text, (xMin, yMin-pad - textHei + pad), cv.FONT_HERSHEY_SIMPLEX,  0.7, lineCol, 1)
                    
                    
                # Creating Box
                
                    



# Main Function
def main():
    cap = cv.VideoCapture(0)
    hand = HandTracker()

    while True:
        success, img = cap.read()
        hand.findHand(img)
        list = hand.findlmk()
        bbox = hand.findBoundingBox(list)
        # tip = hand.landmarkTip(list)

        hand.drawSkl(img, list)


        cv.imshow("Hand Detector", img)

        if cv.waitKey(1) & 0xFF == 27:
            break


if __name__ == "__main__":
    main()
