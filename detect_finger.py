import cv2
import numpy as np
import mediapipe as mp


class detect_fingers():
    def __init__(self):
        print("Initializing the camera...")
        camera_found = False
        for camera_index in [0, 1, 2]:
            for backend in [cv2.CAP_ANY, cv2.CAP_DSHOW]:
                print(f"Trying the camera {camera_index} with a backend {backend}")
                self.cap = cv2.VideoCapture(camera_index, backend)
                if self.cap.isOpened():
                    ret, test_frame = self.cap.read()
                    if ret and test_frame is not None:
                        print(f"Successful: Camera {camera_index}, backend {backend}")
                        camera_found = True
                        break
                    else:
                        self.cap.release()
                else:
                    self.cap.release()
            if camera_found:
                break
        
        if not camera_found:
            print("Couldn't find a working camera")
            self.cap = None
            return       
    
        self.mpHand = mp.solutions.hands
        self.hands = self.mpHand.Hands()
        self.mpDraw = mp.solutions.drawing_utils
    
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        print("Initialization is completed")
    
    
    def count_fingers(self, hand_landmarks):
        finger_tips = [4, 8, 12, 16, 20]
        finger_pips = [3, 6, 10, 14, 18]

        fingers_up = []

        thumb_tip = hand_landmarks.landmark[4]
        thumb_mcp = hand_landmarks.landmark[2]

        if hand_landmarks.landmark[12].y > hand_landmarks.landmark[0].y:
            if thumb_tip.y >= hand_landmarks.landmark[17].y:
                fingers_up.append(1)
            else:
                fingers_up.append(0)

            for i in range(1,5):
                tip = hand_landmarks.landmark[finger_tips[i]]
                pip = hand_landmarks.landmark[finger_pips[i]]

                if tip.y > pip.y:
                    fingers_up.append(1)
                else:
                    fingers_up.append(0)
        else:
            if thumb_tip.y <= hand_landmarks.landmark[17].y:
                fingers_up.append(1)
            else:
                fingers_up.append(0)

            for i in range(1,5):
                tip = hand_landmarks.landmark[finger_tips[i]]
                pip = hand_landmarks.landmark[finger_pips[i]]

                if tip.y < pip.y:
                    fingers_up.append(1)
                else:
                    fingers_up.append(0)

        return sum(fingers_up), fingers_up 

    
    def detecting(self):
        while True:
            r, frame = self.cap.read()
            if not r:
                print("Error: couldn't get a frame")
                break
            
            frame = cv2.flip(frame,1)
            cvt_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(cvt_frame)
            
            if results.multi_hand_landmarks:
                for hand_idx, handLms in enumerate(results.multi_hand_landmarks):
                    finger_count, fingers_status = self.count_fingers(handLms)
                    self.mpDraw.draw_landmarks(frame, handLms, self.mpHand.HAND_CONNECTIONS)
                    info_y = 50 + hand_idx * 100
                    
                    cv2.putText(
                                frame, f'Hand {hand_idx + 1}: {finger_count} fingers', 
                                (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2
                                )
            
            cv2.imshow('Camera',frame)
            key = cv2.waitKey(1) & 0xFF  
            
            if key == ord('q'):
                print("Exit by key Q") 
                break
        self.cap.release()
        cv2.destroyAllWindows()   
