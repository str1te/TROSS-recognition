import cv2
import numpy as np
import mediapipe as mp


class DetectingFingers():
    def __init__(self, index: int = None):
        print("Initializing the camera...")
        
        def find_camera():
                camera_test = False   
                camera_index = 0  
                while True:
                    camera_found = False 
                    for backend in [cv2.CAP_ANY, cv2.CAP_DSHOW]:
                        cap = cv2.VideoCapture(camera_index, backend)
                        if cap.isOpened():
                            camera_found = True
                            camera_test = True
                        else:
                            cap.release()
                            break
                    if camera_found:
                        camera_index += 1
                        
                    if not camera_found:
                            break
                
                    if not camera_test:
                        return 

                    return list(range(camera_index))
        
        available_cameras = find_camera()
    
        if index is not None:
            self.cap = cv2.VideoCapture(index)
            self.camera_test = True
            if not self.cap.isOpened():
                print(f"Error:Camera with index {index} dont working")
                self.camera_test = False
                while True:    
                    answer = str(input('Do you want to identify the available cameras? Y/N\n'))
                    if answer in ('y', 'Y'):
                        answer = 1
                        break
                    else:
                        if answer in ('n', 'N'):
                            self.camera_test = False
                            answer = 0
                            return
                        else:
                            print('Error:Enter Y or N')

                if answer:                                              
                    if available_cameras is None:
                        print("Couldn't find a working camera")
                        return
                    else:
                        while True:    
                            try:
                                print(f"list of working camera indexes: {','.join(map(str,available_cameras))}")
                                index = int(input("Plese, select a camera from the list of found indexes\n"))
                                if index in available_cameras:
                                    self.cap = cv2.VideoCapture(index)
                                    self.camera_test = True
                                    break
                                else:
                                    print("Error: The selected index is not in the list. Please try again.")
                            except ValueError:
                                print("Error: Please enter a valid integer.")
                else:
                    return

        else:
            if available_cameras is None:
                print("Couldn't find a working camera")
                return
            else:
                while True:    
                    try:
                        print(f"list of working camera indexes: {','.join(map(str,available_cameras))}")
                        index = int(input("Plese, select a camera from the list of found indexes\n"))
                        if index in available_cameras:
                            self.cap = cv2.VideoCapture(index)
                            self.camera_test = True
                            break
                        else:
                            print("Error: The selected index is not in the list. Please try again.")
                    except ValueError:
                        print("Error: Please enter a valid integer.")
 
        self.mp_hand = mp.solutions.hands
        self.hands = self.mp_hand.Hands()
        self.mp_draw = mp.solutions.drawing_utils
    
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        print("Initialization is completed!")
    
    def count_fingers(self, hand_landmarks) -> any:
        finger_tips = [4, 8, 12, 16, 20]
        finger_pips = [3, 6, 10, 14, 18]

        fingers_up = []

        thumb_tip = hand_landmarks.landmark[4]
        base_littlefinger = hand_landmarks.landmark[17]
        tip_middlefinger = hand_landmarks.landmark[2]
        base_hand = hand_landmarks.landmark[0]

        if tip_middlefinger.y > base_hand.y:
            if thumb_tip.y >= base_littlefinger.y:
                fingers_up.append(1)
            else:
                fingers_up.append(0)

            for i in range(1, 5):
                tip = hand_landmarks.landmark[finger_tips[i]]
                pip = hand_landmarks.landmark[finger_pips[i]]

                if tip.y > pip.y:
                    fingers_up.append(1)
                else:
                    fingers_up.append(0)
        else:
            if thumb_tip.y <= base_littlefinger.y:
                fingers_up.append(1)
            else:
                fingers_up.append(0)

            for i in range(1, 5):
                tip = hand_landmarks.landmark[finger_tips[i]]
                pip = hand_landmarks.landmark[finger_pips[i]]

                if tip.y < pip.y:
                    fingers_up.append(1)
                else:
                    fingers_up.append(0)

        return sum(fingers_up), fingers_up 

    
    def detecting(self):
        if not self.camera_test:
            return
        
        while True:
            accept_img, frame = self.cap.read()
            if not accept_img:
                print("Error: couldn't get a frame")
                break
            
            mirrored_frame = cv2.flip(frame, 1)
            cvt_frame = cv2.cvtColor(mirrored_frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(cvt_frame)
            
            if results.multi_hand_landmarks:
                for hand_idx, handLms in enumerate(results.multi_hand_landmarks):
                    finger_count, fingers_status = self.count_fingers(handLms)
                    self.mp_draw.draw_landmarks(mirrored_frame, handLms, self.mp_hand.HAND_CONNECTIONS)
                    info_y = 50 + hand_idx * 100
                    
                    cv2.putText(
                                mirrored_frame, f'Hand {hand_idx + 1}: {finger_count} fingers', 
                                (10, info_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                1, (255,255,255), 2
                    )
            
            cv2.imshow('Camera', mirrored_frame)
            key = cv2.waitKey(1) & 0xFF  
            
            if key == ord('q'):
                print("Exit by key Q") 
                break
        
        self.cap.release()
        cv2.destroyAllWindows()   