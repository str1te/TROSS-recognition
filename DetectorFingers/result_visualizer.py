import cv2

class ResultVisualizer:
    
    def __init__(self, window_name='Hand Detection'):
        self.window_name = window_name
        cv2.namedWindow(self.window_name)
    
    def draw_hand_info(self, frame, hand_index, finger_count, fingers_status, position=(10, 50)):

        x, y = position
        y_offset = hand_index * 100
        
        cv2.putText(
            frame, f'Hand {hand_index + 1}: {finger_count} fingers', 
            (x, y + y_offset), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        
        #Детальная информация о пальцах
        finger_counter = FingerCounter()
        status_text = finger_counter.get_finger_status_text(fingers_status)
        cv2.putText(
            frame, status_text,
            (x, y + y_offset + 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
        )
    
    def show_frame(self, frame):
        cv2.imshow(self.window_name, frame)
    
    def close(self):
        cv2.destroyAllWindows()