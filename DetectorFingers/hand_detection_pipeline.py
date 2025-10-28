import cv2
from camera_manager import CameraManager
from hand_detector import HandDetector
from finger_counter import FingerCounter
from result_visualizer import ResultVisualizer
from config import DetectorConfig

class HandDetectionPipeline:
    
    def __init__(self, camera_index=0, config=None):
        self.config = config or DetectorConfig()
        self.camera_manager = None
        self.hand_detector = None
        self.finger_counter = FingerCounter()
        self.visualizer = None
        self.is_running = False
        
        self.initialize_components(camera_index)
    
    def initialize_components(self, camera_index):
        try:
            self.camera_manager = CameraManager(camera_index, self.config)
            self.hand_detector = HandDetector(self.config)
            self.visualizer = ResultVisualizer()
            print("Hand detection pipeline initialized successfully!")
        except Exception as err:
            self.cleanup()
            raise Exception(f"Failed to initialize pipeline: {err}")
    
    def process_frame(self, frame):
        hand_landmarks = self.hand_detector.process_frame(frame)
        
        self.hand_detector.draw_landmarks(frame, hand_landmarks)
        
        #Подсчет пальцев для каждой руки
        for hand_idx, landmarks in enumerate(hand_landmarks):
            finger_count, fingers_status = self.finger_counter.count_fingers(landmarks)
            self.visualizer.draw_hand_info(
                frame, hand_idx, finger_count, fingers_status
            )
        
        self.visualizer.show_frame(frame)
        return True
    
    def run(self):
        self.is_running = True
        print("Starting hand detection. Press 'q' to quit, 'r' to recalibrate.")
        
        try:
            while self.is_running:
                success, frame = self.camera_manager.get_frame()
                if not success:
                    print("Failed to capture frame")
                    continue
                
                if not self.process_frame(frame):
                    break
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Exit by key Q")
                    break
                    
        except KeyboardInterrupt:
            print("Interrupted by user")
        except Exception as err:
            print(f"Error in detection loop: {err}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        self.is_running = False
        if self.camera_manager:
            self.camera_manager.release()
        if self.hand_detector:
            self.hand_detector.close()
        if self.visualizer:
            self.visualizer.close()
        print("Resources cleaned up")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()