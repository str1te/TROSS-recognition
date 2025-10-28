import cv2

class CameraManager:
    
    @staticmethod
    def find_available_cameras(max_check=4):
        available_cameras = []
        for camera_index in range(max_check):
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened():
                available_cameras.append(camera_index)
            cap.release()
        return available_cameras
    
    def __init__(self, camera_index=0, config=None):
        self.config = config or DetectorConfig()
        self.cap = None
        self.setup_camera(camera_index)
    
    def setup_camera(self, camera_index):
        available_cameras = self.find_available_cameras()
        if not available_cameras:
            raise Exception("No cameras available")
        
        if camera_index not in available_cameras:
            camera_index = available_cameras[0]
            print(f"Selected camera not available, using camera {camera_index}")
        
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, self.config.FPS)
        
        if not self.cap.isOpened():
            raise Exception(f"Failed to open camera {camera_index}")
    
    def get_frame(self) -> any:
        if not self.cap:
            return False, None
        
        success, frame = self.cap.read()
        if success:
            frame = cv2.flip(frame, 1)  # Зеркальное отражение
        return success, frame
    
    def release(self):
        if self.cap:
            self.cap.release()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()