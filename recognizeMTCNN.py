import cv2
from mtcnn import MTCNN


class MTCNNRecognizer():
    def __init__(self, img_path: str):
        self.img_path = img_path
        self.detector = MTCNN()
        self.frame = cv2.imread(self.img_path) 
        
        if self.frame is not None and self.frame.size > 0:
            self.rgb_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            self.results = self.detector.detect_faces(self.rgb_frame)
        else:
            self.rgb_frame = None
            self.results = []

    def image_verification(self):
        if self.frame is None or self.frame.size == 0:
            print(f'Error: the image - {self.img_path} failed verification')
            return False
        else:
             return True
        
              
    def detect_faces(self):
        return self.results
        
    def inf_about_faces(self):
            for i,result in enumerate(self.results):
                print(f"Face number: {i+1} detected:",
                    "x:", result['box'][0], 
                    "y:", result['box'][1], 
                    "w:", result['box'][2], 
                    "h:", result['box'][3]
                )
            count_faces = len(self.results)
            print(f"{count_faces} faces were found")
            
            return count_faces

    def show_detection_result(self):
        for result in self.results:
            x, y, w, h = result['box']
            cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Face Detection Result', self.frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()