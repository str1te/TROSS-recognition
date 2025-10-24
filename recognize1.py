#Require TensorFlow
import cv2
from mtcnn import MTCNN


class MTCNNRecognizer():
    def __init__(self, img_path: str):
        self.img_path = img_path
        self.detector = MTCNN()
        self.frame = cv2.imread(self.img_path)
        self.rgb_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        self.results = self.detector.detect_faces(self.rgb_frame)

    def detect_faces(self):
        if self.frame is None:
                print(f"Ошибка: не удалось загрузить изображение {self.img_path}")
                return None
        
        return self.results
        
    def detection_callback(self):
            count = 1
            for result in self.results:

                x, y, w, h = result['box']
                
                print(f"Face numbre {count} detected:",
                    "x:", result['box'][0], 
                    "y:", result['box'][1], 
                    "w:", result['box'][2], 
                    "h:", result['box'][3]
                )
                count += 1

    def show_detection_result(self):
        if self.frame is None or results is None:
            print("Не найденно изображение")
            pass

        for result in self.results:
            x, y, w, h = result['box']
            cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Face Detection Result', self.frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

picture = MTCNNRecognizer("chinetown.png")
results = picture.detect_faces()
MTCNNRecognizer("chinetown.png").detection_callback()