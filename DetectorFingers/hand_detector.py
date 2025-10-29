import cv2
import mediapipe as mp
from config import DetectorConfig


class HandDetector:
    def __init__(self, config=None):
        self.config = config or DetectorConfig()
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.config.MAX_HANDS,
            min_detection_confidence=self.config.HAND_CONFIDENCE,
            min_tracking_confidence=0.5,
        )

    def process_frame(self, frame):
        if frame is None:
            return []

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        return results.multi_hand_landmarks

    def draw_landmarks(self, frame, hand_landmarks):
        for landmarks in hand_landmarks:
            self.mp_draw.draw_landmarks(
                frame, landmarks, self.mp_hands.HAND_CONNECTIONS
            )
