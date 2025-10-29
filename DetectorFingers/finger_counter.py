class FingerCounter:
    FINGER_TIPS = [4, 8, 12, 16, 20]
    FINGER_PIPS = [3, 6, 10, 14, 18]
    FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

    def count_fingers(self, hand_landmarks):
        finger_tips = self.FINGER_TIPS
        finger_pips = self.FINGER_PIPS
        fingers_up = []

        thumb_tip = hand_landmarks.landmark[4]
        base_little_finger = hand_landmarks.landmark[17]
        tip_middle_finger = hand_landmarks.landmark[2]
        base_hand = hand_landmarks.landmark[0]

        is_palm_up = tip_middle_finger.y > base_hand.y

        if is_palm_up:
            thumb_up = thumb_tip.y >= base_little_finger.y
        else:
            thumb_up = thumb_tip.y <= base_little_finger.y
        fingers_up.append(1 if thumb_up else 0)

        for i in range(1, 5):
            tip = hand_landmarks.landmark[finger_tips[i]]
            pip = hand_landmarks.landmark[finger_pips[i]]

            if is_palm_up:
                finger_up = tip.y > pip.y
            else:
                finger_up = tip.y < pip.y

            fingers_up.append(1 if finger_up else 0)

        total_fingers = sum(fingers_up)
        return total_fingers, fingers_up

    def get_finger_status_text(self, fingers_status):
        status_text = []
        for i, status in enumerate(fingers_status):
            finger_name = self.FINGER_NAMES[i]
            state = "UP" if status else "DOWN"
            status_text.append(f"{finger_name}: {state}")
        return ", ".join(status_text)
