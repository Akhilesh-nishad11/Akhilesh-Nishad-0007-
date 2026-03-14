import cv2
import mediapipe as mp
import pyautogui
import math
import time

# MediaPipe modules initialize
mp_hands = mp.solutions.hands.Hands(max_num_hands=2)
mp_face = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
mp_pose = mp.solutions.pose.Pose()
mp_draw = mp.solutions.drawing_utils

# Screen size
screen_w, screen_h = pyautogui.size()

# Nose tracking for activity
prev_nose_x = None

# Default mode
mode = "activity"

# Gesture actions dictionary
gesture_actions = {
    "volume_up": lambda: pyautogui.press("volumeup"),
    "volume_down": lambda: pyautogui.press("volumedown"),
    "open_file": lambda: pyautogui.doubleClick(),
    "close_file": lambda: pyautogui.hotkey("ctrl", "w"),
    "zoom_in": lambda: pyautogui.hotkey("ctrl", "+"),
    "zoom_out": lambda: pyautogui.hotkey("ctrl", "-"),
    "scroll_up": lambda: pyautogui.scroll(300),
    "scroll_down": lambda: pyautogui.scroll(-300),
}

# ------------------- Utility Functions -------------------

def eye_closed(landmarks, eye_indices):
    top = landmarks[eye_indices[1]].y
    bottom = landmarks[eye_indices[5]].y
    return abs(top - bottom) < 0.01

def detect_single_blink(landmarks, eye_indices, side="left"):
    if eye_closed(landmarks, eye_indices):
        if side == "left":
            return "close_file"
        elif side == "right":
            return "open_file"
    return None

def classify_activity(pose_landmarks, face_landmarks=None, prev_nose_x=None):
    if not pose_landmarks:
        return "Standing", prev_nose_x   # Default अब Standing है

    nose_y = pose_landmarks.landmark[0].y
    nose_x = pose_landmarks.landmark[0].x
    left_hand_y = pose_landmarks.landmark[15].y
    right_hand_y = pose_landmarks.landmark[16].y
    left_shoulder_y = pose_landmarks.landmark[11].y
    right_shoulder_y = pose_landmarks.landmark[12].y
    left_hip_y = pose_landmarks.landmark[23].y
    right_hip_y = pose_landmarks.landmark[24].y

    activity = "Standing"   # Default activity explicitly set

    # Stretching
    if left_hand_y < nose_y and right_hand_y < nose_y:
        activity = "Stretching"

    # Waving
    elif left_hand_y < nose_y or right_hand_y < nose_y:
        activity = "Waving"

    # Touching Hair
    elif abs(left_hand_y - nose_y) < 0.05 or abs(right_hand_y - nose_y) < 0.05:
        activity = "Touching Hair"

    # Touching Neck
    elif abs(left_hand_y - left_shoulder_y) < 0.05 or abs(right_hand_y - right_shoulder_y) < 0.05:
        activity = "Touching Neck"

    # Talking / Laughing detection using face landmarks
    if face_landmarks:
        top_lip = face_landmarks.landmark[13].y
        bottom_lip = face_landmarks.landmark[14].y
        mouth_gap = abs(bottom_lip - top_lip)

        if mouth_gap > 0.04 and mouth_gap < 0.07:
            activity = "Talking"
        elif mouth_gap >= 0.07:
            activity = "Laughing"

    # Moving Head
    if prev_nose_x is not None and abs(nose_x - prev_nose_x) > 0.05:
        activity = "Moving Head"

    return activity, nose_x

def is_fist(handLms):
    finger_tip_ids = [8, 12, 16, 20]
    folded = 0
    for tip in finger_tip_ids:
        if handLms.landmark[tip].y > handLms.landmark[tip-2].y:
            folded += 1
    return folded == 4

def detect_gesture(handLms_list=None, faceLms=None):
    if handLms_list and len(handLms_list) == 2:
        hand1 = handLms_list[0]
        hand2 = handLms_list[1]
        if is_fist(hand1) and is_fist(hand2):
            z1 = hand1.landmark[0].z
            z2 = hand2.landmark[0].z
            avg_z = (z1 + z2) / 2
            if avg_z < -0.2:
                return "zoom_in"
            elif avg_z > 0.2:
                return "zoom_out"

    if handLms_list and len(handLms_list) > 0:
        handLms = handLms_list[0]
        x, y = handLms.landmark[8].x, handLms.landmark[8].y
        cursor_x = int((1 - x) * screen_w)
        cursor_y = int(y * screen_h)
        pyautogui.moveTo(cursor_x, cursor_y)

        finger_tip_ids = [8, 12, 16, 20]
        fingers_open = sum(handLms.landmark[tip].y < handLms.landmark[tip-2].y for tip in finger_tip_ids)

        if fingers_open == 2:
            return "volume_up"
        elif fingers_open == 3:
            return "volume_down"

        index_tip_y = handLms.landmark[8].y
        index_base_y = handLms.landmark[6].y
        diff = index_base_y - index_tip_y
        if diff > 0.05:
            return "scroll_up"
        elif diff < -0.05:
            return "scroll_down"

    if faceLms:
        landmarks = faceLms.landmark
        gesture = detect_single_blink(landmarks, [33,160,158,133,153,144], side="left")
        if gesture: return gesture
        gesture = detect_single_blink(landmarks, [362,385,387,263,373,380], side="right")
        if gesture: return gesture

    return None

# ------------------- Main Loop -------------------

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        continue

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results_hands = mp_hands.process(imgRGB)
    results_face = mp_face.process(imgRGB)
    results_pose = mp_pose.process(imgRGB)

    if mode == "activity":
        if results_pose.pose_landmarks:
            mp_draw.draw_landmarks(img, results_pose.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
            face_landmarks = None
            if results_face.multi_face_landmarks:
                face_landmarks = results_face.multi_face_landmarks[0]
            activity, prev_nose_x = classify_activity(results_pose.pose_landmarks, face_landmarks, prev_nose_x)
            cv2.putText(img, f"Activity: {activity}", (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    elif mode == "gesture":
        if results_hands.multi_hand_landmarks:
            handLms_list = results_hands.multi_hand_landmarks
            for handLms in handLms_list:
                mp_draw.draw_landmarks(img, handLms, mp.solutions.hands.HAND_CONNECTIONS)
            gesture = detect_gesture(handLms_list=handLms_list)
            if gesture and gesture in gesture_actions:
                gesture_actions[gesture]()
                time.sleep(0.3)

        if results_face.multi_face_landmarks:
            for faceLms in results_face.multi_face_landmarks:
                gesture = detect_gesture(faceLms=faceLms)
                if gesture and gesture in gesture_actions:
                    gesture_actions[gesture]()
                    time.sleep(0.3)

    cv2.putText(img, f"Mode: {mode}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    cv2.imshow("Gesture + Eye + Activity Control System", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('a'):
        mode = "activity"
    elif key == ord('m'):
        mode = "gesture"

cap.release()
cv2.destroyAllWindows()