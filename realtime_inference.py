import os
import cv2
import time
import json
import numpy as np
from collections import deque
import tensorflow as tf
import mediapipe as mp

from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions


from fsl_helper.helper import (
    extract_features,
    draw_hand_landmarks,
    COLOR_MAP,
)

# -------------------------------
# Constants
# -------------------------------
MODEL_PATH = "models/hand_landmarker.task"
MODEL_FILE = "data/fsl_gru_model.keras"
LABEL_MAP_FILE = "data/label_map.json"

PRE_PADDING_SECONDS = 1
POST_PADDING_SECONDS = 5
FPS = 30
MAX_FRAMES = 15

NUM_HANDS = 2
MIN_CONFIDENCE = 0.5

PRE_FRAMES = PRE_PADDING_SECONDS * FPS
POST_FRAMES = POST_PADDING_SECONDS * FPS
WINDOW_FRAMES = PRE_FRAMES + POST_FRAMES

# -------------------------------
# Load model + labels
# -------------------------------
model = tf.keras.models.load_model(MODEL_FILE)
with open(LABEL_MAP_FILE, "r") as f:
    label_map = {int(k): v for k, v in json.load(f).items()}

# -------------------------------
# MediaPipe (IMAGE MODE)
# -------------------------------
options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.IMAGE,
    num_hands=NUM_HANDS,
    min_hand_detection_confidence=MIN_CONFIDENCE,
)


# -------------------------------
# Main loop
# -------------------------------
def main():
    cap = cv2.VideoCapture(0)

    # Finite buffer, infinite stream
    buffer = deque(maxlen=FPS * 20)  # ~20 seconds rolling video

    hand_active = False
    start_idx = None

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # --- Detect hand ONLY ---
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)

            hand_detected = bool(result.hand_landmarks)

            buffer.append((frame.copy(), hand_detected))

            # --- Hand appears ---
            if hand_detected and not hand_active:
                hand_active = True
                start_idx = len(buffer) - 1
                print("🟢 Hand appeared")

            # --- Hand disappears ---
            if not hand_detected and hand_active:
                end_idx = len(buffer) - 1
                hand_active = False
                print("🔴 Hand disappeared")

                # --- Compute window ---
                start = max(0, start_idx - PRE_FRAMES)
                end = min(len(buffer), end_idx + POST_FRAMES)

                gesture_frames = [fr for fr, _ in list(buffer)[start:end]]

                if len(gesture_frames) >= MAX_FRAMES:
                    # --- Sample like test mode ---
                    indices = np.linspace(
                        0, len(gesture_frames) - 1, MAX_FRAMES
                    ).astype(int)

                    sampled_frames = [gesture_frames[i] for i in indices]

                    # --- Extract features ONLY NOW ---
                    sequence = []
                    for fr in sampled_frames:
                        feat, _, _ = extract_features(fr, landmarker)
                        sequence.append(feat)

                    sequence = np.expand_dims(
                        np.array(sequence, dtype=np.float32), axis=0
                    )

                    preds = model.predict(sequence, verbose=0)
                    pred_id = int(np.argmax(preds))
                    confidence = float(np.max(preds))
                    label = label_map.get(pred_id, "UNKNOWN")

                    print(f"🖐️ Gesture: {label} ({confidence:.2f})")

                # --- Remove processed frames ---
                for _ in range(end):
                    buffer.popleft()

            # --- Visualization ---
            if result.hand_landmarks:
                for hand_lm in result.hand_landmarks:
                    draw_hand_landmarks(frame, hand_lm, COLOR_MAP)

            cv2.imshow("Realtime Video-style FSL", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
