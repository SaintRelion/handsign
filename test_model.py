import os
import cv2
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions

# -------------------------------
# Constants
# -------------------------------
MODEL_PATH = "models/hand_landmarker.task"
TEST_CSV = "test.csv"
MODEL_FILE = "data/fsl_gru_model.keras"
LABEL_MAP_FILE = "data/label_map.json"

MAX_FRAMES = 10
NUM_HANDS = 2
MIN_CONFIDENCE = 0.5

SINGLE_HAND_FEATURES = 1 + 21 * 3  # 64
EXPECTED_FEATURE_LENGTH = SINGLE_HAND_FEATURES * 2  # both hands = 128

# -------------------------------
# Load label map
# -------------------------------
with open(LABEL_MAP_FILE, "r") as f:
    label_map = {int(k): v for k, v in json.load(f).items()}

# -------------------------------
# Load trained model
# -------------------------------
model = tf.keras.models.load_model(MODEL_FILE)

# -------------------------------
# MediaPipe setup
# -------------------------------
BaseOptions = mp.tasks.BaseOptions
HandLandmarkerOptions = vision.HandLandmarkerOptions
HandLandmarker = vision.HandLandmarker
VisionRunningMode = vision.RunningMode


options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=NUM_HANDS,
    min_hand_detection_confidence=MIN_CONFIDENCE,
)

# -------------------------------
# Import your helper functions
# -------------------------------
from fsl_helper.helper import extract_features

# -------------------------------
# Load test CSV
# -------------------------------
df = pd.read_csv(TEST_CSV)
X_test, y_test = [], []

with HandLandmarker.create_from_options(options) as landmarker:
    for _, row in tqdm(df.iterrows(), total=len(df)):
        vid_path = os.path.normpath(row["vid_path"])
        true_label = int(row["id_label"])

        cap = cv2.VideoCapture(vid_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            cap.release()
            continue

        frame_indices = np.linspace(0, total_frames - 1, MAX_FRAMES).astype(int)
        sequence = []

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            feat, _, _ = extract_features(frame, landmarker)
            sequence.append(feat)

        cap.release()

        if len(sequence) == 0:
            continue

        if len(sequence) < MAX_FRAMES:
            pad = np.zeros(
                (MAX_FRAMES - len(sequence), EXPECTED_FEATURE_LENGTH), dtype=np.float32
            )
            sequence = np.vstack([sequence, pad])

        X_test.append(sequence)
        y_test.append(true_label)

X_test = np.array(X_test, dtype=np.float32)
y_test = np.array(y_test, dtype=np.int32)

# -------------------------------
# Evaluate
# -------------------------------
preds = model.predict(X_test)
pred_labels = np.argmax(preds, axis=1)

accuracy = np.mean(pred_labels == y_test)
print(f"\n✅ Test Accuracy: {accuracy:.4f}")

# -------------------------------
# Optional: show mistakes
# -------------------------------
for i in range(len(y_test)):
    true_name = label_map[y_test[i]]
    pred_name = label_map[pred_labels[i]]
    if pred_labels[i] != y_test[i]:
        print(f"❌ True: {true_name} | Pred: {pred_name}")
    else:
        print(f"✅ True: {true_name} | Pred: {pred_name}")
