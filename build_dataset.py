import cv2
import pandas as pd
import os
import json

from helper.helper import draw_bbox

df = pd.read_csv("train.csv")
OUTPUT_ROOT = "dataset"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

for _, row in df.iterrows():
    video_path = row["vid_path"]
    label_id = int(row["id_label"])
    label_name = row["label"]

    out_dir = os.path.join(OUTPUT_ROOT, f"{label_name}")
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0
    paused = False
    saved_count = 0

    cv2.namedWindow("Annotator")
    cv2.setMouseCallback("Annotator", draw_bbox)

    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

        display = frame.copy()
        if bbox:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(
            display,
            f"Frame: {frame_idx}/{total} | Saved: {saved_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Annotator", display)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("q"):
            break

        elif key == ord(" "):
            paused = not paused

        elif key == ord("a"):
            frame_idx = max(0, frame_idx - 2)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        elif key == ord("d"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        elif key == ord("r"):
            bbox = None

        elif key == ord("s") and bbox is not None:
            x1, y1, x2, y2 = map(int, bbox)
            crop = frame[min(y1, y2) : max(y1, y2), min(x1, x2) : max(x1, x2)]

            img_name = f"img_{saved_count:04d}.jpg"
            ann_name = f"img_{saved_count:04d}.json"

            cv2.imwrite(os.path.join(out_dir, img_name), crop)

            ann = {
                "label": label_name,
                "label_id": label_id,
                "bbox": [x1, y1, x2, y2],
                "frame_index": frame_idx,
            }

            with open(os.path.join(out_dir, ann_name), "w") as f:
                json.dump(ann, f, indent=2)

            saved_count += 1
            print("✅ Saved", img_name)

    cap.release()
    cv2.destroyAllWindows()
