import cv2
import pandas as pd
import os

drawing = False
ix, iy = -1, -1
bbox = None


def draw_bbox(event, x, y, flags, param):
    global ix, iy, drawing, bbox

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        bbox = None

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        bbox = (ix, iy, x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        bbox = (ix, iy, x, y)


df = pd.read_csv("train.csv")
OUTPUT_ROOT = "dataset"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

GLOBAL_ROW_IDX = 311
for i, row in enumerate(df.iterrows()):
    if i < GLOBAL_ROW_IDX:
        continue

    video_path = row[1]["vid_path"]
    label_id = int(row[1]["id_label"])
    label_name = row[1]["label"].strip()

    out_dir = os.path.join(OUTPUT_ROOT, f"{label_name}")
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0

    existing_files = [f for f in os.listdir(out_dir) if f.endswith(".jpg")]
    saved_count = len(existing_files)

    paused = False

    cv2.namedWindow("Annotator", cv2.WINDOW_NORMAL)
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
        elif key == 27:  # ESC key to skip current video
            print("⏭ Skipping current video...")
            bbox = None
            break
        elif key == ord(" "):
            paused = not paused
        elif key == ord("a"):
            frame_idx = max(0, frame_idx - 7)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        elif key == ord("d"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        elif key == ord("r"):
            bbox = None
        elif key == ord("s") and bbox is not None:
            x1, y1, x2, y2 = map(int, bbox)
            crop = frame[min(y1, y2) : max(y1, y2), min(x1, x2) : max(x1, x2)]

            # Save crop using global counter
            img_name = f"img_{GLOBAL_ROW_IDX:04d}.jpg"
            cv2.imwrite(os.path.join(out_dir, img_name), crop)
            GLOBAL_ROW_IDX += 1
            saved_count += 1
            print(f"✅ Saved {img_name} in {out_dir}")

    cap.release()
    cv2.destroyAllWindows()
