import cv2
import json
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
