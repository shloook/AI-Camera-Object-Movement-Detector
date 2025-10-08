# detector.py
# Wraps ultralytics YOLO detection model and drawing helpers.
from ultralytics import YOLO
import cv2
import numpy as np

class Detector:
    def __init__(self, model_name='yolov8n.pt', conf=0.35, iou=0.45):
        self.model = YOLO(model_name)
        self.conf = conf
        self.iou = iou

    def predict(self, frame):
        """
        Run inference on a single BGR frame (numpy array).
        Returns a list of detections: dicts with keys: box (x1,y1,x2,y2), conf, class_id, label
        """
        # ultralytics expects RGB images as numpy arrays in HWC format
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model.predict(source=[rgb], imgsz=640, conf=self.conf, iou=self.iou, verbose=False)
        # results is a list; we passed a single image so results[0] corresponds to it
        out = []
        r = results[0]
        # r.boxes.xyxy, r.boxes.conf, r.boxes.cls
        if hasattr(r, 'boxes') and r.boxes is not None:
            boxes = r.boxes
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy[0], 'cpu') else box.xyxy[0].numpy()
                conf = float(box.conf[0]) if hasattr(box.conf[0], 'item') or hasattr(box.conf[0], 'numpy') else float(box.conf[0])
                cls = int(box.cls[0]) if hasattr(box.cls[0], 'item') or hasattr(box.cls[0], 'numpy') else int(box.cls[0])
                label = self.model.names.get(cls, str(cls))
                out.append({
                    "box": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
                    "conf": conf,
                    "class_id": cls,
                    "label": label
                })
        return out

def draw_detections(frame, detections, motion_mask=None):
    """
    Draw bounding boxes and minimal info on frame (BGR).
    If motion_mask is provided, boxes overlapping motion regions will be emphasized.
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det['box']
        label = det['label']
        conf = det['conf']
        centerx = (x1 + x2) // 2
        centery = (y1 + y2) // 2

        # Decide if this box overlaps motion mask
        is_moving = False
        if motion_mask is not None:
            # clamp coordinates
            xx1, yy1 = max(0, x1), max(0, y1)
            xx2, yy2 = min(w-1, x2), min(h-1, y2)
            roi = motion_mask[yy1:yy2+1, xx1:xx2+1]
            if roi.size > 0 and roi.mean() > 10:  # some pixels indicate motion
                is_moving = True

        # color: green for normal, red if human, blue if moving (overrides)
        color = (0, 255, 0)
        if label.lower() == 'person':
            color = (0, 128, 255)  # orange-ish for person
        if is_moving:
            color = (255, 0, 0)  # blue for moving

        thickness = 2 if not is_moving else 3
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)

        # minimal info text: label + confidence (rounded)
        info = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # background rectangle for text
        cv2.rectangle(overlay, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
        cv2.putText(overlay, info, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    # blend overlay
    cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
    return frame
