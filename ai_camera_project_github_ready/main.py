# main.py
# Entry point for the AI camera detector.
import cv2
import yaml
from detector import Detector, draw_detections
from motion import MotionDetector
from utils import save_snapshot

def load_config(path='config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    cfg = load_config()
    model_name = cfg.get('model', 'yolov8n.pt')
    conf_th = float(cfg.get('conf_threshold', 0.35))
    iou_th = float(cfg.get('iou_threshold', 0.45))
    motion_history = int(cfg.get('motion_history', 5))
    motion_area_threshold = int(cfg.get('motion_area_threshold', 500))

    detector = Detector(model_name=model_name, conf=conf_th, iou=iou_th)
    motion = MotionDetector(history=motion_history, area_threshold=motion_area_threshold)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('ERROR: Could not open camera.')
        return

    print('Press q to quit, s to save a snapshot.')
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fgmask, motion_found, motion_boxes = motion.detect(frame)
        detections = detector.predict(frame)
        out = draw_detections(frame, detections, motion_mask=fgmask)

        # annotate motion boxes
        if motion_found:
            for (x1,y1,x2,y2) in motion_boxes:
                cv2.rectangle(out, (x1,y1), (x2,y2), (0,0,255), 1)

        cv2.imshow('AI Camera Detector', out)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            path = save_snapshot(out)
            print(f'Snapshot saved to {path}')

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
