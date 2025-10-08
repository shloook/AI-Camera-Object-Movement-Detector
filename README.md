# AI Camera Object & Movement Detector

A simple AI-based project that uses the system camera to:
- detect objects using a YOLO model (via `ultralytics` / `YOLOv8`),
- highlight detected objects with bounding boxes and minimal info (label + confidence),
- detect humans specifically and annotate them,
- detect movement (motion) and highlight moving objects.

**What's included (files):**
- `requirements.txt` — Python packages to install
- `README.md` — this file
- `main.py` — entrypoint which opens the camera and displays detections
- `detector.py` — wraps YOLO detection and drawing helpers
- `motion.py` — simple movement detection logic using background subtraction
- `utils.py` — small utility helpers
- `config.yaml` — optional config for thresholds
- `LICENSE` — MIT license

**Quick start (example)**

1. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux / macOS
   venv\Scripts\activate    # Windows
   ```

2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   python main.py
   ```

**Notes**
- This project expects internet access on first run to allow `ultralytics` to fetch a small default model (yolov8n). If you cannot download models, change the detection backend to a local weight file or use OpenCV DNN.
- The UI is an OpenCV window. Press `q` to quit, `s` to save a snapshot (in `snapshots/`).
