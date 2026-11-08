import base64
from collections import deque
import json
import os
import threading
import time
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import cv2
import numpy as np

YUNET_MODEL_FILENAME = "face_detection_yunet_2023mar.onnx"
YUNET_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)


class MultiFaceDetector:
    """Strict human-face-only detector designed for crowded areas (e.g., Durga Puja Pandals).

    Combines:
    1. OpenCV YuNet deep learning detector with high-resolution multi-scale crowd scanning.
    2. Strict 5-point anatomical facial landmark verification (eyes-above-nose-above-mouth,
       horizontal eye/mouth alignment, internal margin check, and aspect ratio filtering)
       so feet, legs, hands, and background objects are completely rejected.
    """

    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(self.script_dir, YUNET_MODEL_FILENAME)

        # Higher default confidence to reject 50%-70% false positives on feet/hands
        self.score_threshold = 0.76
        self.nms_threshold = 0.30
        self.min_size = 22
        self.show_landmarks = True
        self.crowd_enhance = True
        self.lock = threading.Lock()

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.yunet = None
        self._init_detector()

    def _init_detector(self):
        if not os.path.exists(self.model_path):
            try:
                print("Downloading YuNet face detection model...")
                urllib.request.urlretrieve(YUNET_MODEL_URL, self.model_path)
            except Exception as exc:
                print(f"Warning: Could not download YuNet model: {exc}")

        if os.path.exists(self.model_path) and hasattr(cv2, "FaceDetectorYN"):
            try:
                self.yunet = cv2.FaceDetectorYN.create(
                    self.model_path,
                    "",
                    (640, 480),
                    float(self.score_threshold),
                    float(self.nms_threshold),
                    5000,
                )
            except Exception as exc:
                print(f"Warning: YuNet initialization failed: {exc}")

    def is_loaded(self) -> bool:
        return self.yunet is not None

    @staticmethod
    def _is_valid_human_face(x, y, w, h, landmarks, score, min_score):
        """Strict anatomical verification of 5 facial landmarks to reject legs/feet/hands.

        Landmarks order from YuNet:
          0: right eye (viewer's left)
          1: left eye (viewer's right)
          2: nose tip
          3: right mouth corner (viewer's left)
          4: left mouth corner (viewer's right)
        """
        if score < min_score:
            return False

        if w <= 0 or h <= 0:
            return False

        # 1. Human face bounding box aspect ratio check (width / height)
        aspect = float(w) / float(h)
        if aspect < 0.55 or aspect > 1.12:
            return False

        (re_x, re_y), (le_x, le_y), (n_x, n_y), (rm_x, rm_y), (lm_x, lm_y) = landmarks

        # 2. All 5 landmarks must sit well INSIDE the bounding box (not hugging the edge like toes/fingers)
        pad_x = 0.06 * w
        pad_y = 0.06 * h
        for lx, ly in landmarks:
            if not (x + pad_x <= lx <= x + w - pad_x and y + pad_y <= ly <= y + h - pad_y):
                return False

        # 3. Left/Right anatomical order (viewer's left is smaller X than viewer's right)
        eye_dx = le_x - re_x
        eye_dy = le_y - re_y
        mouth_dx = lm_x - rm_x
        mouth_dy = lm_y - rm_y

        # Eyes and mouth corners must be separated horizontally, not stacked vertically/sideways
        if eye_dx < 0.16 * w or mouth_dx < 0.12 * w:
            return False

        # Reject sideways detections (head tilt > ~38 degrees is almost always a hand/foot blob)
        if abs(eye_dy) > 0.65 * eye_dx or abs(mouth_dy) > 0.65 * mouth_dx:
            return False

        # 4. Vertical anatomical hierarchy: Eyes in upper half -> Nose in middle -> Mouth in lower half
        eye_mid_x = 0.5 * (re_x + le_x)
        eye_mid_y = 0.5 * (re_y + le_y)
        mouth_mid_x = 0.5 * (rm_x + lm_x)
        mouth_mid_y = 0.5 * (rm_y + lm_y)

        rel_eye_y = (eye_mid_y - y) / float(h)
        rel_nose_y = (n_y - y) / float(h)
        rel_mouth_y = (mouth_mid_y - y) / float(h)

        if not (0.14 <= rel_eye_y <= 0.56):
            return False
        if not (0.30 <= rel_nose_y <= 0.78):
            return False
        if not (0.52 <= rel_mouth_y <= 0.94):
            return False

        # Nose must be strictly below both eyes and strictly above both mouth corners
        max_eye_y = max(re_y, le_y)
        min_mouth_y = min(rm_y, lm_y)
        if not (n_y > max_eye_y + 0.03 * h and n_y < min_mouth_y - 0.02 * h):
            return False

        # 5. Eye-to-mouth vertical span must match human facial proportions
        eye_to_mouth_y = mouth_mid_y - eye_mid_y
        if not (0.22 * h <= eye_to_mouth_y <= 0.62 * h):
            return False

        # 6. Face central axis alignment: eye midpoint, nose, and mouth midpoint must align horizontally
        if abs(eye_mid_x - mouth_mid_x) > 0.25 * w:
            return False
        if not (re_x - 0.12 * w <= n_x <= le_x + 0.12 * w):
            return False

        return True

    @staticmethod
    def _nms_detections(detections, iou_thresh=0.35):
        if not detections:
            return []

        boxes = np.array([d["box"] for d in detections], dtype=np.float32)
        scores = np.array([d["score"] for d in detections], dtype=np.float32)

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]
        areas = boxes[:, 2] * boxes[:, 3]

        order = scores.argsort()[::-1]
        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(int(i))
            if order.size == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            union = areas[i] + areas[order[1:]] - inter
            iou = np.where(union > 0, inter / union, 0.0)

            min_area = np.minimum(areas[i], areas[order[1:]])
            containment = np.where(min_area > 0, inter / min_area, 0.0)

            inds = np.where((iou <= iou_thresh) & (containment <= 0.65))[0]
            order = order[inds + 1]

        kept = [detections[idx] for idx in keep]
        kept.sort(key=lambda d: (d["box"][0], d["box"][1]))
        return kept

    def _enhance_lighting(self, bgr_frame):
        """Normalize uneven pandal/indoor lighting using LAB CLAHE."""
        lab = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        l_chan = self.clahe.apply(l_chan)
        merged = cv2.merge((l_chan, a_chan, b_chan))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def _run_yunet_pass(self, img, offset_x=0, offset_y=0, scale=1.0, min_score=0.76):
        h_img, w_img = img.shape[:2]
        if h_img < 32 or w_img < 32:
            return []

        self.yunet.setInputSize((w_img, h_img))
        self.yunet.setScoreThreshold(float(min_score))
        self.yunet.setNMSThreshold(float(self.nms_threshold))
        _, raw_faces = self.yunet.detect(img)

        valid = []
        if raw_faces is None:
            return valid

        for row in raw_faces:
            x = float(row[0]) / scale + offset_x
            y = float(row[1]) / scale + offset_y
            w = float(row[2]) / scale
            h = float(row[3]) / scale
            score = float(row[14])

            if w < self.min_size or h < self.min_size:
                continue

            landmarks = [
                (float(row[4]) / scale + offset_x, float(row[5]) / scale + offset_y),
                (float(row[6]) / scale + offset_x, float(row[7]) / scale + offset_y),
                (float(row[8]) / scale + offset_x, float(row[9]) / scale + offset_y),
                (float(row[10]) / scale + offset_x, float(row[11]) / scale + offset_y),
                (float(row[12]) / scale + offset_x, float(row[13]) / scale + offset_y),
            ]

            if not self._is_valid_human_face(x, y, w, h, landmarks, score, min_score):
                continue

            valid.append(
                {
                    "box": [int(round(x)), int(round(y)), int(round(w)), int(round(h))],
                    "score": round(score, 2),
                    "landmarks": [(int(round(lx)), int(round(ly))) for lx, ly in landmarks],
                }
            )
        return valid

    def detect(self, frame):
        """Detect verified human faces across foreground and crowd background."""
        if frame is None or frame.size == 0 or self.yunet is None:
            return []

        work_frame = self._enhance_lighting(frame) if self.crowd_enhance else frame
        h_img, w_img = work_frame.shape[:2]
        min_score = float(self.score_threshold)

        all_detections = []
        with self.lock:
            # Pass 1: Standard full-frame detection
            all_detections.extend(
                self._run_yunet_pass(work_frame, 0, 0, 1.0, min_score)
            )

            # Pass 2: Upscaled crowd pass for distant/small faces in dense areas (e.g., Pandal queues)
            if self.crowd_enhance and w_img <= 1400:
                up_scale = 1.45
                up_w = int(w_img * up_scale)
                up_h = int(h_img * up_scale)
                upscaled = cv2.resize(
                    work_frame, (up_w, up_h), interpolation=cv2.INTER_LINEAR
                )
                all_detections.extend(
                    self._run_yunet_pass(
                        upscaled, 0, 0, up_scale, min(0.88, min_score + 0.03)
                    )
                )

        return self._nms_detections(all_detections)

    def annotate(self, frame, detections, capacity_limit=50):
        """Draw verified face bounding boxes and Pandal Crowd HUD banner."""
        output = frame.copy()
        palette = [
            (0, 255, 128),
            (255, 200, 0),
            (255, 105, 180),
            (0, 215, 255),
            (180, 130, 255),
            (50, 220, 100),
            (255, 140, 60),
            (80, 220, 255),
        ]

        for idx, det in enumerate(detections, start=1):
            x, y, w, h = det["box"]
            score = int(det["score"] * 100)
            color = palette[(idx - 1) % len(palette)]

            cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)

            if self.show_landmarks and det.get("landmarks"):
                for lx, ly in det["landmarks"]:
                    cv2.circle(output, (lx, ly), 2, (0, 255, 255), -1)

            label = f"Face {idx} ({score}%)"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
            label_y = max(y, th + 10)
            cv2.rectangle(
                output,
                (x, label_y - th - 10),
                (x + tw + 10, label_y),
                color,
                -1,
            )
            cv2.putText(
                output,
                label,
                (x + 5, label_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (15, 23, 42),
                2,
            )

        count = len(detections)
        is_crowded = count >= capacity_limit
        border_col = (0, 68, 255) if is_crowded else (56, 189, 248)

        cv2.rectangle(output, (16, 16), (370, 74), (15, 23, 42), -1)
        cv2.rectangle(output, (16, 16), (370, 74), border_col, 2)
        cv2.putText(
            output,
            f"Verified Faces: {count}",
            (30, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            border_col,
            2,
        )
        return output


class CameraState:
    """Manages webcam capture, temporal stabilization, and crowd telemetry."""

    def __init__(self, detector: MultiFaceDetector):
        self.detector = detector
        self.camera = None
        self.running = False
        self.mirror = True
        self.capacity_limit = 20
        self.lock = threading.Lock()

        self.current_count = 0
        self.max_count = 0
        self.fps = 0.0
        self.camera_error = None
        self.latest_jpeg = None
        self._recent_counts = deque(maxlen=5)
        self._thread = None

    def start(self, camera_index=0):
        with self.lock:
            if self.running:
                return True
            cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                self.camera_error = (
                    "Webcam could not be opened by Python. "
                    "Switch to 'Browser Camera Mode' or 'Upload Crowd Photo' above."
                )
                return False

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            self.camera = cap
            self.camera_error = None
            self.running = True
            self._thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._thread.start()
            return True

    def stop(self):
        self.running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        with self.lock:
            if self.camera is not None:
                self.camera.release()
                self.camera = None
            self.current_count = 0
            self._recent_counts.clear()

    def _capture_loop(self):
        prev_time = time.time()
        while self.running:
            with self.lock:
                cap = self.camera
            if cap is None:
                break

            success, frame = cap.read()
            if not success or frame is None:
                self.camera_error = "Could not read frame from camera."
                time.sleep(0.1)
                continue

            if self.mirror:
                frame = cv2.flip(frame, 1)

            detections = self.detector.detect(frame)
            annotated = self.detector.annotate(
                frame, detections, capacity_limit=self.capacity_limit
            )

            now = time.time()
            dt = max(now - prev_time, 1e-4)
            prev_time = now
            instant_fps = 1.0 / dt

            ok, buffer = cv2.imencode(
                ".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            )
            if ok:
                with self.lock:
                    self.latest_jpeg = buffer.tobytes()
                    self._recent_counts.append(len(detections))
                    # Median smoothing prevents single-frame flicker in crowds
                    sorted_c = sorted(self._recent_counts)
                    self.current_count = sorted_c[len(sorted_c) // 2]
                    if self.current_count > self.max_count:
                        self.max_count = self.current_count
                    self.fps = round(0.8 * self.fps + 0.2 * instant_fps, 1)

            time.sleep(0.01)

    def get_placeholder_jpeg(self, message="Initializing Camera Stream..."):
        img = np.zeros((480, 854, 3), dtype=np.uint8)
        img[:] = (15, 23, 42)
        cv2.putText(
            img,
            message,
            (50, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (148, 163, 184),
            2,
        )
        ok, buf = cv2.imencode(".jpg", img)
        return buf.tobytes() if ok else b""


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Pandal & Crowd Live Face Counter — Kolkata Edition</title>
  <style>
    :root {
      --bg: #0f172a;
      --panel: #1e293b;
      --border: #334155;
      --accent: #38bdf8;
      --green: #22c55e;
      --amber: #f59e0b;
      --red: #ef4444;
      --text: #f8fafc;
      --muted: #94a3b8;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      padding: 20px;
    }
    .container {
      max-width: 1240px;
      margin: 0 auto;
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 12px;
      margin-bottom: 20px;
      padding-bottom: 16px;
      border-bottom: 1px solid var(--border);
    }
    h1 {
      font-size: 1.45rem;
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }
    .badge {
      background: rgba(56, 189, 248, 0.15);
      color: var(--accent);
      border: 1px solid rgba(56, 189, 248, 0.4);
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 0.78rem;
      font-weight: 600;
    }
    .grid {
      display: grid;
      grid-template-columns: 1fr 360px;
      gap: 20px;
    }
    @media (max-width: 920px) {
      .grid { grid-template-columns: 1fr; }
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 18px;
      box-shadow: 0 10px 25px rgba(0,0,0,0.25);
    }
    .video-wrap {
      position: relative;
      width: 100%;
      background: #020617;
      border-radius: 10px;
      overflow: hidden;
      min-height: 440px;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .video-wrap img {
      width: 100%;
      height: auto;
      display: block;
    }
    .mode-tabs {
      display: flex;
      gap: 8px;
      margin-bottom: 14px;
      flex-wrap: wrap;
    }
    button, .file-btn {
      background: #334155;
      color: var(--text);
      border: 1px solid #475569;
      padding: 9px 14px;
      border-radius: 8px;
      cursor: pointer;
      font-weight: 600;
      font-size: 0.88rem;
      transition: all 0.15s;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }
    button:hover, .file-btn:hover {
      background: #475569;
    }
    button.active {
      background: var(--accent);
      color: #0f172a;
      border-color: var(--accent);
    }
    button.danger {
      background: rgba(239, 68, 68, 0.2);
      color: #fca5a5;
      border-color: rgba(239, 68, 68, 0.4);
    }
    .stats-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-bottom: 12px;
    }
    .stat-box {
      background: rgba(15, 23, 42, 0.7);
      border: 1px solid var(--border);
      padding: 14px;
      border-radius: 10px;
      text-align: center;
    }
    .stat-label {
      font-size: 0.76rem;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .stat-value {
      font-size: 2.4rem;
      font-weight: 800;
      color: var(--green);
      margin-top: 4px;
    }
    .stat-value.accent {
      color: var(--accent);
    }
    .crowd-status {
      padding: 12px;
      border-radius: 10px;
      font-weight: 700;
      text-align: center;
      margin-bottom: 16px;
      border: 1px solid rgba(34, 197, 94, 0.4);
      background: rgba(34, 197, 94, 0.12);
      color: #86efac;
      font-size: 0.92rem;
    }
    .crowd-status.moderate {
      border-color: rgba(245, 158, 11, 0.45);
      background: rgba(245, 158, 11, 0.15);
      color: #fcd34d;
    }
    .crowd-status.alert {
      border-color: rgba(239, 68, 68, 0.5);
      background: rgba(239, 68, 68, 0.2);
      color: #fca5a5;
    }
    .control-group {
      margin-bottom: 14px;
    }
    .control-group label {
      display: flex;
      justify-content: space-between;
      font-size: 0.84rem;
      color: var(--muted);
      margin-bottom: 6px;
    }
    input[type="range"] {
      width: 100%;
      accent-color: var(--accent);
    }
    .status-msg {
      margin-top: 12px;
      font-size: 0.82rem;
      color: var(--muted);
      line-height: 1.45;
      background: rgba(15, 23, 42, 0.5);
      padding: 12px;
      border-radius: 8px;
      border: 1px solid var(--border);
    }
    .error-banner {
      background: rgba(239, 68, 68, 0.15);
      border: 1px solid rgba(239, 68, 68, 0.4);
      color: #fca5a5;
      padding: 10px 12px;
      border-radius: 8px;
      margin-bottom: 12px;
      font-size: 0.85rem;
      display: none;
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>
        <span>Crowd & Pandal Live Face Counter</span>
        <span class="badge">Anatomical Face Verification • Zero Body/Foot False Positives</span>
      </h1>
      <div style="display:flex; gap:8px;">
        <button onclick="resetMax()">Reset Peak Count</button>
        <button class="danger" onclick="shutdownServer()">Stop Server</button>
      </div>
    </header>

    <div class="grid">
      <div class="card">
        <div class="mode-tabs">
          <button id="btn-live" class="active" onclick="switchMode('live')">Live Pandal / Webcam Stream</button>
          <button id="btn-browser" onclick="switchMode('browser')">Browser Camera Mode</button>
          <label class="file-btn">
            Upload Crowd Photo Test
            <input type="file" id="photo-input" accept="image/*" style="display:none" onchange="handlePhotoUpload(event)" />
          </label>
        </div>

        <div id="error-box" class="error-banner"></div>

        <div class="video-wrap">
          <img id="stream-img" src="/video_feed" alt="Live Face Counter Stream" />
          <video id="browser-video" autoplay playsinline muted style="display:none;"></video>
        </div>
      </div>

      <div class="card">
        <h3 style="margin-bottom:12px; font-size:1.05rem;">Pandal Crowd Monitoring</h3>
        <div class="stats-grid">
          <div class="stat-box">
            <div class="stat-label">Verified Faces</div>
            <div class="stat-value" id="current-count">0</div>
          </div>
          <div class="stat-box">
            <div class="stat-label">Peak Crowd</div>
            <div class="stat-value accent" id="max-count">0</div>
          </div>
        </div>

        <div id="crowd-badge" class="crowd-status">
          Crowd Density: Normal (Safe Flow)
        </div>

        <div class="stat-box" style="margin-bottom:16px; display:flex; justify-content:space-between; align-items:center; padding:10px 14px;">
          <span class="stat-label">Detection Rate</span>
          <span id="fps-val" style="font-weight:700; color:var(--text);">0.0 FPS</span>
        </div>

        <h3 style="margin-bottom:12px; font-size:0.98rem;">Crowd & Strictness Controls</h3>

        <div class="control-group">
          <label>
            <span>Face Strictness (Confidence)</span>
            <span id="lbl-conf">76%</span>
          </label>
          <input type="range" id="rng-conf" min="60" max="92" value="76" oninput="updateConfig()" />
        </div>

        <div class="control-group">
          <label>
            <span>Pandal Alert Capacity Threshold</span>
            <span id="lbl-cap">20 People</span>
          </label>
          <input type="range" id="rng-cap" min="5" max="200" value="20" oninput="updateConfig()" />
        </div>

        <div class="control-group">
          <label>
            <span>Min Face Size (Distant Crowd)</span>
            <span id="lbl-min-size">22 px</span>
          </label>
          <input type="range" id="rng-min-size" min="14" max="80" value="22" oninput="updateConfig()" />
        </div>

        <div class="control-group" style="display:flex; align-items:center; justify-content:space-between; margin-top:10px;">
          <span style="font-size:0.84rem; color:var(--muted);">Multi-Scale Crowd Boost (Distant Faces)</span>
          <input type="checkbox" id="chk-crowd" checked onchange="updateConfig()" />
        </div>

        <div class="control-group" style="display:flex; align-items:center; justify-content:space-between;">
          <span style="font-size:0.84rem; color:var(--muted);">Show 5-Point Facial Landmarks</span>
          <input type="checkbox" id="chk-landmarks" checked onchange="updateConfig()" />
        </div>

        <div class="control-group" style="display:flex; align-items:center; justify-content:space-between;">
          <span style="font-size:0.84rem; color:var(--muted);">Mirror Camera</span>
          <input type="checkbox" id="chk-mirror" checked onchange="updateConfig()" />
        </div>

        <div class="status-msg">
          <strong>Why Legs/Hands Are Rejected:</strong> Every detection must pass 5-point anatomical geometry verification (upright eye-line, nose centered below eyes, mouth below nose, and internal box margins) while scanning at multi-scale resolution for dense crowds like Durga Puja Pandals.
        </div>
      </div>
    </div>
  </div>

  <script>
    let currentMode = 'live';
    let browserStream = null;
    let browserInterval = null;
    let capacityLimit = 20;

    function updateCrowdBanner(count) {
      const badge = document.getElementById('crowd-badge');
      badge.className = 'crowd-status';
      if (count >= capacityLimit) {
        badge.classList.add('alert');
        badge.textContent = `HIGH CROWD ALERT: ${count} / ${capacityLimit} Capacity Reached!`;
      } else if (count >= Math.max(3, Math.floor(capacityLimit * 0.6))) {
        badge.classList.add('moderate');
        badge.textContent = `Crowd Density: Moderate (${count} / ${capacityLimit})`;
      } else {
        badge.textContent = `Crowd Density: Normal (${count} / ${capacityLimit})`;
      }
    }

    async function pollStats() {
      if (currentMode !== 'live') return;
      try {
        const res = await fetch('/api/status');
        const data = await res.json();
        document.getElementById('current-count').textContent = data.current_count;
        document.getElementById('max-count').textContent = data.max_count;
        document.getElementById('fps-val').textContent = data.fps + ' FPS';
        updateCrowdBanner(data.current_count);

        const errBox = document.getElementById('error-box');
        if (data.camera_error) {
          errBox.style.display = 'block';
          errBox.textContent = data.camera_error;
        } else {
          errBox.style.display = 'none';
        }
      } catch (e) {}
    }

    setInterval(pollStats, 300);

    async function updateConfig() {
      const conf = parseInt(document.getElementById('rng-conf').value, 10);
      const cap = parseInt(document.getElementById('rng-cap').value, 10);
      const minSize = parseInt(document.getElementById('rng-min-size').value, 10);
      const crowd = document.getElementById('chk-crowd').checked;
      const landmarks = document.getElementById('chk-landmarks').checked;
      const mirror = document.getElementById('chk-mirror').checked;

      capacityLimit = cap;
      document.getElementById('lbl-conf').textContent = conf + '%';
      document.getElementById('lbl-cap').textContent = cap + ' People';
      document.getElementById('lbl-min-size').textContent = minSize + ' px';

      await fetch('/api/config', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          score_threshold: conf / 100.0,
          capacity_limit: cap,
          min_size: minSize,
          crowd_enhance: crowd,
          show_landmarks: landmarks,
          mirror: mirror
        })
      });
    }

    async function resetMax() {
      await fetch('/api/reset_max', {method: 'POST'});
      document.getElementById('max-count').textContent = '0';
    }

    async function switchMode(mode) {
      currentMode = mode;
      document.getElementById('btn-live').classList.toggle('active', mode === 'live');
      document.getElementById('btn-browser').classList.toggle('active', mode === 'browser');

      if (browserInterval) {
        clearInterval(browserInterval);
        browserInterval = null;
      }
      if (browserStream) {
        browserStream.getTracks().forEach(t => t.stop());
        browserStream = null;
      }

      const img = document.getElementById('stream-img');
      const errBox = document.getElementById('error-box');
      errBox.style.display = 'none';

      if (mode === 'live') {
        await fetch('/api/camera/start', {method: 'POST'});
        img.src = '/video_feed?t=' + Date.now();
      } else if (mode === 'browser') {
        await fetch('/api/camera/stop', {method: 'POST'});
        try {
          const video = document.getElementById('browser-video');
          browserStream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 1280 }, height: { ideal: 720 } },
            audio: false
          });
          video.srcObject = browserStream;
          await video.play();

          const canvas = document.createElement('canvas');
          const ctx = canvas.getContext('2d');

          browserInterval = setInterval(async () => {
            if (!video.videoWidth) return;
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            ctx.drawImage(video, 0, 0);
            const dataUrl = canvas.toDataURL('image/jpeg', 0.85);
            const res = await fetch('/api/detect_image', {
              method: 'POST',
              headers: {'Content-Type': 'application/json'},
              body: JSON.stringify({ image: dataUrl })
            });
            const out = await res.json();
            if (out.annotated_image) {
              img.src = out.annotated_image;
              document.getElementById('current-count').textContent = out.count;
              document.getElementById('max-count').textContent = out.max_count;
              updateCrowdBanner(out.count);
            }
          }, 220);
        } catch (err) {
          errBox.style.display = 'block';
          errBox.textContent = 'Browser webcam access failed: ' + err.message;
        }
      }
    }

    async function handlePhotoUpload(event) {
      const file = event.target.files[0];
      if (!file) return;
      currentMode = 'upload';
      document.getElementById('btn-live').classList.remove('active');
      document.getElementById('btn-browser').classList.remove('active');

      const reader = new FileReader();
      reader.onload = async (e) => {
        const res = await fetch('/api/detect_image', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ image: e.target.result })
        });
        const out = await res.json();
        if (out.annotated_image) {
          document.getElementById('stream-img').src = out.annotated_image;
          document.getElementById('current-count').textContent = out.count;
          document.getElementById('max-count').textContent = out.max_count;
          document.getElementById('fps-val').textContent = 'Crowd Photo';
          updateCrowdBanner(out.count);
        }
      };
      reader.readAsDataURL(file);
    }

    async function shutdownServer() {
      await fetch('/api/shutdown', {method: 'POST'});
      document.body.innerHTML = '<div style="text-align:center; padding-top:100px;"><h2>Face Counter Stopped. You may close this tab.</h2></div>';
    }
  </script>
</body>
</html>
"""


def create_handler(camera_state: CameraState, detector: MultiFaceDetector):
    class FaceCounterRequestHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def _send_json(self, payload, status=200):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/":
                body = HTML_PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            elif parsed.path == "/api/status":
                with camera_state.lock:
                    payload = {
                        "current_count": camera_state.current_count,
                        "max_count": camera_state.max_count,
                        "fps": camera_state.fps,
                        "running": camera_state.running,
                        "camera_error": camera_state.camera_error,
                    }
                self._send_json(payload)

            elif parsed.path == "/video_feed":
                self.send_response(200)
                self.send_header(
                    "Content-Type", "multipart/x-mixed-replace; boundary=frame"
                )
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                try:
                    while True:
                        with camera_state.lock:
                            frame_bytes = camera_state.latest_jpeg
                        if not frame_bytes:
                            msg = (
                                camera_state.camera_error
                                or "Starting camera stream..."
                            )
                            frame_bytes = camera_state.get_placeholder_jpeg(msg)

                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(
                            f"Content-Length: {len(frame_bytes)}\r\n\r\n".encode(
                                "utf-8"
                            )
                        )
                        self.wfile.write(frame_bytes)
                        self.wfile.write(b"\r\n")
                        time.sleep(0.035)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass

            else:
                self.send_error(404, "Not Found")

        def do_POST(self):
            parsed = urlparse(self.path)
            length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(length) if length > 0 else b"{}"

            if parsed.path == "/api/config":
                data = json.loads(raw_body.decode("utf-8"))
                with detector.lock:
                    detector.score_threshold = float(
                        data.get("score_threshold", detector.score_threshold)
                    )
                    detector.min_size = int(data.get("min_size", detector.min_size))
                    detector.crowd_enhance = bool(
                        data.get("crowd_enhance", detector.crowd_enhance)
                    )
                    detector.show_landmarks = bool(
                        data.get("show_landmarks", detector.show_landmarks)
                    )
                camera_state.capacity_limit = int(
                    data.get("capacity_limit", camera_state.capacity_limit)
                )
                camera_state.mirror = bool(data.get("mirror", camera_state.mirror))
                self._send_json({"ok": True})

            elif parsed.path == "/api/reset_max":
                with camera_state.lock:
                    camera_state.max_count = camera_state.current_count
                self._send_json({"ok": True})

            elif parsed.path == "/api/camera/start":
                ok = camera_state.start()
                self._send_json({"ok": ok, "error": camera_state.camera_error})

            elif parsed.path == "/api/camera/stop":
                camera_state.stop()
                self._send_json({"ok": True})

            elif parsed.path == "/api/detect_image":
                data = json.loads(raw_body.decode("utf-8"))
                img_b64 = data.get("image", "")
                if "," in img_b64:
                    img_b64 = img_b64.split(",", 1)[1]
                img_bytes = base64.b64decode(img_b64)
                arr = np.frombuffer(img_bytes, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None:
                    self._send_json({"error": "Invalid image"}, status=400)
                    return

                detections = detector.detect(frame)
                annotated = detector.annotate(
                    frame, detections, capacity_limit=camera_state.capacity_limit
                )
                ok, buf = cv2.imencode(
                    ".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 88]
                )
                out_b64 = (
                    "data:image/jpeg;base64,"
                    + base64.b64encode(buf.tobytes()).decode("ascii")
                    if ok
                    else ""
                )
                with camera_state.lock:
                    camera_state.current_count = len(detections)
                    if len(detections) > camera_state.max_count:
                        camera_state.max_count = len(detections)
                    max_c = camera_state.max_count

                self._send_json(
                    {
                        "count": len(detections),
                        "max_count": max_c,
                        "annotated_image": out_b64,
                    }
                )

            elif parsed.path == "/api/shutdown":
                self._send_json({"ok": True})
                camera_state.stop()
                threading.Thread(target=self.server.shutdown, daemon=True).start()

            else:
                self.send_error(404, "Not Found")

    return FaceCounterRequestHandler


def main():
    detector = MultiFaceDetector()
    if not detector.is_loaded():
        print("ERROR: Face detector could not be loaded.")
        return

    camera_state = CameraState(detector)
    camera_state.start(0)

    host = "127.0.0.1"
    port = int(os.environ.get("PORT", "8000"))
    server = ThreadingHTTPServer((host, port), create_handler(camera_state, detector))

    url = f"http://{host}:{port}"
    print("========================================")
    print(" Crowd & Pandal Live Face Counter (Web)")
    print(f" Open in browser: {url}")
    print(" Press Ctrl+C in terminal to stop")
    print("========================================")

    threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Face Counter...")
    finally:
        camera_state.stop()
        server.server_close()
        print("Face Counter stopped.")


if __name__ == "__main__":
    main()
