"""
NOVA Camera Vision System
- Real-time face detection and presence sensing
- Emotion recognition from facial expressions
- Gesture detection (wave, thumbs up, raised hand)
- Person identification (recognizes the owner over time)
- Scene understanding via Llama 3.2 Vision
- Auto-wakes Nova when person sits at desk
"""

import cv2
import base64
import time
import threading
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Callable, Dict, List, Tuple
from dataclasses import dataclass, field
from collections import deque
from io import BytesIO

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

log = logging.getLogger("nova.camera")


@dataclass
class FaceState:
    present: bool = False
    count: int = 0
    emotion: str = "neutral"
    emotion_confidence: float = 0.0
    gaze_direction: str = "forward"  # forward, away, down
    attention_score: float = 1.0     # 0-1, how focused on screen
    last_seen: float = 0.0
    bbox: Optional[Tuple] = None


@dataclass
class GestureEvent:
    gesture: str            # raise_hand, thumbs_up, wave, point
    confidence: float
    timestamp: float


@dataclass
class PresenceEvent:
    arrived: bool           # True=arrived, False=left
    timestamp: float
    face_count: int


class CameraVision:
    """
    Full camera intelligence system for NOVA.
    Runs in background thread, fires callbacks on events.
    """

    def __init__(self):
        self._cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # State
        self.face_state = FaceState()
        self._prev_face_present = False
        self._last_presence_event = 0.0
        self._last_emotion_update = 0.0
        self._last_gesture_check = 0.0
        self._gesture_cooldown = 3.0  # seconds between gesture events

        # History
        self._emotion_history: deque = deque(maxlen=20)
        self._frame_buffer: deque = deque(maxlen=5)  # recent frames for analysis

        # Face detection (OpenCV built-in Haar cascades — no extra models needed)
        self._face_cascade = None
        self._eye_cascade = None
        self._load_cascades()

        # Optional: DeepFace for emotion (installed separately)
        self._deepface_available = False
        self._try_load_deepface()

        # Optional: MediaPipe for hands/gestures
        self._mp_hands = None
        self._mp_drawing = None
        self._try_load_mediapipe()

        # Callbacks
        self.on_presence_change: Optional[Callable[[PresenceEvent], None]] = None
        self.on_emotion_change: Optional[Callable[[str, float], None]] = None
        self.on_gesture: Optional[Callable[[GestureEvent], None]] = None
        self.on_attention_lost: Optional[Callable] = None
        self.on_attention_gained: Optional[Callable] = None

        log.info("Camera vision system initialized")

    def _load_cascades(self):
        """Load OpenCV Haar cascade classifiers."""
        try:
            cv_data = cv2.__file__
            cv_dir = Path(cv_data).parent / "data"

            face_path = str(cv_dir / "haarcascade_frontalface_default.xml")
            eye_path  = str(cv_dir / "haarcascade_eye.xml")

            self._face_cascade = cv2.CascadeClassifier(face_path)
            self._eye_cascade  = cv2.CascadeClassifier(eye_path)

            if self._face_cascade.empty():
                # Try absolute fallback paths
                for candidate in [
                    "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
                    "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
                ]:
                    if Path(candidate).exists():
                        self._face_cascade = cv2.CascadeClassifier(candidate)
                        break

            log.info("OpenCV face cascade loaded")
        except Exception as e:
            log.warning(f"Could not load face cascade: {e}")

    def _try_load_deepface(self):
        """Try to load DeepFace for rich emotion analysis."""
        try:
            from deepface import DeepFace
            self._deepface = DeepFace
            self._deepface_available = True
            log.info("DeepFace emotion analysis available")
        except ImportError:
            log.info("DeepFace not installed — using basic emotion proxy")
            self._deepface_available = False

    def _try_load_mediapipe(self):
        """Try to load MediaPipe Tasks API for gesture detection (2026 compatible)."""
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            # Path to the downloaded model
            model_path = str(Path(__file__).parent / "models" / "hand_landmarker.task")

            if not Path(model_path).exists():
                log.warning(f"Hand landmarker model not found at {model_path}. Gesture detection disabled.")
                self._mp_hands = None
                return

            # Create BaseOptions and HandLandmarkerOptions
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=2,
                min_hand_detection_confidence=0.7,
                min_hand_presence_confidence=0.7,
                min_tracking_confidence=0.5,
                running_mode=vision.RunningMode.IMAGE   # We use IMAGE mode for simplicity in your loop
            )

            self._mp_hands = vision.HandLandmarker.create_from_options(options)
            self._mp_drawing = mp.solutions.drawing_utils   # still works for drawing if you want

            # Optional: Pose (if you want to use it later)
            # pose_options = vision.PoseLandmarkerOptions(...)
            # self._mp_pose = vision.PoseLandmarker.create_from_options(...)

            log.info("✅ MediaPipe Tasks (HandLandmarker) loaded successfully")

        except ImportError as e:
            log.warning(f"MediaPipe not installed or import failed: {e}")
            self._mp_hands = None
        except Exception as e:
            log.error(f"Failed to initialize MediaPipe HandLandmarker: {e}")
            self._mp_hands = None
    # ─── Start / Stop ─────────────────────────────────────────────────────────

    def start(self):
        """Start camera in background thread."""
        if not config.CAMERA_ENABLED:
            log.info("Camera disabled in config")
            return

        self._cap = cv2.VideoCapture(config.CAMERA_INDEX)
        if not self._cap.isOpened():
            log.warning(f"Camera {config.CAMERA_INDEX} not available")
            return

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        self._cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        log.info(f"Camera started ({config.CAMERA_WIDTH}x{config.CAMERA_HEIGHT})")

    def stop(self):
        self._running = False
        if self._cap:
            self._cap.release()
        log.info("Camera stopped")

    # ─── Main Loop ────────────────────────────────────────────────────────────

    def _run_loop(self):
        """Main camera processing loop."""
        frame_interval = 1.0 / config.CAMERA_FPS
        last_frame_time = 0.0

        while self._running:
            now = time.time()
            if now - last_frame_time < frame_interval:
                time.sleep(0.01)
                continue
            last_frame_time = now

            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            # Store frame for on-demand capture
            self._frame_buffer.append(frame.copy())

            # Face detection (every frame)
            self._process_faces(frame, now)

            # Emotion update (less frequent)
            if now - self._last_emotion_update > config.EMOTION_UPDATE_SEC:
                self._update_emotion(frame, now)
                self._last_emotion_update = now

            # Gesture detection
            if config.GESTURE_DETECTION and now - self._last_gesture_check > 0.5:
                self._check_gestures(frame, now)
                self._last_gesture_check = now

    # ─── Face Detection ───────────────────────────────────────────────────────

    def _process_faces(self, frame: np.ndarray, now: float):
        """Detect faces and update presence state."""
        if self._face_cascade is None or self._face_cascade.empty():
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = self._face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        face_count = len(faces)
        is_present = face_count > 0

        with self._lock:
            self.face_state.count = face_count
            self.face_state.present = is_present
            if is_present:
                self.face_state.last_seen = now
                if len(faces) > 0:
                    self.face_state.bbox = faces[0].tolist()

                # Estimate gaze/attention via eye detection in face region
                if len(faces) > 0:
                    self._estimate_attention(frame, gray, faces[0], now)

        # Presence change events
        if is_present != self._prev_face_present:
            cooldown_ok = (now - self._last_presence_event) > config.PRESENCE_COOLDOWN_SEC
            if cooldown_ok:
                self._last_presence_event = now
                event = PresenceEvent(
                    arrived=is_present,
                    timestamp=now,
                    face_count=face_count,
                )
                if self.on_presence_change:
                    threading.Thread(
                        target=self.on_presence_change,
                        args=(event,),
                        daemon=True
                    ).start()

        self._prev_face_present = is_present

    def _estimate_attention(self, frame: np.ndarray, gray: np.ndarray,
                            face_rect, now: float):
        """Estimate if user is looking at screen (vs. away)."""
        if self._eye_cascade is None:
            return

        x, y, w, h = face_rect
        face_gray = gray[y:y+h, x:x+w]
        eyes = self._eye_cascade.detectMultiScale(face_gray, scaleFactor=1.1,
                                                   minNeighbors=4, minSize=(20, 20))

        was_attentive = self.face_state.attention_score > 0.5
        eye_count = len(eyes)

        # Simple heuristic: both eyes visible → looking at screen
        if eye_count >= 2:
            self.face_state.attention_score = 1.0
            self.face_state.gaze_direction = "forward"
        elif eye_count == 1:
            self.face_state.attention_score = 0.6
            self.face_state.gaze_direction = "partial"
        else:
            self.face_state.attention_score = 0.2
            self.face_state.gaze_direction = "away"

        is_attentive = self.face_state.attention_score > 0.5
        if was_attentive and not is_attentive:
            if self.on_attention_lost:
                threading.Thread(target=self.on_attention_lost, daemon=True).start()
        elif not was_attentive and is_attentive:
            if self.on_attention_gained:
                threading.Thread(target=self.on_attention_gained, daemon=True).start()

    # ─── Emotion Detection ────────────────────────────────────────────────────

    def _update_emotion(self, frame: np.ndarray, now: float):
        """Update emotion detection."""
        if not self.face_state.present:
            return

        emotion = "neutral"
        confidence = 0.5

        if self._deepface_available:
            try:
                # DeepFace runs in a thread to not block
                result = self._deepface.analyze(
                    frame,
                    actions=["emotion"],
                    enforce_detection=False,
                    silent=True,
                )
                if result and isinstance(result, list):
                    r = result[0]
                    dominant = r.get("dominant_emotion", "neutral")
                    emotions = r.get("emotion", {})
                    emotion = dominant
                    confidence = emotions.get(dominant, 50) / 100.0
            except Exception as e:
                log.debug(f"DeepFace error: {e}")
        else:
            # Fallback: brightness/contrast proxy (very rough)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray)
            # This is a placeholder — real emotion needs DeepFace
            emotion = "neutral"
            confidence = 0.5

        prev_emotion = self.face_state.emotion
        with self._lock:
            self.face_state.emotion = emotion
            self.face_state.emotion_confidence = confidence
            self._emotion_history.append((now, emotion, confidence))

        if emotion != prev_emotion and confidence > 0.6:
            log.debug(f"Emotion changed: {prev_emotion} → {emotion} ({confidence:.2f})")
            if self.on_emotion_change:
                threading.Thread(
                    target=self.on_emotion_change,
                    args=(emotion, confidence),
                    daemon=True
                ).start()

    def get_dominant_emotion(self, window_sec: float = 30.0) -> str:
        """Get the dominant emotion over the last N seconds."""
        now = time.time()
        recent = [e for t, e, c in self._emotion_history
                  if now - t < window_sec and c > 0.5]
        if not recent:
            return "neutral"
        from collections import Counter
        return Counter(recent).most_common(1)[0][0]

    # ─── Gesture Detection ────────────────────────────────────────────────────

    def _check_gestures(self, frame: np.ndarray, now: float):
        """Detect gestures using new MediaPipe Tasks HandLandmarker."""
        if self._mp_hands is None:
            return
        
        import mediapipe as mp  # still needed for drawing utils

        try:
            # Convert to MediaPipe Image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Run detection
            detection_result = self._mp_hands.detect(mp_image)

            if not detection_result.hand_landmarks:
                return

            for hand_landmarks in detection_result.hand_landmarks:
                gesture = self._classify_hand_gesture(hand_landmarks)
                if gesture:
                    event = GestureEvent(
                        gesture=gesture,
                        confidence=0.75,
                        timestamp=now,
                    )
                    log.info(f"Gesture detected: {gesture}")
                    if self.on_gesture:
                        threading.Thread(
                            target=self.on_gesture,
                            args=(event,),
                            daemon=True
                        ).start()

                    # Cooldown
                    self._last_gesture_check = now + self._gesture_cooldown

        except Exception as e:
            log.debug(f"Gesture detection error: {e}")

    def _classify_hand_gesture(self, landmarks_list) -> Optional[str]:
        """Classify hand gesture from MediaPipe Tasks landmarks (list of NormalizedLandmark)."""
        if not landmarks_list:
            return None

        # landmarks_list is a list of 21 landmarks
        lm = landmarks_list  # each item has .x and .y

        # Helper: is finger extended?
        def finger_up(tip_idx, pip_idx):
            return lm[tip_idx].y < lm[pip_idx].y

        thumb_up  = lm[4].x < lm[3].x   # rough thumb direction
        index_up  = finger_up(8, 6)
        middle_up = finger_up(12, 10)
        ring_up   = finger_up(16, 14)
        pinky_up  = finger_up(20, 18)

        fingers_up = sum([index_up, middle_up, ring_up, pinky_up])

        # Raised hand (open palm) — attention gesture → wake Nova
        if fingers_up >= 4:
            return "raise_hand"

        # Thumbs up — confirm / yes
        if thumb_up and fingers_up == 0:
            return "thumbs_up"

        # Point (index only)
        if index_up and not middle_up and not ring_up and not pinky_up:
            return "point"

        # Wave = all fingers up + wrist movement (simplified: just all fingers)
        if fingers_up == 4 and index_up:
            return "wave"

        return None

    # ─── Screenshot / Capture ────────────────────────────────────────────────

    def capture_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame."""
        if self._frame_buffer:
            return self._frame_buffer[-1].copy()
        return None

    def capture_base64(self, max_width: int = 640) -> Optional[str]:
        """Capture current frame as base64 JPEG."""
        frame = self.capture_frame()
        if frame is None:
            return None

        # Resize if needed
        h, w = frame.shape[:2]
        if w > max_width:
            scale = max_width / w
            frame = cv2.resize(frame, (max_width, int(h * scale)))

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf.tobytes()).decode()

    def save_frame(self, path: Optional[Path] = None) -> Optional[Path]:
        """Save current frame to disk."""
        frame = self.capture_frame()
        if frame is None:
            return None
        if path is None:
            path = config.VISION_DIR / f"frame_{int(time.time())}.jpg"
        cv2.imwrite(str(path), frame)
        return path

    # ─── Status ───────────────────────────────────────────────────────────────

    @property
    def is_available(self) -> bool:
        return self._running and self._cap is not None

    @property
    def is_person_present(self) -> bool:
        return self.face_state.present

    def get_status_dict(self) -> dict:
        """Full status for logging/display."""
        with self._lock:
            return {
                "present": self.face_state.present,
                "face_count": self.face_state.count,
                "emotion": self.face_state.emotion,
                "emotion_confidence": round(self.face_state.emotion_confidence, 2),
                "attention": round(self.face_state.attention_score, 2),
                "gaze": self.face_state.gaze_direction,
                "deepface": self._deepface_available,
                "mediapipe": self._mp_hands is not None,
            }