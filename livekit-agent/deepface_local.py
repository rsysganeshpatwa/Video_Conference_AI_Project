from deepface import DeepFace
from collections import defaultdict, deque, Counter
import imutils
import cv2


class DeepEmotionRecognizer:
    def __init__(self, window_size=15, threshold=0.70, enforce_detection=False):
        """
        Real-time emotion recognizer with smoothing over a sliding window.
        :param window_size: Number of recent emotions to consider for smoothing.
        :param threshold: Minimum confidence required to accept emotion prediction.
        :param enforce_detection: Whether DeepFace should fail on no-face.
        """
        self.window_size = window_size
        self.threshold = threshold
        self.enforce_detection = enforce_detection
        self.emotion_history = defaultdict(lambda: deque(maxlen=self.window_size))
        self.check_device()

    def check_device(self):
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                print(f"[Device] ✅ TensorFlow GPU: {[gpu.name for gpu in gpus]}")
            else:
                print("[Device] ⚠️ TensorFlow using CPU")
        except ImportError:
            print("[Device] ℹ️ TensorFlow not installed")

        try:
            import torch
            if torch.cuda.is_available():
                print(f"[Device] ✅ PyTorch GPU: {torch.cuda.get_device_name(0)}")
            else:
                print("[Device] ⚠️ PyTorch using CPU")
        except ImportError:
            print("[Device] ℹ️ PyTorch not installed")

    def analyze(self, frame, participant_identity):
        """
        Analyze a frame and return a smoothed emotion label for the given participant.
        :param frame: Video frame (BGR, as from OpenCV)
        :param participant_identity: Unique ID for the participant
        :return: Dominant emotion (smoothed), or None
        """
        try:
            frame = imutils.resize(frame, width=600)

            results = DeepFace.analyze(
                frame,
                actions=['emotion'],
                enforce_detection=self.enforce_detection,
              #  detector_backend='retinaface',  # More stable; optional
            )

            # Normalize to list
            results = results if isinstance(results, list) else [results]
            if not results:
                return None

            # Pick the face with highest emotion confidence
            most_confident_face = max(
                results,
                key=lambda r: max(r['emotion'].values())
            )

            emotion_scores = most_confident_face.get('emotion', {})
            dominant_emotion = most_confident_face.get('dominant_emotion', None)

            if not dominant_emotion or dominant_emotion not in emotion_scores:
                return None

            confidence = emotion_scores[dominant_emotion]
            if confidence > 1:  # Normalize 0–100 scale
                confidence /= 100.0

            if confidence >= self.threshold:
                self.emotion_history[participant_identity].append(dominant_emotion)

            if self.emotion_history[participant_identity]:
                return Counter(self.emotion_history[participant_identity]).most_common(1)[0][0]
            else:
                return None

        except Exception as e:
            print(f"[EmotionAnalyzer] ❌ Error processing frame: {e}")
            return None
