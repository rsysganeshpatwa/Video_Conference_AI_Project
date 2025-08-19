from deepface import DeepFace
from collections import defaultdict, deque, Counter
import imutils
import cv2


class DeepEmotionRecognizer:
    def __init__(self, window_size=1, threshold=0.80):
        """
        Initialize emotion recognizer optimized for real-time response.
        """
        self.window_size = window_size  # Set to 1 for immediate response
        self.threshold = threshold  # Lowered for faster detection
        self.emotion_history = defaultdict(lambda: deque(maxlen=window_size))
        self.check_device()

    def check_device(self):
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                print(f"[Device] ✅ TensorFlow GPU: {[gpu.name for gpu in gpus]}")
            else:
                print("[Device] ❌ TensorFlow running on CPU")
        except ImportError:
            pass

        try:
            import torch
            if torch.cuda.is_available():
                print(f"[Device] ✅ PyTorch GPU: {torch.cuda.get_device_name(0)}")
            else:
                print("[Device] ❌ PyTorch running on CPU")
        except ImportError:
            pass

    def analyze(self, frame, participant_identity):
        """
        Analyze a frame from a participant and return stable dominant emotion.
        """
        try:
            frame = imutils.resize(frame, width=600)

            results = DeepFace.analyze(
                frame,
                actions=['emotion'],
                enforce_detection=False,
                #detector_backend='retinaface',  # More stable than default
            )

            # Handle both single and multiple face detection
            results = results if isinstance(results, list) else [results]
            if not results:
                return None

            # Get most confident face (or you can iterate over all)
            most_confident_face = max(results, key=lambda r: max(r['emotion'].values()))

            emotion = most_confident_face['dominant_emotion']
            confidence = most_confident_face['emotion'][emotion] / 100.0

            if confidence >= self.threshold:
                self.emotion_history[participant_identity].append(emotion)

            if self.emotion_history[participant_identity]:
                return Counter(self.emotion_history[participant_identity]).most_common(1)[0][0]
            else:
                return None

        except Exception as e:
            print(f"[EmotionAnalyzer] Error: {e}")
            return None
