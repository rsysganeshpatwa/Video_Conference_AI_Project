import boto3
from collections import defaultdict, deque, Counter
import imutils
import cv2


class AWSRekognitionEmotionManager:
    def __init__(self, window_size=20, threshold=50.0, region='us-east-1'):
        """
        AWS Rekognition-based emotion recognizer with basic smoothing.
        """
        self.window_size = window_size
        self.threshold = threshold
        self.client = boto3.client('rekognition', region_name=region)
        self.emotion_history = defaultdict(lambda: deque(maxlen=self.window_size))

    def analyze(self, frame, participant_identity):
        """
        Analyze a frame and return the smoothed dominant emotion.
        """
        frame_resized = imutils.resize(frame, width=640)
        _, buffer = cv2.imencode(".jpg", frame_resized)
        image_bytes = buffer.tobytes()

        try:
            response = self.client.detect_faces(
                Image={"Bytes": image_bytes},
                Attributes=["ALL"]
            )

            for face in response.get("FaceDetails", []):
                emotions = face.get("Emotions", [])
                if not emotions:
                    continue

                dominant = max(emotions, key=lambda e: e["Confidence"])
                if dominant["Confidence"] >= self.threshold:
                    self.emotion_history[participant_identity].append(dominant["Type"])

            if self.emotion_history[participant_identity]:
                return Counter(self.emotion_history[participant_identity]).most_common(1)[0][0]

        except Exception as e:
            print(f"[AWSRekognitionEmotionManager] Error: {e}")

        return None
