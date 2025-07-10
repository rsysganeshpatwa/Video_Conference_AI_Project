from fer import FER
import cv2
import numpy as np
import os

class DeepEmotionRecognizer:
    def __init__(self, enhance_brightness=True, confidence_threshold=0.5, output_path="output.mp4"):
        self.detector = FER(mtcnn=True)
        self.enhance_brightness = enhance_brightness
        self.confidence_threshold = confidence_threshold
        self.output_path = output_path
        self.video_writer = None  # Will initialize once
        self.frame_size = (640, 480)
        self.fps = 20

    def preprocess(self, img):
        # Resize for consistency
        img = cv2.resize(img, self.frame_size)

        # # Enhance brightness/contrast if enabled
        # if self.enhance_brightness:
        #     img = cv2.convertScaleAbs(img, alpha=1.2, beta=20)

        # Convert to RGB
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return rgb

    def analyze(self, img):
        try:
            rgb = self.preprocess(img)
            frame = cv2.resize(img, self.frame_size)  # Original frame resized for annotation

            # # Initialize video writer once
            # if self.video_writer is None:
            #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            #     self.video_writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, self.frame_size)

            # Detect emotions
            results = self.detector.detect_emotions(rgb)

            if results:
                for face in results:
                    emotions = face["emotions"]
                    top_emotion, score = max(emotions.items(), key=lambda x: x[1])

                    if score >= self.confidence_threshold:
                        (x, y, w, h) = face["box"]
                        label = f"{top_emotion} ({score*100:.1f}%)"
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # ✅ Save the annotated frame to video
            #self.video_writer.write(frame)

            # Return emotion name (or None if no confident detection)
            # default to the first detected emotion
         
            if results:
                emotions = results[0]["emotions"]
                top_emotion, score = max(emotions.items(), key=lambda x: x[1])
                if score >= self.confidence_threshold:
                    return top_emotion
            return 'neutral'  # Default if no confident detection

        except Exception as e:
            print(f"❌ Emotion detection error: {e}")
            return None

    def release(self):
        if self.video_writer is not None:
            self.video_writer.release()
