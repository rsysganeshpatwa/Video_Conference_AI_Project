import os
import torch
import soundfile as sf
import torchaudio
import logging
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

# ✅ Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


class EmotionRecognizer:
    def __init__(
        self,
        model_id: str = "superb/wav2vec2-base-superb-er",
        room_name: str = "default_room"
    ):
        self.model_id = model_id
        self.room_name = room_name

        # try:
        #     logger.info("🔄 Loading feature extractor & model…")
        #     self.feature_extractor = None#Wav2Vec2FeatureExtractor.from_pretrained(model_id)
        #     self.model = None #Wav2Vec2ForSequenceClassification.from_pretrained(model_id)
        #     #self.model.eval()
        #     self.target_sr = self.feature_extractor.sampling_rate
        #     logger.info(f"✅ Emotion model loaded successfully with target sample rate: {self.target_sr}")
        # except Exception as e:
        #     logger.exception(f"❌ Failed to load emotion model '{model_id}': {e}")
        #     raise RuntimeError(f"❌ Failed to load emotion model '{model_id}': {e}")

    def recognize(self, audio_path: str, return_confidence: bool = False, min_confidence: float = 0.5):
        logger.info("🧠 Entered `recognize` method.")
        logger.info(f"🔍 Recognizing emotion from: {audio_path}")

        if not os.path.exists(audio_path):
            logger.warning(f"⚠️ Audio file does not exist: {audio_path}")
            return ("unknown", 0.0) if return_confidence else "unknown"

        try:
            # 1️⃣ Load audio
            waveform_np, sr = sf.read(audio_path)
            if waveform_np.ndim > 1:
                waveform_np = waveform_np.mean(axis=1)

            waveform_np = np.array(waveform_np, dtype=np.float32)
            waveform = torch.from_numpy(waveform_np).unsqueeze(0)

            # 2️⃣ Resample if needed
            if sr != self.target_sr:
                logger.info(f"🔁 Resampling from {sr} Hz to {self.target_sr} Hz")
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
                waveform = resampler(waveform)
                sr = self.target_sr

            # 3️⃣ Trim or pad
            max_duration = 10  # seconds
            max_samples = int(self.target_sr * max_duration)
            if waveform.shape[1] > max_samples:
                waveform = waveform[:, :max_samples]
            elif waveform.shape[1] < self.target_sr * 2:
                logger.warning("⚠️ Audio too short for reliable detection.")

            # 4️⃣ Feature extraction & inference
            inputs = self.feature_extractor(
                waveform.squeeze(),
                sampling_rate=sr,
                return_tensors="pt"
            )
            with torch.no_grad():
                logits = self.model(**inputs).logits
                pred_id = torch.argmax(logits, dim=-1).item()
                label = self.model.config.id2label[pred_id]
                confidence = torch.nn.functional.softmax(logits, dim=-1)[0][pred_id].item()

            logger.info(f"🎯 Detected Emotion: {label} (Confidence: {confidence:.2f})")

            if confidence < min_confidence:
                logger.warning(f"⚠️ Low confidence {confidence:.2f} for predicted emotion '{label}'")
                return ("uncertain", confidence) if return_confidence else "uncertain"

            return (label, confidence) if return_confidence else label

        except Exception as e:
            logger.exception(f"❌ Error during emotion recognition: {e}")
            return ("unknown", 0.0) if return_confidence else "unknown"


# ✅ CLI test
if __name__ == "__main__":
    er = EmotionRecognizer()
    audio_file = "output.wav"
    emotion = er.recognize(audio_file, return_confidence=True, min_confidence=0.5)
    print(f"Emotion: {emotion[0]}, Confidence: {emotion[1]:.2f}")

    print(f"🎯 Final Detected Emotion: {emotion}")
