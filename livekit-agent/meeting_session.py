import asyncio
import glob
import os
import wave
import contextlib
import numpy as np
from scipy.signal import resample
from collections import defaultdict
import livekit.rtc as rtc
from model_managers.mom_manager import MoMManager
from model_managers.transcription_manager import TranscriptionManager
# from model_managers.emotion_manager import EmotionRecognizer
from emotion_helper import save_emotion_to_db, clear_room_emotions
import soundfile as sf
import shutil
import traceback
from deepface_local import DeepEmotionRecognizer
import cv2
#json
import json





class MeetingSession:
    def __init__(self, room_name, mom_manager: MoMManager, transcription_manager: TranscriptionManager, emotion_manager: None):
        self.room_name = room_name
       
        self.emotion_manager = None #EmotionRecognizer()
        self.room = None
        self.mom_manager = mom_manager
        self.transcription_manager = transcription_manager
        self.audio_buffers = []
        self.participant_audio_map = defaultdict(list)
        self.participant_identity_map = {}
        self.emotion_task = None
        self.running = False
        self.emotion_recognizer = DeepEmotionRecognizer()

    def session_dir(self):
        return os.path.join("livekit_sessions", self.room_name)

    def session_file(self, filename):
        return os.path.join(self.session_dir(), filename)

    def ensure_speakers_dir(self):
        os.makedirs(self.speakers_dir(), exist_ok=True)

    def speakers_dir(self):
        return os.path.join("sessions", "emotion", self.room_name, "speakers")

    def list_speakers(self):
        path = self.speakers_dir()
        return os.listdir(path) if os.path.exists(path) else []

    def speaker_audio(self, speaker_id):
        return os.path.join(self.speakers_dir(), f"{speaker_id}.wav")

    async def track_emotions(self):
        print(f"🎯 Starting emotion tracking for {self.room_name}")
        self.ensure_speakers_dir()
        self.running = True

        while self.running:
            if not self.participant_audio_map:
                await asyncio.sleep(10)
                continue

            for sid, pcms in self.participant_audio_map.items():
                if not pcms:
                    continue

                identity = self.participant_identity_map.get(sid, f"user-{sid[:4]}")
                try:
                    all_data = np.concatenate(pcms)
                    last_10sec_samples = 480000
                    last_segment = all_data[-last_10sec_samples:] if len(all_data) > last_10sec_samples else all_data

                    output_path = self.speaker_audio(identity)
                    sf.write(output_path, last_segment, samplerate=48000)
                    print(f"💾 [Emotion] Saved 10s for speaker: {identity}")

                    label, confidence = self.emotion_manager.recognize(audio_path=output_path, return_confidence=True)
                    save_emotion_to_db(self.room_name, identity, label, confidence)
                    print(f"🎭 [Emotion] {identity}: {label} ({confidence:.2f})")

                except Exception as e:
                    print(f"⚠️ Emotion recognition failed for {identity}: {e}")
                    traceback.print_exc()
            await asyncio.sleep(10)

    async def receive_audio(self, stream, sid):
        async for event in stream:
            frame = event.frame
            pcm = np.frombuffer(frame.data, dtype=np.int16)
            if np.max(np.abs(pcm)) > 0:
                self.participant_audio_map[sid].append(pcm)
                self.audio_buffers.append(pcm)
        await stream.aclose()

    
    def convert_i420_to_bgr(self,yuv_data: memoryview, width: int, height: int) -> np.ndarray:
        # I420 (YUV420 planar) has 1.5 bytes per pixel
        frame_size = width * height
        uv_size = frame_size // 4

        # Extract Y, U, V planes
        y = np.frombuffer(yuv_data[:frame_size], dtype=np.uint8).reshape((height, width))
        u = np.frombuffer(yuv_data[frame_size:frame_size + uv_size], dtype=np.uint8).reshape((height // 2, width // 2))
        v = np.frombuffer(yuv_data[frame_size + uv_size:], dtype=np.uint8).reshape((height // 2, width // 2))

        # Resize U and V to match Y
        u_up = cv2.resize(u, (width, height), interpolation=cv2.INTER_LINEAR)
        v_up = cv2.resize(v, (width, height), interpolation=cv2.INTER_LINEAR)

        # Stack into YUV image
        yuv = cv2.merge([y, u_up, v_up])

        # Convert YUV to BGR
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        return bgr

    

    async def receive_video(self, video_stream, participant_sid):
        async for event in video_stream:
            try:
                frame = event.frame
                img = self.convert_i420_to_bgr(frame.data, frame.width, frame.height)


                emotion = self.emotion_recognizer.analyze(img, participant_sid)
                print(f"👤 {self.participant_identity_map.get(participant_sid, participant_sid)}: Detected emotion: {emotion}")
                

                await self.send_emotion_update(
                    self.room,
                    self.participant_identity_map.get(participant_sid),
                    emotion
                )
            except Exception as e:
                print(f"⚠️ DeepFace failed: {e}")


            
    async def send_emotion_update(self,room: rtc.Room, participant_identity: str, emotion: str):
        message = {
            "type": "emotion",
            "participant": participant_identity,
            "emotion": emotion
        }

        await room.local_participant.publish_data(
            payload=json.dumps(message),       # Can be str or bytes
            reliable=True,                     # RELIABLE delivery (default)
                 # or ["participant1", ...] if targeting specific ones
            topic="emotion"                    # optional topic filter
        )

        #print(f"📤 Sent emotion update: {emotion} for {participant_identity}")
        
    def _on_track_subscribed(self, track, publication, participant):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            stream = rtc.AudioStream(track)
            self.participant_identity_map[participant.sid] = participant.identity
            asyncio.create_task(self.receive_audio(stream, participant.sid))
            
        # elif track.kind == rtc.TrackKind.KIND_VIDEO:
        #     # Handle video subscription
        #     print(f"🎥 Video track subscribed from {participant.identity}")
        #     video_stream = rtc.VideoStream(track)
        #     self.participant_identity_map[participant.sid] = participant.identity
        #     asyncio.create_task(self.receive_video(video_stream, participant.sid))
            
            
    


    async def start(self, url, token):
        if os.path.exists(self.session_dir()):
            shutil.rmtree(self.session_dir())
        print(f"🗑️ Cleared previous session data for room: {self.room_name}")

        os.makedirs(self.session_dir(), exist_ok=True)
        self.room = rtc.Room()
        self.room.on("track_subscribed", self._on_track_subscribed)
        await self.room.connect(url, token)
        print(f"[{self.room_name}] Connected.")
        #self.emotion_task = asyncio.create_task(self.track_emotions())
        self.running = True

    async def stop(self):
        if not self.room:
            return None
        await self.room.disconnect()
        self.room = None

        self.running = False
        if self.emotion_task:
            self.emotion_task.cancel()
            #clear_room_emotions(self.room_name)
        print(f"🛑 Emotion tracker stopped.")

        if not self.audio_buffers:
            print(f'No Audio Saved')
            return None

        await self.save_audio()
        return await self.process_audio()

    async def save_audio(self):
        print('Saving Audio')
        data = np.concatenate(self.audio_buffers)
        with wave.open(self.session_file("audio.wav"), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(48000)
            wf.writeframes(data.tobytes())

    def save_individual_speakers(self):
        speakers_dir = os.path.join(self.session_dir(), "speakers")
        os.makedirs(speakers_dir, exist_ok=True)
        wav_files = []
        for sid, pcms in self.participant_audio_map.items():
            identity = self.participant_identity_map.get(sid, f"user-{sid[:4]}")
            file = os.path.join(speakers_dir, f"{identity}.wav")
            with wave.open(file, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(48000)
                wf.writeframes(np.concatenate(pcms).tobytes())
            wav_files.append((identity, file))
        return wav_files

    def transcribe(self, wav_files):
        return self.transcription_manager.transcribe(wav_files, output_dir=self.session_dir())

    def generate_mom(self, transcript_path):
        return self.mom_manager.generate_from_transcript(transcript_path)

    async def process_audio(self):
        wavs = self.save_individual_speakers()  # List of (identity, wav_path)
        durations = {}

        # ✅ Compute duration for each speaker's WAV
        for identity, wav_path in wavs:
            with contextlib.closing(wave.open(wav_path, 'r')) as f:
                frames = f.getnframes()
                rate = f.getframerate()
                duration_sec = frames / float(rate)
                minutes = int(duration_sec // 60)
                seconds = int(duration_sec % 60)
                durations[identity] = f"{minutes} min {seconds} sec"

        # ✅ Transcribe from speaker audio
        transcript = self.transcribe(wavs)  # Assumes transcribe takes dict {identity: path}
        if not transcript or not transcript.strip():
            return None

     
        # Save transcript
        with open(self.session_file("transcript.txt"), "w") as f:
            f.write(transcript.strip())

        # ✅ Generate MoM from enhanced transcript
        mom = self.generate_mom(self.session_file("transcript.txt"))
        
           # ✅ Prepend duration info
        speaker_info = "\n".join([f"{ identity.split('_')[0] if '_' in identity else identity}: {dur}" for identity, dur in durations.items()])
        full_mom = f"Speaker Durations:\n{speaker_info}\n\n{mom}"

        if full_mom:
            with open(self.session_file("mom.txt"), "w") as f:
                f.write(full_mom)

        return full_mom