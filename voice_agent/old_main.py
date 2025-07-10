import os
import asyncio
import numpy as np
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer
from scipy.io.wavfile import write
from scipy.signal import butter, lfilter
from utils.record import record_audio
from utils.llm import get_llm_response
from utils.open_voice_tts import synthesize_with_voice_clone
from utils.audio_streamer import stream_audio_file
from livekit import rtc
from dotenv import load_dotenv

load_dotenv()

SAMPLE_RATE = 48000
NUMBER_CHANNELS = 1
VOICE_CLONE_PATH = "models/voice_sample.wav"
AUDIO_INPUT_PATH = "input.wav"
TTS_OUTPUT_PATH = "output.wav"
WHISPER_MODEL = "medium"

participant_identity_map = {}
participant_audio_map = {}

def noise_reduction(audio):
    # Simple noise reduction using a high-pass filter
    def butter_highpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return b, a

    def highpass_filter(data, cutoff=100.0, fs=48000, order=5):
        b, a = butter_highpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y

    return highpass_filter(audio)

class AIAgent:
    def __init__(self):
        self.whisper = WhisperModel(WHISPER_MODEL)
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.room = None
        self.audio_source = rtc.AudioSource(SAMPLE_RATE, NUMBER_CHANNELS)
        self.audio_track = rtc.LocalAudioTrack.create_audio_track("ai-response", self.audio_source)
        self.current_interaction_task = None
        self.tts_streaming_task = None
        self.processing_audio = False

    async def connect_to_room(self, url, token):
        self.room = rtc.Room()
        await self.room.connect(url, token)
        await self.room.local_participant.publish_track(self.audio_track)
        self.room.on("track_subscribed", self._on_track_subscribed)
        print("✅ Connected to LiveKit room and published audio track.")
        # ✅ Manually subscribe to existing participants' tracks
          # 🔁 Process already-connected participants
        for identity, participant in self.room.remote_participants.items():
            print(f"👤 Existing participant: {identity} ({participant.sid})")
            for tid, publication in participant.track_publications.items():
                print(f"🔍 Track ID: {tid}")
                print(f"\tKind: {publication.kind}")
                print(f"\tName: {publication.name}")
                print(f"\tSource: {publication.source}")

                # ✅ Check if track is already subscribed and is audio
                if publication.track and publication.track.kind == rtc.TrackKind.KIND_AUDIO:
                    print(f"🔁 Manually subscribing to {participant.identity}'s audio track")
                    self._on_track_subscribed(publication.track, publication, participant)
                
    def _on_track_subscribed(self, track, publication, participant):
        print(f"🔊 Subscribed to {participant.identity}")
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            stream = rtc.AudioStream(track)
            participant_identity_map[participant.sid] = participant.identity
            participant_audio_map[participant.sid] = []
            asyncio.create_task(self.receive_audio(stream, participant.sid))

    async def receive_audio(self, stream, sid):
        print(f"🎧 Receiving audio from {participant_identity_map[sid]}")
        async for event in stream:
            frame = event.frame
            pcm = np.frombuffer(frame.data, dtype=np.int16)
            volume = np.max(np.abs(pcm))
        
            print(f"volume{volume}");
             
            if volume > 20000:
                if self.tts_streaming_task and not self.tts_streaming_task.done():
                    print("🛑 Interrupting TTS streaming...")
                    self.tts_streaming_task.cancel()
                    try:
                        await self.tts_streaming_task
                    except asyncio.CancelledError:
                        print("⚡ TTS stream cancelled.")

                participant_audio_map[sid].append(pcm)
            elif len(participant_audio_map[sid]) > 30 and not self.processing_audio:
                print(f"🧠 Triggering interaction for {participant_identity_map[sid]} with buffered audio.")
                self.processing_audio = True
                audio_np = np.concatenate(participant_audio_map[sid])
                participant_audio_map[sid] = []
                self.current_interaction_task = asyncio.create_task(self.run_interaction(audio_np))

    async def run_interaction(self, audio_np):
        try:
            print("🔧 Applying noise reduction...")
            filtered_audio = noise_reduction(audio_np.astype(np.float32))

            print("💾 Saving received audio to WAV file...")
            write(AUDIO_INPUT_PATH, SAMPLE_RATE, filtered_audio.astype(np.int16))

            print("🧠 Transcribing...")
            segments, _ = await asyncio.to_thread(self.whisper.transcribe, AUDIO_INPUT_PATH, language="en")
            full_text = " ".join([seg.text for seg in segments]).strip()
            print(f"🗣️ You said: {full_text}")

            if not full_text:
                print("❌ No speech detected. Retrying...")
                return

            print("💬 Generating response...")
            response = await asyncio.to_thread(get_llm_response, full_text)
            print(f"🤖 Response: {response}")

            print("🗣️ Synthesizing voice...")
            await asyncio.to_thread(synthesize_with_voice_clone, response, speaker="default", speed=0.9, save_as=TTS_OUTPUT_PATH)

            print("📡 Streaming audio into room...")
            self.tts_streaming_task = asyncio.create_task(stream_audio_file(TTS_OUTPUT_PATH, self.audio_source))
            await self.tts_streaming_task
            print("✅ Done.")
        finally:
            self.processing_audio = False
            self.tts_streaming_task = None

async def main():
    url = os.getenv("LIVEKIT_URL")
    token = os.getenv("LIVEKIT_TOKEN")

    if not url or not token:
        raise EnvironmentError("LIVEKIT_URL and LIVEKIT_TOKEN must be set in the environment.")

    agent = AIAgent()
    await agent.connect_to_room(url, token)

    while True:
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())