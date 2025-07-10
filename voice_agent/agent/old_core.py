# agent/core.py
import asyncio
import numpy as np
from scipy.io.wavfile import write
from livekit import rtc
from utils.audio_streamer import stream_audio_file
from utils.audio_utils import noise_reduction
from utils.vad import VADProcessor
import os
import json
import shutil

SAMPLE_RATE = 48000
NUMBER_CHANNELS = 1
AUDIO_INPUT_PATH = "input.wav"
TTS_OUTPUT_PATH = "output.wav"




class AIAgent:
    def __init__(self, url, token, llm, stt=None, tts=None, room_name=None):
    
        
        self.url = url
        self.token = token
        self.llm = llm
        self.stt = stt
        self.tts = tts
        
        self.audio_source = rtc.AudioSource(SAMPLE_RATE, NUMBER_CHANNELS)
        self.audio_track = rtc.LocalAudioTrack.create_audio_track("ai-response", self.audio_source)
        self.room = None
        self.room_name = room_name or "default_room"
        self.participant_audio_map = {}
        self.target_identities = set()  # 🔑 store multiple identities
        self.current_interaction_task = None
        self.tts_streaming_task = None
        self.processing_audio = False
        self.vad_processor = VADProcessor()
        self.participant_buffer = {}  # Buffer for each participant's audio
        self.speech_chunk_buffer = {}
        self.silence_counter = {}
        
    def session_dir(self):
        return os.path.join("voice_sessions", self.room_name)

    def session_file(self, filename):
        return os.path.join(self.session_dir(), filename)

    async def connect(self):
        if os.path.exists(self.session_dir()):
         shutil.rmtree(self.session_dir())
         print(f"🗑️ Cleared previous session data for room: {self.room_name}")

        os.makedirs(self.session_dir(), exist_ok=True)
        self.room = rtc.Room()  
        await self.room.connect(self.url, self.token)
        await self.room.local_participant.publish_track(self.audio_track)
        self.room.on("track_subscribed", self._on_track_subscribed)
        self.room.on("track_unsubscribed", self._on_track_unsubscribed)
        self.room.on("participant_left", self._on_participant_left)
        
        print("✅ Agent connected.")
            # 🔍 Manually subscribe to already-present participants
        self.subscribe_to_tracks();
        
    async def send_agent_status(self,room: rtc.Room, participant_identity: str, status: str):
        message = {
            "type": "voice_agent_status",
            "identity": participant_identity,  # Your agent's identity
            "status": status  # thinking, speaking,searching
        }

        await room.local_participant.publish_data(
            payload=json.dumps(message),       # Can be str or bytes
            reliable=True,                     # RELIABLE delivery (default)
                 # or ["participant1", ...] if targeting specific ones
            topic="voice_agent_status"                    # optional topic filter
        )

        
        
        
    def subscribe_to_tracks(self):
     if self.room and self.room.remote_participants:
        for identity, participant in self.room.remote_participants.items():
            if identity in self.target_identities:
                for tid, publication in participant.track_publications.items():
                    if publication.kind == rtc.TrackKind.KIND_AUDIO:
                        # This will simulate track_subscribed
                        self._on_track_subscribed(
                            publication.track, publication, participant
                        )
        
    def add_target_participant(self, identity):
        if identity not in self.target_identities:
            self.target_identities.add(identity)
            self.subscribe_to_tracks()  # Subscribe to tracks of this participant 
            print(f"🔍 Now listening to: {identity}")
        else:
            print(f"⚠️ Already listening to: {identity}")
            
    def get_target_participants(self):
        return list(self.target_identities)
     
    def remove_target_participant(self, identity):
        if identity in self.target_identities:
            self.target_identities.remove(identity)
            print(f"❌ Stopped listening to: {identity}")
            # Unsubscribe from tracks of this participant
            for participant in self.room.remote_participants.values():
                if participant.identity == identity:
                    for publication in participant.track_publications.values():
                        if publication.kind == rtc.TrackKind.KIND_AUDIO:
                            self._on_track_unsubscribed(
                                publication.track, publication, participant
                            )
            
            
                            
        
                            
                                    
        else:
            print(f"⚠️ Not listening to: {identity}")

    def _on_track_subscribed(self, track, publication, participant):
        if participant.identity not in self.target_identities:
            print(f"❌ Ignoring track from {participant.identity} (not a target participant)")
            return
        print(f"🎧 Listening to {participant.identity}")
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            stream = rtc.AudioStream(track)
            self.participant_audio_map[participant.sid] = []
            asyncio.create_task(self.receive_audio(stream, participant.sid,participant.identity))
            
    def _on_track_unsubscribed(self, track, publication, participant):
        if participant.sid in self.participant_audio_map:
            del self.participant_audio_map[participant.sid]
            print(f"❌ Stopped listening to {participant.identity}")
        if participant.identity in self.target_identities:
            print(f"❌ Participant {participant.identity} unsubscribed, stopping audio processing.")
            # If this participant was a target, stop interaction
            if self.current_interaction_task and not self.current_interaction_task.done():
                self.current_interaction_task.cancel()
                print(f"❌ Interaction task cancelled for {participant.identity}")
            # Remove from target identities
            self.remove_target_participant(participant.identity)
    
    def _on_participant_left(self, participant):
        if participant.sid in self.participant_audio_map:
            del self.participant_audio_map[participant.sid]
            print(f"❌ Participant {participant.identity} left, stopped listening.")
            # check if target participant complete empty
            if not any(p.identity in self.target_identities for p in self.room.remote_participants.values()):
                print("❌ No target participants left, stopping interaction.")
                self.stop_interaction(participant)
                # call reset to clear state
                asyncio.create_task(self.reset(self.room_name))
           
                
        if self.current_interaction_task and not self.current_interaction_task.done():
            self.current_interaction_task.cancel()
            print(f"❌ Interaction task cancelled for {participant.identity}")

    async def receive_audio(self, stream, sid,participant_identity):
        self.participant_buffer.setdefault(sid, [])
        self.speech_chunk_buffer.setdefault(sid, [])
        self.silence_counter.setdefault(sid, 0)

        async for event in stream:
            pcm = np.frombuffer(event.frame.data, dtype=np.int16)
            self.participant_buffer[sid].append(pcm)
            
            
            # check if participant is still in target identities
            if participant_identity not in self.target_identities:
                print(f"❌ Participant {participant_identity} is no longer a target, stopping audio processing.")
                break
            
            
           
           
            buffered_audio = np.concatenate(self.participant_buffer[sid])

            # Wait until we have 0.5s worth of audio
            if len(buffered_audio) < 24000:
                continue
            
          
            is_voice = self.vad_processor.is_human_voice(buffered_audio)
           

            #print(f"🔊 Audio from {sid}, human voice: {is_voice}")
            
            

            if is_voice:
                self.silence_counter[sid] = 0
                self.speech_chunk_buffer[sid].append(buffered_audio)
                if self.tts_streaming_task and not self.tts_streaming_task.done():
                 print("🔇 Interrupting agent response due to new speech")
                 self.stop_playback()
                await self.send_agent_status(self.room, self.room.local_participant.identity, "listening")
                print(f'${participant_identity} is speaking')

            else:
                self.silence_counter[sid] += 1

            # If we've had at least 3 speech chunks and 2 silence chunks, trigger interaction
            if len(self.speech_chunk_buffer[sid]) >=2 and self.silence_counter[sid] >= 3:
                
                local_identity = self.room.local_participant.identity
                print(f"Local participant identity: {local_identity}")
                
               
                if not self.processing_audio:
                    self.processing_audio = True

                    audio_np = np.concatenate(self.speech_chunk_buffer[sid])
                    self.speech_chunk_buffer[sid] = []
                    

                    self.current_interaction_task = asyncio.create_task(
                        self.run_interaction(audio_np, participant_identity)
                    )

            # Clear buffer
            self.participant_buffer[sid] = []
            
    def stop_playback(self):
     if self.tts_streaming_task and not self.tts_streaming_task.done():
        print("⏹️ Cancelling audio playback...")
        self.tts_streaming_task.cancel()

   

    async def run_interaction(self, audio_np, participant_identity):
        try:
            filtered_audio = noise_reduction(audio_np.astype(np.float32))
            session_input_path = self.session_file(AUDIO_INPUT_PATH)
            session_out_path = self.session_file(TTS_OUTPUT_PATH)
    
            
            write(session_input_path, SAMPLE_RATE, filtered_audio.astype(np.int16))
            
            if self.stt is None:
             raise RuntimeError("❌ STT manager is not initialized. Check context.py and startup init.")
           
            
            await self.send_agent_status(self.room, self.room.local_participant.identity, "thinking")
            
            # ✅ Transcription
            segments, _ = await asyncio.to_thread(self.stt.transcribe, session_input_path)
            full_text = #" ".join([seg.text for seg in segments]).strip()
            print(f'full_text${full_text}')

            if not full_text:
                return

            # ✅ LLM response
            
            # web search
            
            # get identity of  current participant
            local_identity = self.room.local_participant.identity
            response = await asyncio.to_thread(self.llm.get_llm_response, participant_identity,full_text, stream=False)
            # ✅ Split LLM output to chunks under 400 tokens
            chunks = self.tts.chunk_text(response)

            for idx, chunk in enumerate(chunks):
                output_file = f"{TTS_OUTPUT_PATH}_{idx}.wav"
                print(f"[TTS] Synthesizing chunk {idx+1}/{len(chunks)}")
                await asyncio.to_thread(self.tts.synthesize, chunk, save_as=session_out_path, out_path=self.session_dir())

                # ✅ Stream each output chunk one-by-one
                
                await self.send_agent_status(self.room, self.room.local_participant.identity, "speaking")
                
                self.tts_streaming_task = asyncio.create_task(
                    stream_audio_file(session_out_path, self.audio_source)
                )
                await self.tts_streaming_task
            
            await self.send_agent_status(self.room, self.room.local_participant.identity, "idle")
            # # ✅ TTS synthesis
            # await asyncio.to_thread(self.tts.synthesize, response, save_as=TTS_OUTPUT_PATH)

            # # ✅ Stream output
            # self.tts_streaming_task = asyncio.create_task(stream_audio_file(TTS_OUTPUT_PATH, self.audio_source))
            # await self.tts_streaming_task

        finally:
            self.processing_audio = False
            
    async def reset(self, room_name=None):
        """🧹 Clear conversation and reset to system prompt."""
        self.room_name = room_name or self.room_name
        self.participant_audio_map.clear()
        self.target_identities.clear()
        self.current_interaction_task = None
        self.tts_streaming_task = None
        self.processing_audio = False
        self.participant_buffer.clear()
        self.speech_chunk_buffer.clear()
        self.silence_counter.clear()
        if self.room:
            await self.room.disconnect()
            self.room = None
        if os.path.exists(self.session_dir()):
            shutil.rmtree(self.session_dir())
            print(f"🗑️ Cleared session data for room: {self.room_name}")
        self.room_name = None
        print("♻️ Agent state reset.")
