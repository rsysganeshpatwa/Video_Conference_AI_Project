from fastapi import FastAPI,Depends
from pydantic import BaseModel
from api import fetch_token_from_node
from meeting_session import MeetingSession  # The new class
from fastapi.middleware.cors import CORSMiddleware
from context import get_mom_manager, get_transcription_manager,init_all_managers
from model_managers.mom_manager import MoMManager
from model_managers.transcription_manager import TranscriptionManager
# from model_managers.emotion_manager import EmotionRecognizer
import os



app = FastAPI()




app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    print("🔧 Initializing model managers at startup...")
    init_all_managers()
    
    
SESSIONS = {}

class StartRequest(BaseModel):
    room_name: str

class EmotionRequest(BaseModel):
    room_name: str
    participant_identity: str

@app.post("/start")
async def start_meeting(
    req: StartRequest,
    mom_manager: MoMManager = Depends(get_mom_manager),
    transcription_manager: TranscriptionManager = Depends(get_transcription_manager),
    # emotion_manager: EmotionRecognizer = Depends(get_emotion_manager)
):
    if req.room_name in SESSIONS:
        return {"status": "already running"}

    session = MeetingSession(req.room_name, mom_manager, transcription_manager, None)
    SESSIONS[req.room_name] = session

    token = fetch_token_from_node(os.getenv("NODE_API_URL"), "mom-bot", req.room_name)
    await session.start(os.getenv("LIVEKIT_URL"), token)

    return {"status": "started"}

@app.post("/stop")
async def stop_meeting(req: StartRequest):
    session = SESSIONS.pop(req.room_name, None)
    if not session:
        return {"error": "session not found"}
    mom = await session.stop()
    return {"mom": mom or "No MoM generated"}

@app.post("/start-emotion")
async def start_emotion_tracking(req: EmotionRequest):
    """Start emotion tracking for a specific participant in a running room"""
    session = SESSIONS.get(req.room_name)
    if not session:
        return {"error": "session not found"}
    
    if not session.running:
        return {"error": "session not running"}
    
    # Add participant to emotion tracking whitelist
    if not hasattr(session, 'emotion_enabled_participants'):
        session.emotion_enabled_participants = set()
    
    session.emotion_enabled_participants.add(req.participant_identity)
    
    return {
        "status": "emotion tracking started",
        "room_name": req.room_name,
        "participant": req.participant_identity,
        "enabled_participants": list(session.emotion_enabled_participants)
    }

@app.post("/stop-emotion")
async def stop_emotion_tracking(req: EmotionRequest):
    """Stop emotion tracking for a specific participant"""
    session = SESSIONS.get(req.room_name)
    if not session:
        return {"error": "session not found"}
    
    if hasattr(session, 'emotion_enabled_participants'):
        session.emotion_enabled_participants.discard(req.participant_identity)
        
        return {
            "status": "emotion tracking stopped",
            "room_name": req.room_name,
            "participant": req.participant_identity,
            "enabled_participants": list(session.emotion_enabled_participants)
        }
    
    return {"status": "no emotion tracking was active"}

@app.get("/emotion-status/{room_name}")
async def get_emotion_status(room_name: str):
    """Get current emotion tracking status for a room"""
    session = SESSIONS.get(room_name)
    if not session:
        return {"error": "session not found"}
    
    enabled_participants = []
    if hasattr(session, 'emotion_enabled_participants'):
        enabled_participants = list(session.emotion_enabled_participants)
    
    return {
        "room_name": room_name,
        "session_running": session.running,
        "emotion_enabled_participants": enabled_participants,
        "total_participants": len(session.participant_identity_map),
        "participant_identities": list(session.participant_identity_map.values())
    }
