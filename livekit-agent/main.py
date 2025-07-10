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
