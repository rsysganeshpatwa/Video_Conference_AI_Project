from fastapi import FastAPI, Request
from pydantic import BaseModel
from core.llm_manager import LLMManager
from core.stt_manager import STTManager
import uvicorn

app = FastAPI()

@app.on_event("startup")
async def on_startup():
    app.state.llm = LLMManager()
    app.state.stt = STTManager()

class TranscribeRequest(BaseModel):
    audio_path: str
    

@app.get("/")
async def root():
    return {"message": "Welcome to the AI Model Server"}

@app.post("/transcribe")
async def transcribe_audio(req: TranscribeRequest, request: Request):
    result = request.app.state.stt.transcribe(req.audio_path)
    return {"text": result}

class ChatRequest(BaseModel):
    user_query: str
    
class VoiceChatRequest(BaseModel):
    chat_history: list = []  # Optional, can be used for context in voice chat

@app.post("/chat")
async def chat(req: ChatRequest, request: Request):
    result = request.app.state.llm.get_llm_response(
        user_query=req.user_query,
     
    )
    return {"response": result}

@app.post("/voice_chat")
async def voice_chat(req: VoiceChatRequest, request: Request):
    result = request.app.state.llm.get_llm_response_chat(
        chat_history=req.chat_history,
    )
    return {"response": result}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000,reload=True)