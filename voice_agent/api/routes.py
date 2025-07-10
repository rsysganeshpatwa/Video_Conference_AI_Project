from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from agent.core import AIAgent
from agent.token_fetcher import fetch_token_from_node
import os
from dotenv import load_dotenv
from fastapi import Request

load_dotenv()

router = APIRouter()
room_agent_map = {}  # room_name -> AIAgent
NODE_API_URL = os.getenv("NODE_API_URL")
LIVEKIT_URL = os.getenv("LIVEKIT_URL")


class StartAgentRequest(BaseModel):
    room_name: str
    participant_identity: str
    
class StopAgentRequest(BaseModel):
    room_name: str
    participant_identity: str


@router.post("/start-agent")
async def start_agent(req: StartAgentRequest, request: Request):
    room_name = req.room_name
    participant_identity = req.participant_identity

   
    if room_name in room_agent_map:
        agent = room_agent_map[room_name]
        agent.add_target_participant(participant_identity)
        return {"message": f"Agent already running in '{room_name}'. Now listening to: {participant_identity}"}
    
    token = fetch_token_from_node(NODE_API_URL, "AI-Agent", room_name)
    if not token:
        raise HTTPException(status_code=500, detail="Failed to fetch token from node API.")
    
    
    managers = request.app.state.managers
    agent = AIAgent(LIVEKIT_URL, token, llm=managers["llm"],
        stt=managers["stt"],
        tts=managers["tts"],room_name=room_name)
    agent.add_target_participant(participant_identity)
    await agent.connect()
    room_agent_map[room_name] = agent

    return {"message": f"Agent started for room '{room_name}', listening to '{participant_identity}'."}


@router.post("/stop-agent")
async def stop_agent(req: StopAgentRequest):
    room_name = req.room_name
    identity = req.participant_identity

    if not room_name:
        raise HTTPException(status_code=400, detail="Room name is required.")

    if room_name not in room_agent_map:
        raise HTTPException(status_code=400, detail=f"No agent running in room '{room_name}'")

    agent = room_agent_map[room_name]

    # Remove identity from agent via its method
    agent.remove_target_participant(identity)

    # Check if there are any remaining target participants
    if not agent.get_target_participants():
        print(f"❌ No target participants left. Disconnecting agent from room '{room_name}'")
        await agent.reset(room_name=room_name)
        del room_agent_map[room_name]
        return {"message": f"Agent in room '{room_name}' stopped (no targets remaining)."}

    return {
        "message": f"✅ '{identity}' removed. Remaining targets in room '{room_name}': {agent.get_target_participants()}"
    }
