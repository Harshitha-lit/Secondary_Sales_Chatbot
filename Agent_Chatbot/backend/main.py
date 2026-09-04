import os
import json
import logging
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
load_dotenv()

from agent import run_agent_loop
from sqlalchemy.orm import Session
import database
import models
import crud

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Agent Chatbot API")

# Create database tables if they don't exist
try:
    if database.engine:
        models.Base.metadata.create_all(bind=database.engine)
except Exception as e:
    logger.error(f"Could not initialize database: {e}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    messages: List[ChatMessage]
    
class Evidence(BaseModel):
    sources: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    provenance: str = "No provenance provided."

class ChartData(BaseModel):
    render_chart: bool = False
    type: str = "bar"
    data: List[Dict[str, Any]] = Field(default_factory=list)

class ChatResponse(BaseModel):
    session_id: Optional[str] = None
    text_answer: str = "Sorry, I couldn't generate a text answer."
    evidence: Evidence = Field(default_factory=Evidence)
    chart_data: Optional[ChartData] = Field(default_factory=ChartData)
    suggested_questions: List[str] = Field(default_factory=list)

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, db: Session = Depends(database.get_db)):
    try:
        # Get or create session
        session_id = request.session_id
        if not session_id:
            session = crud.create_chat_session(db)
            session_id = session.session_id
        else:
            session = crud.get_chat_session(db, session_id)
            if not session:
                session = crud.create_chat_session(db, session_id=session_id)
                
        # Save the latest user message
        if request.messages:
            last_msg = request.messages[-1]
            if last_msg.role == "user":
                crud.add_message(db, session_id, "user", last_msg.content)
                
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        structured_response = await run_agent_loop(messages)
        
        # Ensure we always return a dictionary that fits the schema even if LLM hallucinated
        if not isinstance(structured_response, dict):
            structured_response = {"text_answer": str(structured_response)}
        elif "text_answer" not in structured_response:
            # Check if there is an error key inside the hallucinated dict
            if "error" in structured_response:
                friendly_text = f"An error occurred while retrieving data: {structured_response['error']}"
            else:
                hallucinated_data = json.dumps(structured_response, indent=2)
                friendly_text = f"I processed your request, but the data could not be formatted properly. Here are the raw results:\n```json\n{hallucinated_data}\n```"
                
            structured_response = {
                "text_answer": friendly_text,
                "evidence": {"sources": ["System Check"], "confidence": 0.5, "provenance": "Data was retrieved but visual formatting failed."},
                "chart_data": {"render_chart": False, "type": "bar", "data": []},
                "suggested_questions": []
            }
            
        structured_response["session_id"] = session_id
        
        # Save AI response to DB
        crud.add_message(db, session_id, "assistant", structured_response)
            
        return structured_response
        
    except Exception as e:
        logger.exception("Error in chat endpoint")
        return {
            "session_id": getattr(request, "session_id", None),
            "text_answer": f"Backend Error: {str(e)}",
            "evidence": {"sources": [], "confidence": 0, "provenance": "Error occurred during execution."},
            "chart_data": {"render_chart": False, "type": "bar", "data": []},
            "suggested_questions": []
        }

@app.get("/chats")
def get_chats(skip: int = 0, limit: int = 50, db: Session = Depends(database.get_db)):
    sessions = crud.get_chat_sessions(db, skip=skip, limit=limit)
    return [{"session_id": s.session_id, "created_at": s.created_at, "updated_at": s.updated_at} for s in sessions]

@app.get("/chats/{session_id}")
def get_chat_history(session_id: str, db: Session = Depends(database.get_db)):
    session = crud.get_chat_session(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    messages = crud.get_messages_for_session(db, session_id)
    return {
        "session_id": session.session_id, 
        "messages": [{"role": m.role, "content": m.content, "created_at": m.created_at} for m in messages]
    }

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8001"))
    uvicorn.run(app, host=host, port=port)
