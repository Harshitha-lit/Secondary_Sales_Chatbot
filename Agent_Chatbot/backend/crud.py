from sqlalchemy.orm import Session
import models
import datetime

def create_chat_session(db: Session, session_id: str = None):
    db_session = models.ChatSession(session_id=session_id) if session_id else models.ChatSession()
    db.add(db_session)
    db.commit()
    db.refresh(db_session)
    return db_session

def get_chat_sessions(db: Session, skip: int = 0, limit: int = 50):
    return db.query(models.ChatSession).order_by(models.ChatSession.updated_at.desc()).offset(skip).limit(limit).all()

def get_chat_session(db: Session, session_id: str):
    return db.query(models.ChatSession).filter(models.ChatSession.session_id == session_id).first()

def add_message(db: Session, session_id: str, role: str, content):
    db_message = models.ChatMessage(session_id=session_id, role=role, content=content)
    db.add(db_message)
    
    # Update the session's updated_at timestamp
    db_session = get_chat_session(db, session_id)
    if db_session:
        db_session.updated_at = datetime.datetime.utcnow()
        
    db.commit()
    db.refresh(db_message)
    return db_message

def get_messages_for_session(db: Session, session_id: str):
    return db.query(models.ChatMessage).filter(models.ChatMessage.session_id == session_id).order_by(models.ChatMessage.created_at.asc()).all()
