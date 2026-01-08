"""
Database Repository Layer

Data access layer that provides clean interface to database operations.
Follows Repository pattern for separation of concerns.

Design Pattern: Repository
---------------------------
Separates business logic from data access logic.
Makes code testable (can mock repository in tests).

Features:
------------------------
- Single responsibility (one repository per entity)
- Clean interface (hide SQL details)
- Type hints (clear contracts)
- Error handling (proper exceptions)
- Performance (optimized queries)
"""

from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import desc, func
from datetime import datetime

from src.database.models import (
    ResearchSession,
    SearchResult,
    Fact,
    RiskFlag,
    Connection
)


class ResearchSessionRepository:
    """
    Repository for ResearchSession operations.
    
    Provides high-level interface for session management.
    """
    
    @staticmethod
    def create(db: Session, session_data: Dict[str, Any]) -> ResearchSession:
        """
        Create new research session.
        
        Args:
            db: Database session
            session_data: Session data (from ResearchState)
        
        Returns:
            Created ResearchSession
        """
        session = ResearchSession(**session_data)
        db.add(session)
        db.commit()
        db.refresh(session)
        return session
    
    @staticmethod
    def get_by_id(db: Session, session_id: str) -> Optional[ResearchSession]:
        """Get session by ID"""
        return db.query(ResearchSession).filter(
            ResearchSession.id == session_id,
            ResearchSession.deleted_at.is_(None)
        ).first()
    
    @staticmethod
    def get_all(db: Session, limit: int = 100) -> List[ResearchSession]:
        """Get all sessions (most recent first)"""
        return db.query(ResearchSession).filter(
            ResearchSession.deleted_at.is_(None)
        ).order_by(desc(ResearchSession.created_at)).limit(limit).all()
    
    @staticmethod
    def update_status(db: Session, session_id: str, status: str) -> None:
        """Update session status"""
        db.query(ResearchSession).filter(
            ResearchSession.id == session_id
        ).update({"status": status, "updated_at": datetime.utcnow()})
        db.commit()
    
    @staticmethod
    def soft_delete(db: Session, session_id: str) -> None:
        """Soft delete session (set deleted_at timestamp)"""
        db.query(ResearchSession).filter(
            ResearchSession.id == session_id
        ).update({"deleted_at": datetime.utcnow()})
        db.commit()


class FactRepository:
    """Repository for Fact operations"""
    
    @staticmethod
    def create(db: Session, fact_data: Dict[str, Any]) -> Fact:
        """Create new fact"""
        fact = Fact(**fact_data)
        db.add(fact)
        db.commit()
        db.refresh(fact)
        return fact
    
    @staticmethod
    def get_by_session(db: Session, session_id: str) -> List[Fact]:
        """Get all facts for a session"""
        return db.query(Fact).filter(
            Fact.session_id == session_id
        ).order_by(desc(Fact.confidence_score)).all()
    
    @staticmethod
    def get_by_category(db: Session, session_id: str, category: str) -> List[Fact]:
        """Get facts by category"""
        return db.query(Fact).filter(
            Fact.session_id == session_id,
            Fact.category == category
        ).all()
    
    @staticmethod
    def count_by_session(db: Session, session_id: str) -> int:
        """Count facts in session"""
        return db.query(func.count(Fact.id)).filter(
            Fact.session_id == session_id
        ).scalar()


class RiskFlagRepository:
    """Repository for RiskFlag operations"""
    
    @staticmethod
    def create(db: Session, risk_data: Dict[str, Any]) -> RiskFlag:
        """Create new risk flag"""
        risk = RiskFlag(**risk_data)
        db.add(risk)
        db.commit()
        db.refresh(risk)
        return risk
    
    @staticmethod
    def get_by_session(db: Session, session_id: str) -> List[RiskFlag]:
        """Get all risks for a session"""
        return db.query(RiskFlag).filter(
            RiskFlag.session_id == session_id
        ).order_by(desc(RiskFlag.impact_score)).all()
    
    @staticmethod
    def get_by_severity(db: Session, session_id: str, severity: str) -> List[RiskFlag]:
        """Get risks by severity level"""
        return db.query(RiskFlag).filter(
            RiskFlag.session_id == session_id,
            RiskFlag.severity == severity
        ).all()


class ConnectionRepository:
    """Repository for Connection operations"""
    
    @staticmethod
    def create(db: Session, connection_data: Dict[str, Any]) -> Connection:
        """Create new connection"""
        connection = Connection(**connection_data)
        db.add(connection)
        db.commit()
        db.refresh(connection)
        return connection
    
    @staticmethod
    def get_by_session(db: Session, session_id: str) -> List[Connection]:
        """Get all connections for a session"""
        return db.query(Connection).filter(
            Connection.session_id == session_id
        ).order_by(desc(Connection.strength)).all()
    
    @staticmethod
    def get_by_entity(db: Session, session_id: str, entity_name: str) -> List[Connection]:
        """Get all connections involving an entity"""
        return db.query(Connection).filter(
            Connection.session_id == session_id,
            (Connection.entity_1 == entity_name) | (Connection.entity_2 == entity_name)
        ).all()


# Export all repositories
__all__ = [
    "ResearchSessionRepository",
    "FactRepository",
    "RiskFlagRepository",
    "ConnectionRepository"
]