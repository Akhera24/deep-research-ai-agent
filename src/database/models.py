"""
Database Models for Research Agent

SQLAlchemy ORM models that map to PostgreSQL tables.
These models store all research data for persistence and analysis.

Design Decisions:
-----------------
1. JSONB fields: Flexible storage for dynamic data (facts, evidence)
2. Indexes: Optimized for common queries (session lookup, fact search)
3. Soft deletes: deleted_at field instead of actual deletion
4. Timestamps: created_at, updated_at for audit trail
5. Relationships: Proper foreign keys with cascade options
6. Type hints: All fields have explicit types

Features: 
------------------------
- Production-tested schema patterns
- Proper indexing for performance
- Data integrity with constraints
- Audit trail support
- Scalable design (handles millions of records)

-----------------
- Stores all research data (facts, risks, connections)
- Supports evaluation framework (queryable results)
- Enables reporting (structured data export)
- Provides audit trail (timestamps, soft deletes)
"""

from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, Float, DateTime,
    Text, Boolean, ForeignKey, Index, CheckConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
import uuid

# Base class for all models
Base = declarative_base()


class ResearchSession(Base):
    """
    Main research session record.
    
    One session = one complete research operation on a target entity.
    All facts, risks, connections link back to a session.
    
    Query Patterns:
    - Get all sessions: SELECT * FROM research_sessions ORDER BY created_at DESC
    - Get session by ID: SELECT * FROM research_sessions WHERE id = ?
    - Get active sessions: SELECT * FROM research_sessions WHERE status = 'running'
    """
    __tablename__ = "research_sessions"
    
    # Primary Key
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Input Data
    query = Column(Text, nullable=False, index=True)  # Original user query
    target_entity = Column(JSONB, nullable=False)  # Extracted entity info
    research_parameters = Column(JSONB, default={})  # User preferences
    
    # Status Tracking
    status = Column(String, default="pending", index=True)  # pending, running, completed, failed
    current_stage = Column(String)  # Current workflow stage
    
    # Results Summary
    total_facts = Column(Integer, default=0)
    total_risk_flags = Column(Integer, default=0)
    total_connections = Column(Integer, default=0)
    
    # Performance Metrics
    total_searches = Column(Integer, default=0)
    total_api_calls = Column(Integer, default=0)
    total_cost = Column(Float, default=0.0)
    duration_seconds = Column(Float)
    
    # Quality Metrics
    confidence_score = Column(Float)  # Overall confidence (0-1)
    coverage_score = Column(Float)  # Completeness score (0-1)
    
    # Timestamps (audit trail)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Soft delete
    deleted_at = Column(DateTime, nullable=True)
    
    # Relationships (enable JOIN queries)
    facts = relationship("Fact", back_populates="session", cascade="all, delete-orphan")
    searches = relationship("SearchResult", back_populates="session", cascade="all, delete-orphan")
    risk_flags = relationship("RiskFlag", back_populates="session", cascade="all, delete-orphan")
    connections = relationship("Connection", back_populates="session", cascade="all, delete-orphan")
    
    # Indexes for performance
    # GIN index enabled with pg_trgm for advanced text search
    __table_args__ = (
        # Composite index for common query pattern (status + timestamp)
        Index('ix_sessions_status_created', 'status', 'created_at'),
        
        # GIN trigram index for fuzzy text search on queries
        # Enables fast LIKE, similarity, and regex searches
        # Example: WHERE query % 'Sarah Chen' (similarity search)
        Index('ix_sessions_query', 'query', 
              postgresql_using='gin',
              postgresql_ops={'query': 'gin_trgm_ops'}),
    )
    
    def __repr__(self):
        return f"<ResearchSession(id='{self.id[:8]}...', query='{self.query[:30]}...', status='{self.status}')>"


class SearchResult(Base):
    """
    Individual search result from search engines.
    
    Stores both metadata and full content for analysis.
    Multiple searches per session (consecutive search strategy).
    """
    __tablename__ = "search_results"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, ForeignKey("research_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Search Query
    query = Column(Text, nullable=False, index=True)
    iteration = Column(Integer, default=0)  # Which search iteration
    
    # Result Data
    url = Column(Text, nullable=False)
    title = Column(Text)
    snippet = Column(Text)
    full_content = Column(Text)  # Extracted page content
    
    # Metadata
    source_reliability = Column(Float, default=0.5)  # 0-1 based on domain
    relevance_score = Column(Float)  # 0-1 from search engine
    search_engine = Column(String)  # brave, serper, etc.
    
    # Performance
    fetch_duration_ms = Column(Integer)
    content_length = Column(Integer)
    
    # Timestamps
    fetched_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    session = relationship("ResearchSession", back_populates="searches")
    
    __table_args__ = (
        Index('ix_search_session_iteration', 'session_id', 'iteration'),
        Index('ix_search_url', 'url'),
    )
    
    def __repr__(self):
        return f"<SearchResult(url='{self.url[:50]}...', query='{self.query[:30]}...')>"


class Fact(Base):
    """
    Extracted and verified fact.
    
    Core entity - represents discovered information about target.
    Multiple facts per session, categorized for analysis.
    """
    __tablename__ = "facts"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, ForeignKey("research_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Fact Content
    content = Column(Text, nullable=False)
    category = Column(String, nullable=False, index=True)  # biographical, professional, financial, legal, connections
    
    # Verification Status
    confidence_score = Column(Float, nullable=False, index=True)  # 0-1 confidence
    verification_status = Column(String, default="unverified")  # unverified, verified, conflicting
    
    # Source Information
    source_urls = Column(JSONB, default=list)  # List of URLs
    evidence = Column(JSONB, default=list)  # Supporting quotes/evidence
    cross_reference_count = Column(Integer, default=1)  # Number of sources
    
    # Metadata
    extracted_by = Column(String)  # Which agent/model extracted this
    conflicting = Column(Boolean, default=False)  # Conflicting information found
    
    # Timestamps
    extracted_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    verified_at = Column(DateTime)
    
    # Relationships
    session = relationship("ResearchSession", back_populates="facts")
    
    __table_args__ = (
        Index('ix_fact_session_category', 'session_id', 'category'),
        Index('ix_fact_confidence', 'confidence_score'),
        CheckConstraint('confidence_score >= 0 AND confidence_score <= 1', name='check_confidence_range'),
    )
    
    def __repr__(self):
        return f"<Fact(category='{self.category}', confidence={self.confidence_score:.2f}, content='{self.content[:40]}...')>"


class RiskFlag(Base):
    """
    Identified risk or red flag.
    
    Represents potential concerns discovered during research.
    Categorized by type and severity for prioritization.
    """
    __tablename__ = "risk_flags"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, ForeignKey("research_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Risk Information
    category = Column(String, nullable=False, index=True)  # financial, legal, reputational, professional
    description = Column(Text, nullable=False)
    severity = Column(String, nullable=False, index=True)  # low, medium, high, critical
    
    # Scoring
    confidence = Column(Float, nullable=False)  # 0-1 confidence in this risk
    impact_score = Column(Float, nullable=False)  # 0-10 potential impact
    
    # Evidence
    evidence = Column(JSONB, default=list)  # Fact IDs or evidence
    source_urls = Column(JSONB, default=list)  # Supporting URLs
    
    # Metadata
    detected_by = Column(String)  # Which agent detected this
    false_positive = Column(Boolean, default=False)  # Manual override
    
    # Timestamps
    detected_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    session = relationship("ResearchSession", back_populates="risk_flags")
    
    __table_args__ = (
        Index('ix_risk_session_severity', 'session_id', 'severity'),
        Index('ix_risk_category', 'category'),
        CheckConstraint('confidence >= 0 AND confidence <= 1', name='check_risk_confidence_range'),
        CheckConstraint('impact_score >= 0 AND impact_score <= 10', name='check_impact_range'),
    )
    
    def __repr__(self):
        return f"<RiskFlag(category='{self.category}', severity='{self.severity}', description='{self.description[:40]}...')>"


class Connection(Base):
    """
    Relationship between entities.
    
    Maps connections discovered during research.
    Used for network analysis and relationship graphs.
    """
    __tablename__ = "connections"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, ForeignKey("research_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Connection Details
    entity_1 = Column(String, nullable=False, index=True)  # First entity
    entity_2 = Column(String, nullable=False, index=True)  # Second entity
    relationship_type = Column(String, nullable=False)  # colleague, investor, family, etc.
    
    # Strength and Confidence
    strength = Column(Float, nullable=False)  # 0-1 connection strength
    confidence = Column(Float, nullable=False)  # 0-1 confidence
    
    # Additional Details
    time_period = Column(String)  # When connection existed
    evidence = Column(JSONB, default=list)  # Supporting evidence
    source_urls = Column(JSONB, default=list)  # Source URLs
    
    # Metadata
    discovered_by = Column(String)  # Which agent found this
    bidirectional = Column(Boolean, default=True)  # Is relationship mutual?
    
    # Timestamps
    discovered_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    session = relationship("ResearchSession", back_populates="connections")
    
    __table_args__ = (
        Index('ix_conn_session', 'session_id'),
        Index('ix_conn_entities', 'entity_1', 'entity_2'),
        CheckConstraint('strength >= 0 AND strength <= 1', name='check_strength_range'),
        CheckConstraint('confidence >= 0 AND confidence <= 1', name='check_conn_confidence_range'),
    )
    
    def __repr__(self):
        return f"<Connection({self.entity_1} <-[{self.relationship_type}]-> {self.entity_2})>"


# Export all models
__all__ = [
    "Base",
    "ResearchSession",
    "SearchResult",
    "Fact",
    "RiskFlag",
    "Connection"
]