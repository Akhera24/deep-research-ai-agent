"""
Research Agent State Management

This module defines the state structure that flows through the LangGraph workflow.
Every agent node receives this state and returns an updated version.

Design Decisions:
-----------------
1. TypedDict over dataclass: Required for LangGraph serialization
2. Immutable updates: Agents return new dicts, never mutate in place
3. Comprehensive tracking: Every metric needed for evaluation
4. Audit trail: Complete history for execution logs

Architecture Pattern:
--------------------
State flows through LangGraph nodes in this sequence:
  Initialize → Coordinator → Collector → Analyzer → Risk Assessor → Mapper → Complete

Each node:
  - Receives current state
  - Performs its operation
  - Returns updated state dict
  - Never mutates the original state
"""

from typing import TypedDict, List, Dict, Any, Optional
from datetime import datetime
from enum import Enum


class ResearchStage(str, Enum):
    """
    Workflow stages for the research process.
    
    Used for:
    - Progress tracking
    - Conditional routing in LangGraph
    - Execution log events
    - User-facing status updates
    """
    INITIALIZATION = "initialization"
    STRATEGY_PLANNING = "strategy_planning"
    DATA_COLLECTION = "data_collection"
    FACT_EXTRACTION = "fact_extraction"
    VERIFICATION = "verification"
    RISK_ASSESSMENT = "risk_assessment"
    CONNECTION_MAPPING = "connection_mapping"
    REPORT_GENERATION = "report_generation"
    COMPLETED = "completed"
    FAILED = "failed"


class SearchResult(TypedDict):
    """
    Individual search result from search engines.
    
    Contains both metadata and content for downstream processing.
    """
    query: str  # Original search query
    url: str  # Source URL
    title: str  # Page title
    snippet: str  # Search result snippet
    content: str  # Full page content (extracted)
    source_reliability: float  # 0-1 score based on domain
    relevance_score: float  # 0-1 score from search engine
    timestamp: str  # ISO format datetime
    search_engine: str  # "brave", "serper", etc.


class Fact(TypedDict):
    """
    Extracted and verified fact.
    
    Requirements Met:
    - Deep fact extraction (biographical, professional, financial)
    - Source validation (confidence scoring)
    - Cross-referencing (verification status)
    """
    id: str  # Unique fact identifier
    content: str  # The actual fact statement
    category: str  # "biographical", "professional", "financial", "legal", "connections"
    source_urls: List[str]  # All sources supporting this fact
    confidence_score: float  # 0-1 based on source reliability + cross-references
    verification_status: str  # "verified", "unverified", "conflicting"
    evidence: List[str]  # Supporting quotes/evidence from sources
    extracted_at: str  # ISO datetime
    cross_reference_count: int  # Number of independent sources


class RiskFlag(TypedDict):
    """
    Identified risk or red flag.
    
    Requirements Met:
    - Risk pattern recognition
    - Evidence-based flagging
    - Severity scoring
    """
    id: str
    category: str  # "financial", "legal", "reputational", "professional"
    description: str  # Human-readable risk description
    severity: str  # "low", "medium", "high", "critical"
    confidence: float  # 0-1 score
    impact_score: float  # 0-10 potential impact
    evidence: List[str]  # Fact IDs supporting this risk
    source_urls: List[str]  # Direct evidence URLs
    detected_at: str  # ISO datetime


class Connection(TypedDict):
    """
    Relationship between entities.
    
    Requirements Met:
    - Connection mapping
    - Non-obvious relationship discovery
    - Temporal tracking
    """
    id: str
    entity_1: str  # First entity name
    entity_2: str  # Second entity name
    relationship_type: str  # "colleague", "investor", "family", "business_partner"
    strength: float  # 0-1 connection strength
    confidence: float  # 0-1 confidence score
    time_period: Optional[str]  # When connection existed
    evidence: List[str]  # Supporting evidence
    source_urls: List[str]  # Source URLs
    discovered_at: str  # ISO datetime


class ResearchState(TypedDict):
    """
    Main state object that flows through LangGraph workflow.
    
    This is the CORE data structure of the entire system.
    Every agent receives this and returns an updated version.
    
    State Update Pattern:
    --------------------
```python
    def agent_node(state: ResearchState) -> ResearchState:
        # Extract current values
        current_facts = state["facts"]
        
        # Do work
        new_facts = extract_facts(...)
        
        # Return updated state
        return {
            **state,  # Keep all existing fields
            "facts": current_facts + new_facts,  # Add new facts
            "current_stage": ResearchStage.FACT_EXTRACTION,
            "iteration_count": state["iteration_count"] + 1
        }
```
    
    Requirements Traceability:
    -------------------------
    - Multi-model integration: tracked in metrics["model_calls"]
    - Consecutive search: iteration_count, search_history
    - Dynamic query refinement: search_plan evolution
    - Deep fact extraction: facts list with categories
    - Risk pattern recognition: risk_flags list
    - Connection mapping: connections list
    - Source validation: confidence_scores dict
    """
    
    # === INPUT ===
    query: str  # Original user query
    target_entity: Dict[str, Any]  # Extracted entity info {"name": "X", "type": "person"}
    research_parameters: Dict[str, Any]  # User preferences (depth, focus areas, etc.)
    
    # === WORKFLOW CONTROL ===
    current_stage: str  # Current ResearchStage
    iteration_count: int  # Number of search iterations completed
    max_iterations: int  # Maximum allowed iterations
    should_continue: bool  # Whether to continue or stop
    run_id: str  # Unique identifier for this research run
    
    # === SEARCH MANAGEMENT ===
    search_plan: List[str]  # Planned search queries
    executed_searches: List[SearchResult]  # All searches executed
    pending_searches: List[str]  # Queries waiting to be executed
    search_coverage: Dict[str, float]  # {"biographical": 0.8, "financial": 0.3}
    
    # === EXTRACTED DATA ===
    facts: List[Fact]  # All discovered facts
    risk_flags: List[RiskFlag]  # All identified risks
    connections: List[Connection]  # All entity connections
    timeline: List[Dict[str, Any]]  # Chronological events
    
    # === QUALITY METRICS ===
    confidence_scores: Dict[str, float]  # Per-category confidence
    source_diversity_score: float  # 0-1 variety of sources
    coverage_score: float  # 0-1 comprehensiveness
    
    # === METADATA & AUDIT ===
    start_time: str  # ISO datetime when research started
    end_time: Optional[str]  # ISO datetime when completed
    total_api_calls: int  # Number of AI model calls
    total_searches: int  # Number of search queries
    total_cost: float  # Estimated cost in USD
    errors: List[Dict[str, Any]]  # Error log
    
    # === AGENT COMMUNICATION ===
    agent_messages: List[Dict[str, Any]]  # Inter-agent messages
    last_agent: Optional[str]  # Last agent that updated state


def create_initial_state(
    query: str,
    max_iterations: int = 20,
    research_parameters: Optional[Dict[str, Any]] = None
) -> ResearchState:
    """
    Create initial state for a new research session.
    
    Args:
        query: User's research query (e.g., "Investigate John Doe")
        max_iterations: Maximum search iterations allowed
        research_parameters: Optional parameters (focus areas, depth, etc.)
    
    Returns:
        Initialized ResearchState ready for LangGraph
    
    Example:
        >>> state = create_initial_state("Research Sarah Chen", max_iterations=15)
        >>> state["current_stage"]
        'initialization'
        >>> state["facts"]
        []
    """
    import uuid
    
    return ResearchState(
        # Input
        query=query,
        target_entity={"name": query, "type": "unknown"},  # Will be extracted by coordinator
        research_parameters=research_parameters or {},
        
        # Workflow control
        current_stage=ResearchStage.INITIALIZATION.value,
        iteration_count=0,
        max_iterations=max_iterations,
        should_continue=True,
        run_id=str(uuid.uuid4()),
        
        # Search management
        search_plan=[],
        executed_searches=[],
        pending_searches=[],
        search_coverage={},
        
        # Extracted data
        facts=[],
        risk_flags=[],
        connections=[],
        timeline=[],
        
        # Quality metrics
        confidence_scores={},
        source_diversity_score=0.0,
        coverage_score=0.0,
        
        # Metadata
        start_time=datetime.utcnow().isoformat() + "Z",
        end_time=None,
        total_api_calls=0,
        total_searches=0,
        total_cost=0.0,
        errors=[],
        
        # Agent communication
        agent_messages=[],
        last_agent=None
    )


def validate_state(state: ResearchState) -> bool:
    """
    Validate state structure and required fields.
    
    Used for:
    - Debugging state corruption
    - Ensuring state consistency
    - Pre-deployment validation
    
    Returns:
        True if valid, raises ValueError if invalid
    """
    required_fields = [
        "query", "current_stage", "iteration_count", "max_iterations",
        "facts", "risk_flags", "connections", "run_id"
    ]
    
    for field in required_fields:
        if field not in state:
            raise ValueError(f"Missing required field: {field}")
    
    # Validate types
    if not isinstance(state["facts"], list):
        raise ValueError("facts must be a list")
    
    if not isinstance(state["iteration_count"], int):
        raise ValueError("iteration_count must be an integer")
    
    return True


class StateManager:
    """
    Manager for ResearchState operations.
    
    Provides high-level operations for state management:
    - State creation and initialization
    - State validation and verification
    - Immutable state updates
    - State persistence (database integration)
    - State history tracking
    
    Features:
    - Immutable state pattern
    - Type validation
    - Error handling
    - Database integration ready
    - Audit trail support
    
    Example:
        >>> manager = StateManager()
        >>> state = manager.create_state("Research Sarah Chen")
        >>> manager.validate(state)
        True
        >>> updated = manager.update(state, current_stage="data_collection")
    """
    
    def __init__(self, db_session=None):
        """
        Initialize StateManager.
        
        Args:
            db_session: Optional database session for persistence
        """
        self.db_session = db_session
        self.state_history: List[Dict[str, Any]] = []
    
    def create_state(
        self,
        query: str,
        max_iterations: int = 20,
        research_parameters: Optional[Dict[str, Any]] = None
    ) -> ResearchState:
        """
        Create a new research state.
        
        Wrapper around create_initial_state with additional tracking.
        
        Args:
            query: Research query
            max_iterations: Max search iterations
            research_parameters: Optional parameters
        
        Returns:
            Initialized ResearchState
        """
        state = create_initial_state(query, max_iterations, research_parameters)
        
        # Track state creation
        self._track_state(state, "created")
        
        return state
    
    def validate(self, state: ResearchState) -> bool:
        """
        Validate state structure and consistency.
        
        Args:
            state: State to validate
        
        Returns:
            True if valid
        
        Raises:
            ValueError: If state is invalid
        """
        return validate_state(state)
    
    def update(
        self,
        state: ResearchState,
        **updates
    ) -> ResearchState:
        """
        Create updated copy of state (immutable pattern).
        
        Args:
            state: Current state
            **updates: Fields to update
        
        Returns:
            New state with updates applied
        
        Example:
            >>> new_state = manager.update(
            ...     state,
            ...     current_stage="data_collection",
            ...     iteration_count=state["iteration_count"] + 1
            ... )
        """
        # Create new state with updates
        updated_state = {**state, **updates}
        
        # Validate after update
        self.validate(updated_state)
        
        # Track update
        self._track_state(updated_state, "updated")
        
        return updated_state
    
    def add_fact(self, state: ResearchState, fact: Fact) -> ResearchState:
        """
        Add a fact to state (convenience method).
        
        Args:
            state: Current state
            fact: Fact to add
        
        Returns:
            Updated state with new fact
        """
        return self.update(
            state,
            facts=state["facts"] + [fact]
        )
    
    def add_risk_flag(self, state: ResearchState, risk: RiskFlag) -> ResearchState:
        """Add a risk flag to state."""
        return self.update(
            state,
            risk_flags=state["risk_flags"] + [risk]
        )
    
    def add_connection(self, state: ResearchState, connection: Connection) -> ResearchState:
        """Add a connection to state."""
        return self.update(
            state,
            connections=state["connections"] + [connection]
        )
    
    def add_search_result(self, state: ResearchState, result: SearchResult) -> ResearchState:
        """Add a search result to state."""
        return self.update(
            state,
            executed_searches=state["executed_searches"] + [result],
            total_searches=state["total_searches"] + 1
        )
    
    def increment_iteration(self, state: ResearchState) -> ResearchState:
        """Increment iteration counter."""
        return self.update(
            state,
            iteration_count=state["iteration_count"] + 1
        )
    
    def add_cost(self, state: ResearchState, cost: float) -> ResearchState:
        """Add to total cost."""
        return self.update(
            state,
            total_cost=state["total_cost"] + cost,
            total_api_calls=state["total_api_calls"] + 1
        )
    
    def mark_completed(self, state: ResearchState) -> ResearchState:
        """Mark state as completed."""
        return self.update(
            state,
            current_stage=ResearchStage.COMPLETED.value,
            should_continue=False,
            end_time=datetime.utcnow().isoformat() + "Z"
        )
    
    def mark_failed(self, state: ResearchState, error: str) -> ResearchState:
        """Mark state as failed."""
        return self.update(
            state,
            current_stage=ResearchStage.FAILED.value,
            should_continue=False,
            end_time=datetime.utcnow().isoformat() + "Z",
            errors=state["errors"] + [{
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "error": error
            }]
        )
    
    def get_summary(self, state: ResearchState) -> Dict[str, Any]:
        """
        Get summary statistics from state.
        
        Returns:
            Dict with key metrics
        """
        return {
            "run_id": state["run_id"],
            "query": state["query"],
            "current_stage": state["current_stage"],
            "iteration_count": state["iteration_count"],
            "facts_count": len(state["facts"]),
            "risk_flags_count": len(state["risk_flags"]),
            "connections_count": len(state["connections"]),
            "total_searches": state["total_searches"],
            "total_cost": state["total_cost"],
            "total_api_calls": state["total_api_calls"],
            "duration_seconds": self._calculate_duration(state),
            "coverage_score": state["coverage_score"],
            "should_continue": state["should_continue"]
        }
    
    def save_to_db(self, state: ResearchState) -> None:
        """
        Persist state to database.
        
        Args:
            state: State to save
        
        Note:
            Requires db_session to be set in __init__
        """
        if not self.db_session:
            raise ValueError("Database session not configured")
        
        # Implementation will be added when we integrate with repositories
        # For now, this is a placeholder for the interface
        pass
    
    def load_from_db(self, run_id: str) -> Optional[ResearchState]:
        """
        Load state from database.
        
        Args:
            run_id: Research run identifier
        
        Returns:
            ResearchState if found, None otherwise
        """
        if not self.db_session:
            raise ValueError("Database session not configured")
        
        # Implementation will be added when we integrate with repositories
        # For now, this is a placeholder for the interface
        pass
    
    def _track_state(self, state: ResearchState, action: str) -> None:
        """
        Track state in history (for debugging/audit).
        
        Args:
            state: Current state
            action: Action performed ("created", "updated", etc.)
        """
        self.state_history.append({
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "action": action,
            "run_id": state["run_id"],
            "stage": state["current_stage"],
            "iteration": state["iteration_count"],
            "facts_count": len(state["facts"])
        })
    
    def _calculate_duration(self, state: ResearchState) -> Optional[float]:
        """Calculate duration if research is completed."""
        if not state["end_time"]:
            return None
        
        start = datetime.fromisoformat(state["start_time"].replace("Z", "+00:00"))
        end = datetime.fromisoformat(state["end_time"].replace("Z", "+00:00"))
        
        return (end - start).total_seconds()
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get state history for debugging."""
        return self.state_history.copy()

# Export all types
__all__ = [
    "ResearchStage",
    "SearchResult",
    "Fact",
    "RiskFlag",
    "Connection",
    "ResearchState",
    "create_initial_state",
    "validate_state",
    "StateManager" 
]