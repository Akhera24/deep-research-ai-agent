"""
LangGraph Research Workflow - Production-Ready Implementation

Complete research pipeline orchestration using LangGraph with:
- Multi-stage consecutive search strategy
- AI-powered fact extraction
- Verification and confidence scoring  
- Risk assessment
- Connection mapping
- State management and checkpointing

Features :
 "Uses LangGraph for agent orchestration"
 "Consecutive Search Strategy: Design an intelligent search progression"
 "Dynamic Query Refinement: Agent must adapt search strategies"
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import re



from langgraph.graph import StateGraph, END

# SqliteSaver is OPTIONAL - checkpointing is a nice-to-have feature
# The system works perfectly without it!
SqliteSaver = None  # Default to no checkpointing

# Try to import SqliteSaver if available (not required)
try:
    from langgraph_checkpoint.sqlite import SqliteSaver
    _CHECKPOINT_AVAILABLE = True
except ImportError:
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        _CHECKPOINT_AVAILABLE = True
    except ImportError:
        _CHECKPOINT_AVAILABLE = False

from config.logging_config import get_logger
from src.core.state_manager import ResearchState
from src.models.router import ModelRouter
from src.search.strategy import SearchStrategyEngine, SearchQuery, SearchCategory
from src.search.executor import SearchExecutor, SearchResult
from src.extraction.extractor import FactExtractor, Fact

logger = get_logger(__name__)


# ============================================================================
# RESEARCH ORCHESTRATOR
# ============================================================================

class ResearchOrchestrator:
    """
    Main research orchestration engine using LangGraph.
    
    This is the PRIMARY ENTRY POINT for all research operations.
    
    Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │  START                                                   │
    │    ↓                                                     │
    │  Initialize (validate input, setup)                     │
    │    ↓                                                     │
    │  Plan Strategy (generate initial queries)               │
    │    ↓                                                     │
    │  ╔═══════════════════════════════════╗                 │
    │  ║  CONSECUTIVE SEARCH LOOP          ║                 │
    │  ║  ┌──────────────────────────┐    ║                 │
    │  ║  │ Execute Searches         │    ║                 │
    │  ║  └──────────┬───────────────┘    ║                 │
    │  ║             ↓                      ║                 │
    │  ║  ┌──────────────────────────┐    ║                 │
    │  ║  │ Extract Facts            │    ║                 │
    │  ║  └──────────┬───────────────┘    ║                 │
    │  ║             ↓                      ║                 │
    │  ║  ┌──────────────────────────┐    ║                 │
    │  ║  │ Refine Queries           │    ║                 │
    │  ║  │ (based on findings)      │    ║                 │
    │  ║  └──────────┬───────────────┘    ║                 │
    │  ║             ↓                      ║                 │
    │  ║  Decision: Continue or Stop?      ║                 │
    │  ║    ├─ Continue → Loop back        ║                 │
    │  ║    └─ Stop → Exit loop           ║                 │
    │  ╚═══════════════════════════════════╝                 │
    │    ↓                                                     │
    │  Verify Facts (cross-reference)                        │
    │    ↓                                                     │
    │  Assess Risks (pattern detection)                      │
    │    ↓                                                     │
    │  Map Connections (relationship graph)                  │
    │    ↓                                                     │
    │  Generate Report                                        │
    │    ↓                                                     │
    │  END                                                     │
    └─────────────────────────────────────────────────────────┘
    
    Example:
        >>> orchestrator = ResearchOrchestrator()
        >>> result = await orchestrator.research("Sarah Chen")
        >>> print(f"Found {len(result['facts'])} facts")
        >>> print(f"Risk score: {result['risk_score']}")
    """
    
    def __init__(
        self,
        max_iterations: int = 5,
        max_facts: int = 50,
        enable_checkpoints: bool = True
    ):
        """
        Initialize research orchestrator.
        
        Args:
            max_iterations: Max consecutive search iterations
            max_facts: Max facts to extract per target
            enable_checkpoints: Enable LangGraph checkpointing
        """
        self.max_iterations = max_iterations
        self.max_facts = max_facts
        self.enable_checkpoints = enable_checkpoints
        
        # Initialize components
        self.router = ModelRouter()
        self.strategy_engine = SearchStrategyEngine(self.router)
        self.search_executor = SearchExecutor()
        self.fact_extractor = FactExtractor(self.router)
        
        # Build LangGraph workflow
        self.workflow = self._build_workflow()
        
        logger.info(
            "Research orchestrator initialized",
            extra={
                "max_iterations": max_iterations,
                "max_facts": max_facts,
                "checkpoints_requested": enable_checkpoints,
                "checkpoints_available": _CHECKPOINT_AVAILABLE and SqliteSaver is not None
            }
        )
    
    # ========================================================================
    # PUBLIC API
    # ========================================================================
    
    async def research(
        self,
        target_name: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute complete research on target.
        
        Args:
            target_name: Person/entity to research
            context: Optional context (occupation, location, etc.)
            
        Returns:
            Dictionary with:
            - facts: List of extracted facts
            - queries_executed: Number of searches
            - coverage: Research coverage metrics
            - risk_flags: Identified risks
            - connections: Mapped relationships
            - metadata: Research metadata
        
        Example:
            >>> result = await orchestrator.research(
            ...     "Sarah Chen",
            ...     context={"occupation": "CEO", "company": "TechCorp"}
            ... )
            >>> print(f"Research complete: {len(result['facts'])} facts found")
        """
        if not target_name or not target_name.strip():
            raise ValueError("target_name cannot be empty")
        
        logger.info(
            "Starting research",
            extra={
                "target": target_name,
                "has_context": bool(context)
            }
        )
        
        # Initialize state
        initial_state = {
            "target_name": target_name,
            "context": context or {},
            "iteration": 0,
            "max_iterations": self.max_iterations,
            "queries": [],
            "search_results": [],
            "facts": [],
            "risk_flags": [],
            "connections": [],
            "stage": "initialization",
            "start_time": datetime.now(),
            "errors": []
        }
        
        try:
            # Run workflow
            final_state = await self.workflow.ainvoke(initial_state)
            
            # Format results
            result = self._format_results(final_state)
            
            logger.info(
                "Research complete",
                extra={
                    "target": target_name,
                    "facts": len(result["facts"]),
                    "iterations": result["metadata"]["iterations"],
                    "duration_seconds": result["metadata"]["duration_seconds"]
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "Research failed",
                extra={"target": target_name, "error": str(e)},
                exc_info=True
            )
            raise
    
    # ========================================================================
    # LANGGRAPH WORKFLOW CONSTRUCTION
    # ========================================================================
    
    def _build_workflow(self) -> StateGraph:
        """
        Build LangGraph workflow.
        
        This is the core orchestration logic that defines how
        the research process flows from start to finish.
        """
        # Create state graph
        workflow = StateGraph(Dict[str, Any])
        
        # Add nodes (each node is an async function)
        workflow.add_node("initialize", self._node_initialize)
        workflow.add_node("plan_strategy", self._node_plan_strategy)
        workflow.add_node("execute_searches", self._node_execute_searches)
        workflow.add_node("extract_facts", self._node_extract_facts)
        workflow.add_node("refine_queries", self._node_refine_queries)
        workflow.add_node("verify_facts", self._node_verify_facts)
        workflow.add_node("assess_risks", self._node_assess_risks)
        workflow.add_node("map_connections", self._node_map_connections)
        workflow.add_node("generate_report", self._node_generate_report)
        
        # Define edges (workflow transitions)
        workflow.set_entry_point("initialize")
        workflow.add_edge("initialize", "plan_strategy")
        workflow.add_edge("plan_strategy", "execute_searches")
        workflow.add_edge("execute_searches", "extract_facts")
        workflow.add_edge("extract_facts", "refine_queries")
        
        # Conditional edge: decide whether to continue searching
        workflow.add_conditional_edges(
            "refine_queries",
            self._decide_continue_or_finish,
            {
                "continue": "execute_searches",  # Loop back for more searches
                "verify": "verify_facts"          # Move to verification
            }
        )
        
        workflow.add_edge("verify_facts", "assess_risks")
        workflow.add_edge("assess_risks", "map_connections")
        workflow.add_edge("map_connections", "generate_report")
        workflow.add_edge("generate_report", END)
        
        # Compile workflow with optional checkpointing
        if self.enable_checkpoints and SqliteSaver is not None and _CHECKPOINT_AVAILABLE:
            try:
                checkpointer = SqliteSaver.from_conn_string("checkpoints.db")
                logger.info("Workflow compiled WITH checkpointing (can resume interrupted research)")
                return workflow.compile(checkpointer=checkpointer)
            except Exception as e:
                logger.warning(f"Checkpointing initialization failed: {e}. Continuing without checkpointing.")
                return workflow.compile()
        else:
            if self.enable_checkpoints:
                logger.info(
                    "Checkpointing requested but not available. "
                    "Workflow will run without checkpointing (this is fine!)."
                )
            else:
                logger.info("Workflow compiled WITHOUT checkpointing (as requested)")
            return workflow.compile()
    
    # ========================================================================
    # WORKFLOW NODES (Each is an agent action)
    # ========================================================================
    
    async def _node_initialize(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Initialize research session.
        
        Validates input and sets up initial state.
        """
        logger.info(f"Node: Initialize - {state['target_name']}")
        
        state["stage"] = "initialization"
        state["initialized_at"] = datetime.now()
        
        # Reset strategy engine for new research
        self.strategy_engine.reset()
        
        return state
    
    async def _node_plan_strategy(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Plan initial research strategy.
        
        Uses SearchStrategyEngine to generate intelligent initial queries.
        """
        logger.info(f"Node: Plan Strategy - {state['target_name']}")
        
        state["stage"] = "strategy_planning"
        
        # Generate initial queries
        queries = self.strategy_engine.generate_initial_queries(
            target_name=state["target_name"],
            context=state["context"],
            max_queries=15
        )
        
        state["queries"] = queries
        state["pending_queries"] = [q.text for q in queries]
        
        logger.info(f"Generated {len(queries)} initial queries")
        
        return state
    
    async def _node_execute_searches(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute search queries.
        
        Takes top N queries and executes them in parallel.
        """
        logger.info(f"Node: Execute Searches (iteration {state['iteration']})")
        
        state["stage"] = "data_collection"
        
        # Get next batch of queries (top 3)
        batch_size = 3
        queries_to_execute = state["queries"][:batch_size]
        
        # Execute searches in parallel
        all_results = []
        for query in queries_to_execute:
            try:
                results = await self.search_executor.search(
                    query.text,
                    max_results=10
                )
                all_results.extend(results)
                
                # Mark query as executed
                self.strategy_engine.mark_executed(query, len(results))
                
            except Exception as e:
                logger.error(f"Search failed for '{query.text}': {e}")
                state["errors"].append({
                    "stage": "search",
                    "query": query.text,
                    "error": str(e)
                })
        
        # Update state
        if "search_results" not in state:
            state["search_results"] = []
        state["search_results"].extend(all_results)
        
        # Remove executed queries
        state["queries"] = state["queries"][batch_size:]
        
        logger.info(f"Executed {len(queries_to_execute)} queries, got {len(all_results)} results")
        
        return state
    
    async def _node_extract_facts(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract facts from search results.
        
        Uses FactExtractor to parse structured facts from unstructured text.
        """
        logger.info(f"Node: Extract Facts")
        
        state["stage"] = "fact_extraction"
        
        # Extract facts from recent results
        recent_results = state["search_results"]
        
        if recent_results:
            facts = await self.fact_extractor.extract(
                search_results=recent_results,
                target_name=state["target_name"],
                max_facts=self.max_facts
            )
            
            # Add to state
            if "facts" not in state:
                state["facts"] = []
            state["facts"].extend(facts)
            
            # Update coverage based on facts
            for fact in facts:
                try:
                    category = SearchCategory(fact.category)
                    self.strategy_engine.update_coverage(category, increment=0.15)
                except:
                    pass
            
            logger.info(f"Extracted {len(facts)} facts (total: {len(state['facts'])})")
        
        return state
    
    async def _node_refine_queries(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Refine search queries based on findings.
        
        THIS IS THE CORE OF CONSECUTIVE SEARCH!
        
        Analyzes discovered facts and generates targeted follow-up queries.
        """
        logger.info(f"Node: Refine Queries")
        
        state["stage"] = "query_refinement"
        state["iteration"] += 1
        
        # Convert facts to findings format
        findings = [
            {
                "content": fact.content,
                "category": fact.category,
                "entities": fact.entities_mentioned
            }
            for fact in state.get("facts", [])
        ]
        
        # Generate follow-up queries
        if findings:
            refined_queries = self.strategy_engine.refine_based_on_findings(
                target_name=state["target_name"],
                findings=findings,
                max_follow_ups=10
            )
            
            # Add refined queries to queue
            state["queries"].extend(refined_queries)
            
            logger.info(f"Generated {len(refined_queries)} follow-up queries")
        
        return state
    
    async def _node_verify_facts(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify facts through cross-referencing.
        
        Increases confidence for facts that appear in multiple sources.
        """
        logger.info(f"Node: Verify Facts")
        
        state["stage"] = "verification"
        
        # Verification is already handled in fact_extractor
        # Here we just log statistics
        
        facts = state.get("facts", [])
        high_conf = [f for f in facts if f.is_high_confidence()]
        verified = [f for f in facts if f.verified]
        
        logger.info(
            f"Verification complete: {len(high_conf)} high-confidence, "
            f"{len(verified)} verified"
        )
        
        return state

# ==============================================================================
# RISK ASSESSMENT METHOD
# ==============================================================================

    async def _node_assess_risks(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced AI-powered risk assessment with more sophisticated analysis.
        
        Features:
        1. 7 risk categories that are analyzed
        2. Trend detection (patterns across facts)
        3. Relationship risk analysis
        4. Compliance risk detection
        5. Evidence linking to specific facts
        """
        logger.info("Node: Assess Risks (Enhanced)")
        state["stage"] = "risk_assessment"
        
        facts = state.get("facts", [])
        connections = state.get("connections", [])
        target_name = state["target_name"]
        
        # Quick return if insufficient data
        if not facts or len(facts) < 5:
            state["risk_flags"] = []
            logger.info("Insufficient facts for risk assessment")
            return state
        
        try:
            # Build comprehensive context for AI
            facts_summary = []
            for i, fact in enumerate(facts[:40], 1):  # Analyze up to 40 facts
                category = getattr(fact, 'category', 'unknown')
                content = getattr(fact, 'content', '')
                confidence = getattr(fact, 'confidence', 0.5)
                facts_summary.append(f"{i}. [{category}] {content} (conf: {confidence:.2f})")
            
            facts_text = "\n".join(facts_summary)
            
            # Add connection context if available
            connections_text = ""
            if connections:
                connections_text = "\n\nKNOWN CONNECTIONS:\n"
                for i, conn in enumerate(connections[:15], 1):
                    entity_2 = conn.get('entity_2', 'Unknown')
                    rel_type = conn.get('relationship_type', 'unknown')
                    connections_text += f"{i}. {target_name} ←→ {entity_2} ({rel_type})\n"
            
            # Enhanced AI prompt with more categories
            prompt = f"""Analyze these facts about {target_name} and identify potential risks or red flags.

    FACTS DISCOVERED:
    {facts_text}
    {connections_text}

    COMPREHENSIVE RISK ASSESSMENT:
    Identify ANY risks in these categories:

    1. **Financial**: conflicts of interest, unusual transactions, compensation issues, hidden assets
    2. **Legal**: lawsuits, investigations, regulatory scrutiny, compliance issues
    3. **Reputational**: controversies, scandals, negative associations, criticism, bad press
    4. **Professional**: employment gaps, conflicts, questionable decisions, performance issues
    5. **Integrity**: contradictions, inconsistencies, misleading information
    6. **Compliance**: regulatory violations, policy breaches, ethical concerns
    7. **Relationship**: problematic associations, conflict of interest through connections

    For EACH risk identified, provide:
    - category: financial, legal, reputational, professional, integrity, compliance, or relationship
    - severity: low, medium, high, or critical
    - description: Clear, specific explanation (2-3 sentences)
    - confidence: 0.0 to 1.0 score
    - impact_score: 1-10 potential impact rating
    - evidence: List of specific facts that support this risk (fact numbers)
    - trend: Is this an isolated issue or pattern? (isolated/emerging/established)

    IMPORTANT GUIDELINES:
    - Be conservative - only flag genuine concerns with solid evidence
    - Don't create risks from normal business activities
    - Look for PATTERNS across multiple facts (more concerning than isolated incidents)
    - Consider both direct risks AND risks from connections
    - If NO significant risks found, return empty array []

    Return ONLY a JSON array of risk objects.

    Example output:
    ```json
    [
    {{
        "category": "legal",
        "severity": "medium",
        "description": "Subject mentioned in ongoing antitrust investigation. Microsoft faces FTC scrutiny over cloud computing practices and market dominance concerns.",
        "confidence": 0.75,
        "impact_score": 7.0,
        "evidence": ["Fact 5", "Fact 12"],
        "trend": "emerging"
    }},
    {{
        "category": "integrity",
        "severity": "low",
        "description": "Multiple duplicate facts suggest data quality issues. 6 facts appear identical across different search iterations.",
        "confidence": 0.9,
        "impact_score": 2.0,
        "evidence": ["Facts 1&19", "Facts 6&18"],
        "trend": "isolated"
    }}
    ]
    ```

    Analyze carefully and return ONLY the JSON array.
    """
            
            # Call AI model for risk assessment
            from src.models.router import TaskType
            
            logger.debug(f"Calling AI for enhanced risk assessment of {target_name}")
            
            response = self.router.route(
                prompt=prompt,
                task_type=TaskType.RISK_ASSESSMENT
            )
            
            logger.debug(f"AI response received, parsing risks")
            
            # Parse AI response
            risk_flags = self._parse_json_array_response(
                response.content,
                target_name,
                "risk_flags"
            )
            
            # Add metadata to each risk flag
            for i, risk in enumerate(risk_flags):
                risk["id"] = f"risk_{i}"
                risk["detected_at"] = datetime.now().isoformat()
                risk["source"] = "ai_analysis"
                
                # Ensure all required fields exist with sensible defaults
                risk.setdefault("category", "unknown")
                risk.setdefault("severity", "low")
                risk.setdefault("description", "Risk identified through analysis")
                risk.setdefault("confidence", 0.5)
                risk.setdefault("impact_score", 5.0)
                risk.setdefault("evidence", [])
                risk.setdefault("trend", "isolated")
                
                # Validate trend field
                if risk.get("trend") not in ["isolated", "emerging", "established"]:
                    risk["trend"] = "isolated"
            
            state["risk_flags"] = risk_flags
            
            logger.info(
                f"Identified {len(risk_flags)} risk flags via AI (enhanced)",
                extra={
                    "target": target_name,
                    "risks": len(risk_flags),
                    "categories": list(set(r.get("category") for r in risk_flags)) if risk_flags else [],
                    "severity_breakdown": self._count_by_severity(risk_flags)
                }
            )
            
            return state
            
        except Exception as e:
            logger.error(
                f"AI risk assessment failed: {e}",
                extra={"target": target_name, "error": str(e)},
                exc_info=True
            )
            
            # FALLBACK: Pattern-based risk detection
            logger.warning("Falling back to pattern-based risk detection")
            state["risk_flags"] = self._fallback_pattern_risk_detection(facts, target_name)
            logger.info(f"Fallback found {len(state['risk_flags'])} risk flags")
            
            return state


    def _count_by_severity(self, risks: List[Dict]) -> Dict[str, int]:
        """Helper to count risks by severity level"""
        counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        for risk in risks:
            severity = risk.get("severity", "low")
            counts[severity] = counts.get(severity, 0) + 1
        return counts

# ==============================================================================
# CONNECTION MAPPING METHOD (Replace _node_map_connections at line ~525)
# ==============================================================================

    async def _node_map_connections(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        AI-powered connection mapping with entity extraction.
        
        CORRECTED VERSION: Works with Fact dataclass objects
        
        Extracts:
        - Professional relationships (colleagues, predecessors, successors)
        - Organizational connections (employers, board memberships)
        - Educational connections (universities, alma maters)
        - Business connections (investors, partners, co-founders)
        
        FAANG Pattern: AI extraction with regex fallback
        """
        logger.info("Node: Map Connections")
        state["stage"] = "connection_mapping"
        
        facts = state.get("facts", [])
        target_name = state["target_name"]
        
        # Quick return if insufficient data
        if not facts or len(facts) < 3:
            state["connections"] = []
            logger.info("Insufficient facts for connection mapping")
            return state
        
        try:
            # Filter facts most likely to contain connections
            relevant_facts = []
            for fact in facts:
                # ✅ CORRECTED: Use attribute access for Fact dataclass
                content = getattr(fact, 'content', '')
                category = getattr(fact, 'category', '')
                
                # Prioritize connection-rich categories
                if category in ['professional', 'connections', 'biographical']:
                    relevant_facts.append(content)
                # Also include facts with connection keywords
                elif any(keyword in content.lower() for keyword in [
                    'worked', 'joined', 'succeeded', 'replaced', 'partnered',
                    'founded', 'board', 'colleague', 'attended', 'graduated',
                    'married', 'family', 'friend', 'mentor', 'student', 'chairman',
                    'ceo of', 'president of', 'member of'
                ]):
                    relevant_facts.append(content)
            
            # If no relevant facts found, use all
            if not relevant_facts:
                relevant_facts = [getattr(f, 'content', '') for f in facts[:30]]
            else:
                relevant_facts = relevant_facts[:35]  # Limit to 35 most relevant
            
            # Build numbered fact list for AI
            facts_text = "\n".join([f"{i}. {fact}" for i, fact in enumerate(relevant_facts, 1)])
            
            # AI prompt for connection extraction
            prompt = f"""Extract ALL connections and relationships for {target_name} from these facts:

    FACTS:
    {facts_text}

    CONNECTION EXTRACTION TASK:
    Identify EVERY person, organization, or institution that {target_name} has a connection to.

    For EACH connection, provide:
    1. entity_2: The other entity's name (person, company, university, organization)
    2. relationship_type: Type of relationship:
    - predecessor/successor (people they replaced or who replaced them)
    - colleague (worked together)
    - employer (companies they worked for)
    - leadership (organizations they lead/led)
    - board_member (boards they serve/served on)
    - education (schools attended)
    - family (relatives)
    - investor/partner (business relationships)
    3. strength: 0.0-1.0 score:
    - 1.0 = direct family, direct reports, long-term employment
    - 0.8-0.9 = close colleagues, board membership, alma mater
    - 0.6-0.7 = indirect colleagues, brief employment
    - 0.4-0.5 = distant connections
    4. confidence: 0.0-1.0 score based on evidence quality
    5. time_period: When connection existed (e.g., "2014", "1992-present", "2010-2015")
    6. evidence: Direct quote from facts showing this connection

    Return ONLY a JSON array. Extract ALL connections you can find.

    Expected output format:
    ```json
    [
    {{
        "entity_1": "{target_name}",
        "entity_2": "Steve Ballmer",
        "relationship_type": "predecessor",
        "strength": 0.9,
        "confidence": 0.95,
        "time_period": "2014",
        "evidence": "succeeded Steve Ballmer as CEO of Microsoft in 2014"
    }},
    {{
        "entity_1": "{target_name}",
        "entity_2": "Microsoft",
        "relationship_type": "employer",
        "strength": 1.0,
        "confidence": 1.0,
        "time_period": "1992-present",
        "evidence": "joined Microsoft in 1992"
    }},
    {{
        "entity_1": "{target_name}",
        "entity_2": "University of Chicago",
        "relationship_type": "education",
        "strength": 0.8,
        "confidence": 0.9,
        "time_period": "1997",
        "evidence": "earned MBA from University of Chicago Booth School of Business in 1997"
    }}
    ]
    ```

    IMPORTANT: Extract EVERY entity mentioned - people, companies, schools, organizations.
    """
            
            # Call AI model for connection extraction
            from src.models.router import TaskType
            
            logger.debug(f"Calling AI for connection mapping of {target_name}")
            
            response = self.router.route(
                prompt=prompt,
                task_type=TaskType.ANALYSIS 
            )
            
            logger.debug(f"AI response received, parsing connections")
            
            # Parse AI response
            connections = self._parse_json_array_response(
                response.content,
                target_name,
                "connections"
            )
            
            # Add metadata and validate each connection
            for i, conn in enumerate(connections):
                conn["id"] = f"conn_{i}"
                conn["discovered_at"] = datetime.now().isoformat()
                conn["source"] = "ai_extraction"
                
                # Ensure entity_1 is always the target
                if conn.get("entity_1") != target_name:
                    conn["entity_1"] = target_name
                
                # Ensure all required fields exist
                conn.setdefault("entity_2", "Unknown")
                conn.setdefault("relationship_type", "associated")
                conn.setdefault("strength", 0.5)
                conn.setdefault("confidence", 0.5)
                conn.setdefault("time_period", "Unknown")
                conn.setdefault("evidence", [])
                
                # If evidence is a string, make it a list
                if isinstance(conn.get("evidence"), str):
                    conn["evidence"] = [conn["evidence"]]
            
            # Deduplicate connections (same entity + relationship type)
            seen = set()
            unique_connections = []
            for conn in connections:
                entity_2 = conn.get("entity_2", "")
                rel_type = conn.get("relationship_type", "")
                
                key = (conn.get("entity_1"), entity_2, rel_type)
                
                # Filter out invalid connections
                if (entity_2 and 
                    entity_2 != "Unknown" and
                    entity_2 != target_name and
                    len(entity_2) > 1 and
                    key not in seen):
                    seen.add(key)
                    unique_connections.append(conn)
            
            state["connections"] = unique_connections
            
            logger.info(
                f"Mapped {len(unique_connections)} connections via AI",
                extra={
                    "target": target_name,
                    "connections": len(unique_connections),
                    "types": list(set(c.get("relationship_type") for c in unique_connections))
                }
            )
            
            return state
            
        except Exception as e:
            logger.error(
                f"AI connection mapping failed: {e}",
                extra={"target": target_name, "error": str(e)},
                exc_info=True
            )
            
            # FALLBACK: Pattern-based connection extraction
            logger.warning("Falling back to pattern-based connection extraction")
            state["connections"] = self._fallback_pattern_connection_extraction(facts, target_name)
            logger.info(f"Fallback found {len(state['connections'])} connections")
            
            return state
    
    # ==============================================================================
    # HELPER METHODS (Add these after the main methods above)
    # ==============================================================================

    def _parse_json_array_response(self, content: str, target_name: str, response_type: str) -> List[Dict]:
        """
        Parse AI JSON array response with robust error handling.
        
        Handles various response formats:
        - Clean JSON arrays
        - JSON wrapped in markdown code blocks
        - Text containing JSON
        - Malformed JSON (returns empty list)
        """
        try:
            # Clean up response content
            cleaned = content.strip()
            
            # Remove markdown code blocks
            if "```json" in cleaned:
                match = re.search(r'```json\s*(\[.*?\])\s*```', cleaned, re.DOTALL)
                if match:
                    cleaned = match.group(1)
            elif "```" in cleaned:
                match = re.search(r'```\s*(\[.*?\])\s*```', cleaned, re.DOTALL)
                if match:
                    cleaned = match.group(1)
            
            # Try to find JSON array in text
            if not cleaned.startswith('['):
                match = re.search(r'\[.*?\]', cleaned, re.DOTALL)
                if match:
                    cleaned = match.group(0)
                else:
                    # No array found
                    logger.warning(f"No JSON array found in {response_type} response")
                    return []
            
            # Parse JSON
            result = json.loads(cleaned)
            
            if not isinstance(result, list):
                logger.warning(f"{response_type} response is not a list: {type(result)}")
                return []
            
            # Validate items
            valid_items = []
            for item in result:
                if isinstance(item, dict) and item:  # Non-empty dict
                    valid_items.append(item)
            
            logger.debug(f"Successfully parsed {len(valid_items)} {response_type} from AI response")
            return valid_items
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing failed for {response_type}: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error parsing {response_type}: {e}")
            return []


    def _fallback_pattern_risk_detection(self, facts: List, target_name: str) -> List[Dict]:
        """
        Fallback: Pattern-based risk detection.
        
        CORRECTED VERSION: Works with Fact dataclass objects
        
        Used when AI risk assessment fails.
        Provides basic risk identification using keyword matching.
        """
        risks = []
        
        # ✅ CORRECTED: Use attribute access for Fact dataclass
        # Check for legal-related facts
        legal_facts = [f for f in facts if getattr(f, 'category', '') == 'legal']
        if legal_facts:
            risks.append({
                "id": "risk_legal_1",
                "category": "legal",
                "severity": "medium",
                "description": f"Found {len(legal_facts)} legal-related facts requiring review",
                "confidence": 0.7,
                "impact_score": 6.0,
                "evidence": [getattr(f, 'content', '')[:100] + "..." for f in legal_facts[:2]],
                "detected_at": datetime.now().isoformat(),
                "source": "pattern_matching"
            })
        
        # Check for low financial transparency
        financial_facts = [f for f in facts if getattr(f, 'category', '') == 'financial']
        if len(financial_facts) < 2 and len(facts) > 20:
            risks.append({
                "id": "risk_transparency_1",
                "category": "financial",
                "severity": "low",
                "description": "Limited public financial information available",
                "confidence": 0.5,
                "impact_score": 4.0,
                "evidence": [f"Only {len(financial_facts)} financial facts discovered from {len(facts)} total facts"],
                "detected_at": datetime.now().isoformat(),
                "source": "pattern_matching"
            })
        
        # Check for risk keywords in facts
        risk_keywords = {
            'lawsuit': ('legal', 'medium', 7.0),
            'sued': ('legal', 'medium', 7.0),
            'investigation': ('legal', 'high', 8.0),
            'fraud': ('legal', 'critical', 9.0),
            'controversy': ('reputational', 'medium', 6.0),
            'scandal': ('reputational', 'high', 8.0),
            'conflict of interest': ('financial', 'high', 8.0),
            'insider trading': ('financial', 'critical', 9.0),
        }
        
        for fact in facts:
            content = getattr(fact, 'content', '').lower()
            for keyword, (category, severity, impact) in risk_keywords.items():
                if keyword in content:
                    risks.append({
                        "id": f"risk_keyword_{len(risks)}",
                        "category": category,
                        "severity": severity,
                        "description": f"Potential {category} risk: {keyword} mentioned",
                        "confidence": 0.6,
                        "impact_score": impact,
                        "evidence": [getattr(fact, 'content', '')[:150] + "..."],
                        "detected_at": datetime.now().isoformat(),
                        "source": "pattern_matching"
                    })
                    break  # Only flag once per fact
        
        # Deduplicate by description
        seen_descriptions = set()
        unique_risks = []
        for risk in risks:
            desc = risk.get("description", "")
            if desc not in seen_descriptions:
                seen_descriptions.add(desc)
                unique_risks.append(risk)
        
        return unique_risks[:10]  # Limit to 10 risks


    def _fallback_pattern_connection_extraction(self, facts: List, target_name: str) -> List[Dict]:
        """
        Fallback: Regex pattern-based connection extraction.
        
        CORRECTED VERSION: Works with Fact dataclass objects
        
        Used when AI connection mapping fails.
        Uses regex patterns to find entity relationships.
        """
        connections = []
        connection_id = 0
        
        # Define relationship patterns
        patterns = [
            # People relationships
            (r'succeeded\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s+)?[A-Z][a-z]+)', 'predecessor', 0.9),
            (r'replaced\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s+)?[A-Z][a-z]+)', 'predecessor', 0.9),
            (r'preceded\s+by\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s+)?[A-Z][a-z]+)', 'predecessor', 0.9),
            (r'worked\s+with\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s+)?[A-Z][a-z]+)', 'colleague', 0.7),
            
            # Organizations
            (r'worked\s+(?:at|for)\s+([A-Z][A-Za-z&\s]+?)(?:\s+(?:in|as|from)|\.|,|$)', 'employer', 0.8),
            (r'joined\s+([A-Z][A-Za-z&\s]+?)(?:\s+(?:in|as)|\.|,|$)', 'employer', 0.8),
            (r'(?:CEO|Chairman|President|VP)\s+of\s+([A-Z][A-Za-z&\s]+)', 'leadership', 0.9),
            (r'board\s+of\s+(?:directors\s+of\s+)?([A-Z][A-Za-z&\s]+)', 'board_member', 0.8),
            
            # Education
            (r'graduated\s+from\s+([A-Z][A-Za-z&\s]+(?:University|Institute|College|School))', 'education', 0.8),
            (r'attended\s+([A-Z][A-Za-z&\s]+(?:University|Institute|College|School))', 'education', 0.8),
            (r'(?:degree|MBA|MS|BS|PhD)\s+from\s+([A-Z][A-Za-z&\s]+)', 'education', 0.8),
            (r'studied\s+at\s+([A-Z][A-Za-z&\s]+(?:University|Institute|College|School))', 'education', 0.7),
        ]
        
        for fact in facts:
            # ✅ CORRECTED: Use attribute access for Fact dataclass
            content = getattr(fact, 'content', '')
            
            for pattern, rel_type, strength in patterns:
                matches = re.findall(pattern, content)
                
                for entity in matches:
                    # Clean up entity name
                    entity = entity.strip()
                    entity = re.sub(r'\s+(in|as|and|the|from)$', '', entity, flags=re.IGNORECASE)
                    entity = entity.strip()
                    
                    # Validate entity
                    if (entity and 
                        len(entity) > 2 and 
                        entity != target_name and
                        not entity.lower() in ['the', 'a', 'an', 'and', 'or']):
                        
                        connections.append({
                            "id": f"conn_{connection_id}",
                            "entity_1": target_name,
                            "entity_2": entity,
                            "relationship_type": rel_type,
                            "strength": strength,
                            "confidence": getattr(fact, 'confidence', 0.7),
                            "time_period": "Unknown",
                            "evidence": [content],
                            "source_urls": getattr(fact, 'source_urls', []),
                            "discovered_at": datetime.now().isoformat(),
                            "source": "pattern_matching"
                        })
                        connection_id += 1
        
        # Deduplicate connections
        seen = set()
        unique_connections = []
        for conn in connections:
            key = (conn["entity_1"], conn["entity_2"], conn["relationship_type"])
            if key not in seen:
                seen.add(key)
                unique_connections.append(conn)
        
        return unique_connections[:30]  # Limit to 30 connection
    
    
    async def _node_generate_report(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate final research report.
        
        Synthesizes all findings into structured output.
        """
        logger.info(f"Node: Generate Report")
        
        state["stage"] = "report_generation"
        state["completed_at"] = datetime.now()
        
        # Calculate summary statistics
        facts = state.get("facts", [])
        
        state["summary"] = {
            "total_facts": len(facts),
            "high_confidence_facts": len([f for f in facts if f.is_high_confidence()]),
            "verified_facts": len([f for f in facts if f.verified]),
            "risk_flags": len(state.get("risk_flags", [])),
            "connections": len(state.get("connections", [])),
            "coverage": self.strategy_engine.coverage.to_dict()
        }
        
        logger.info(f"Report generated: {state['summary']}")
        
        return state
    
    # ========================================================================
    # DECISION LOGIC
    # ========================================================================
    
    def _decide_continue_or_finish(self, state: Dict[str, Any]) -> str:
        """
        Decide whether to continue searching or move to verification.
        
        Decision criteria:
        1. Max iterations reached → verify
        2. Coverage adequate → verify
        3. No more queries → verify
        4. Sufficient facts → verify
        5. Otherwise → continue
        """
        iteration = state["iteration"]
        max_iter = state["max_iterations"]
        
        # Check iteration limit
        if iteration >= max_iter:
            logger.info(f"Max iterations ({max_iter}) reached")
            return "verify"
        
        # Check coverage
        if self.strategy_engine.is_coverage_adequate():
            logger.info("Coverage adequate, moving to verification")
            return "verify"
        
        # Check if we have queries left
        if not state.get("queries"):
            logger.info("No more queries, moving to verification")
            return "verify"
        
        # Check if we have enough facts
        if len(state.get("facts", [])) >= self.max_facts * 0.8:
            logger.info("Sufficient facts collected")
            return "verify"
        
        # Continue searching
        logger.info(f"Continuing search (iteration {iteration}/{max_iter})")
        return "continue"
    
    # ========================================================================
    # RESULT FORMATTING
    # ========================================================================
    
    def _format_results(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Format final results for output"""
        
        return {
            "target_name": state["target_name"],
            "facts": [f.to_dict() for f in state.get("facts", [])],
            "risk_flags": state.get("risk_flags", []),
            "connections": state.get("connections", []),
            "summary": state.get("summary", {}),
            "metadata": {
                "iterations": state["iteration"],
                "queries_executed": len(self.strategy_engine.executed_queries),
                "duration_seconds": (
                    state["completed_at"] - state["start_time"]
                ).total_seconds(),
                "coverage": self.strategy_engine.coverage.to_dict()
            }
        }


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

async def research(target_name: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convenience function for quick research.
    
    Example:
        >>> from src.core.workflow import research
        >>> result = await research("Sarah Chen")
        >>> print(result["summary"])
    """
    orchestrator = ResearchOrchestrator()
    return await orchestrator.research(target_name, context)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = ["ResearchOrchestrator", "research"]


# ============================================================================
# CLI TESTING
# ============================================================================

if __name__ == "__main__":
    """Quick test from command line"""
    import sys
    
    async def main():
        if len(sys.argv) < 2:
            print("Usage: python workflow.py <target_name>")
            sys.exit(1)
        
        target = sys.argv[1]
        
        print(f"🔍 Researching: {target}")
        print("=" * 60)
        
        result = await research(target)
        
        print("\n✅ Research Complete!")
        print("=" * 60)
        print(f"Facts: {result['summary']['total_facts']}")
        print(f"High-confidence: {result['summary']['high_confidence_facts']}")
        print(f"Risk flags: {result['summary']['risk_flags']}")
        print(f"Duration: {result['metadata']['duration_seconds']:.1f}s")
        
        print("\nTop Facts:")
        for i, fact in enumerate(result['facts'][:5], 1):
            print(f"{i}. [{fact['category']}] {fact['content']} ({fact['confidence']:.2f})")
    
    asyncio.run(main())