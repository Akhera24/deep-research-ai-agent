# Technical Architecture Document
## Deep Research AI Agent - Detailed Implementation Guide

**Version**: 1.0  
**Last Updated**: January 2026

---

## Table of Contents
1. [Core Architecture](#core-architecture)
2. [LangGraph Implementation](#langgraph-implementation)
3. [Agent System Design](#agent-system-design)
4. [Multi-Model Router](#multi-model-router)
5. [Search Strategy Engine](#search-strategy-engine)
6. [Database Schema](#database-schema)
7. [API Design](#api-design)
8. [Code Examples](#code-examples)

---

## 1. Core Architecture

### 1.1 Technology Decisions & Rationale

#### Why LangGraph?
```python
"""
LangGraph provides:
1. State management across multi-step workflows
2. Conditional branching based on agent outputs
3. Built-in checkpointing for error recovery
4. Visualization of agent workflows
5. Native LangChain integration

Alternative Considered: CrewAI
- Rejected: Less flexible state management, harder to customize
"""
```

#### Why FastAPI?
```python
"""
FastAPI advantages:
1. Async/await support for concurrent operations
2. Automatic OpenAPI documentation
3. Pydantic validation built-in
4. WebSocket support for real-time updates
5. High performance (comparable to Node.js/Go)
6. Type hints for better IDE support
"""
```

#### Why PostgreSQL + Redis?
```python
"""
PostgreSQL:
- JSONB support for flexible fact storage
- Full-text search capabilities
- Transaction support for consistency
- Mature, battle-tested

Redis:
- Sub-millisecond latency for caching
- Pub/sub for real-time updates
- Rate limiting implementation
- Session management
"""
```

---

## 2. LangGraph Implementation

### 2.1 State Definition

```python
# src/core/state_manager.py

from typing import TypedDict, List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

class ResearchStage(str, Enum):
    """Research workflow stages"""
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
    """Individual search result structure"""
    query: str
    url: str
    title: str
    snippet: str
    content: str
    source_reliability: float
    timestamp: datetime
    search_engine: str

class Fact(TypedDict):
    """Extracted fact structure"""
    id: str
    content: str
    category: str  # biographical, professional, financial, legal, etc.
    source_urls: List[str]
    confidence_score: float
    verification_status: str  # verified, unverified, conflicting
    extracted_at: datetime
    evidence: List[str]

class RiskFlag(TypedDict):
    """Risk assessment flag"""
    id: str
    category: str
    description: str
    severity: str  # low, medium, high, critical
    evidence: List[str]
    confidence: float
    impact_score: float

class Connection(TypedDict):
    """Entity connection/relationship"""
    entity_1: str
    entity_2: str
    relationship_type: str
    strength: float
    time_period: Optional[str]
    evidence: List[str]
    confidence: float

class ResearchState(TypedDict):
    """
    Main state object that flows through LangGraph workflow.
    
    Design Decisions:
    - Immutable updates: Each agent returns new state, doesn't modify
    - Rich metadata: Track every decision for debugging/auditing
    - Error accumulation: Don't fail fast, collect errors for analysis
    """
    # Input
    query: str
    target_entity: Dict[str, Any]
    research_parameters: Dict[str, Any]
    
    # Workflow control
    current_stage: ResearchStage
    iteration_count: int
    max_iterations: int
    should_continue: bool
    
    # Search management
    search_plan: List[str]
    executed_searches: List[SearchResult]
    pending_searches: List[str]
    search_coverage: Dict[str, float]  # topic -> coverage percentage
    
    # Extracted data
    facts: List[Fact]
    risk_flags: List[RiskFlag]
    connections: List[Connection]
    timeline: List[Dict[str, Any]]
    
    # Quality metrics
    confidence_scores: Dict[str, float]
    source_diversity_score: float
    coverage_score: float
    
    # Metadata
    start_time: datetime
    end_time: Optional[datetime]
    total_api_calls: int
    total_cost: float
    errors: List[Dict[str, Any]]
    
    # Agent communications
    agent_messages: List[Dict[str, Any]]
    last_agent: Optional[str]
```

### 2.2 LangGraph Workflow Definition

```python
# src/core/workflow.py

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import Literal
import logging

logger = logging.getLogger(__name__)

def create_research_workflow():
    """
    Creates the main research workflow using LangGraph.
    
    Architecture:
    - Directed graph with conditional transitions
    - Checkpointing for error recovery
    - Parallel execution where possible
    
    Design Decisions:
    1. Coordinator runs first to plan strategy
    2. Collection can happen in parallel
    3. Analysis is sequential (depends on collection)
    4. Risk assessment and connection mapping can be parallel
    5. Report generation is final step
    """
    
    # Initialize checkpoint saver for persistence
    checkpointer = SqliteSaver.from_conn_string("checkpoints.db")
    
    # Create state graph
    workflow = StateGraph(ResearchState)
    
    # Add nodes (agents)
    workflow.add_node("initialize", initialize_research)
    workflow.add_node("coordinator", research_coordinator_agent)
    workflow.add_node("collector", data_collection_agent)
    workflow.add_node("analyzer", analysis_verification_agent)
    workflow.add_node("risk_assessor", risk_assessment_agent)
    workflow.add_node("mapper", connection_mapping_agent)
    workflow.add_node("reporter", report_generation_agent)
    
    # Define entry point
    workflow.set_entry_point("initialize")
    
    # Add transitions
    workflow.add_edge("initialize", "coordinator")
    workflow.add_edge("coordinator", "collector")
    
    # Conditional edge: continue collecting or move to analysis
    workflow.add_conditional_edges(
        "collector",
        should_continue_collecting,
        {
            "continue": "collector",  # Loop back for more searches
            "analyze": "analyzer"      # Move to analysis phase
        }
    )
    
    workflow.add_edge("analyzer", "risk_assessor")
    
    # Parallel execution: risk assessment and connection mapping
    # Note: LangGraph doesn't have native parallel, so we handle in node
    workflow.add_edge("risk_assessor", "mapper")
    workflow.add_edge("mapper", "reporter")
    workflow.add_edge("reporter", END)
    
    # Compile with checkpointing
    app = workflow.compile(checkpointer=checkpointer)
    
    return app

def should_continue_collecting(
    state: ResearchState
) -> Literal["continue", "analyze"]:
    """
    Decide whether to continue data collection or move to analysis.
    
    Decision Logic:
    1. If iteration limit reached → analyze
    2. If coverage threshold met → analyze
    3. If no new information in last 3 searches → analyze
    4. If sufficient facts found → analyze
    5. Otherwise → continue
    
    This is a critical decision point that affects:
    - Research depth vs. cost
    - Time to completion
    - Information quality
    """
    
    # Check iteration limit
    if state["iteration_count"] >= state["max_iterations"]:
        logger.info(f"Max iterations ({state['max_iterations']}) reached")
        return "analyze"
    
    # Check coverage threshold
    avg_coverage = sum(state["search_coverage"].values()) / len(state["search_coverage"])
    if avg_coverage > 0.85:
        logger.info(f"Coverage threshold met: {avg_coverage:.2%}")
        return "analyze"
    
    # Check for diminishing returns
    recent_facts = [f for f in state["facts"] if f["extracted_at"] > 
                   datetime.now() - timedelta(minutes=2)]
    if len(recent_facts) < 2 and state["iteration_count"] > 5:
        logger.info("Diminishing returns detected")
        return "analyze"
    
    # Check if we have minimum facts
    if len(state["facts"]) > 30:
        logger.info(f"Sufficient facts found: {len(state['facts'])}")
        return "analyze"
    
    # Continue collecting
    logger.info(f"Continuing collection (iteration {state['iteration_count']})")
    return "continue"

# Agent implementations

def initialize_research(state: ResearchState) -> ResearchState:
    """
    Initialize research session with default values and logging.
    """
    logger.info(f"Initializing research for: {state['query']}")
    
    return {
        **state,
        "current_stage": ResearchStage.INITIALIZATION,
        "iteration_count": 0,
        "start_time": datetime.now(),
        "facts": [],
        "risk_flags": [],
        "connections": [],
        "search_coverage": {},
        "confidence_scores": {},
        "errors": [],
        "agent_messages": []
    }

def research_coordinator_agent(state: ResearchState) -> ResearchState:
    """
    High-level research planning and strategy.
    Uses Claude Opus 4 for superior reasoning.
    
    Responsibilities:
    1. Analyze initial query
    2. Generate search strategy
    3. Prioritize research areas
    4. Set quality thresholds
    """
    from src.agents.coordinator import ResearchCoordinator
    
    logger.info("Running research coordinator")
    coordinator = ResearchCoordinator(model="claude-opus-4-20250514")
    
    try:
        # Generate comprehensive search plan
        plan = coordinator.generate_search_plan(
            query=state["query"],
            target_entity=state["target_entity"],
            max_searches=state["max_iterations"]
        )
        
        return {
            **state,
            "current_stage": ResearchStage.STRATEGY_PLANNING,
            "search_plan": plan["queries"],
            "search_coverage": {topic: 0.0 for topic in plan["topics"]},
            "pending_searches": plan["queries"][:5],  # Start with top 5
            "agent_messages": state["agent_messages"] + [{
                "agent": "coordinator",
                "action": "strategy_planning",
                "output": plan
            }]
        }
    except Exception as e:
        logger.error(f"Coordinator error: {e}")
        return {
            **state,
            "errors": state["errors"] + [{
                "agent": "coordinator",
                "error": str(e),
                "timestamp": datetime.now()
            }]
        }

def data_collection_agent(state: ResearchState) -> ResearchState:
    """
    Execute searches and collect data from multiple sources.
    Uses Gemini 2.5 for efficient multi-source processing.
    """
    from src.agents.collector import DataCollector
    
    logger.info(f"Running data collector (iteration {state['iteration_count']})")
    collector = DataCollector(model="gemini-2.5-pro")
    
    # Get next batch of searches
    next_searches = state["pending_searches"][:3]  # Parallel batch of 3
    
    # Execute searches
    results = []
    for query in next_searches:
        try:
            search_results = collector.search(query)
            results.extend(search_results)
        except Exception as e:
            logger.error(f"Search error for '{query}': {e}")
    
    # Update state
    return {
        **state,
        "current_stage": ResearchStage.DATA_COLLECTION,
        "iteration_count": state["iteration_count"] + 1,
        "executed_searches": state["executed_searches"] + results,
        "pending_searches": state["pending_searches"][3:],  # Remove executed
    }

# Similar implementations for other agents...
```

---

## 3. Agent System Design

### 3.1 Base Agent Class

```python
# src/agents/base_agent.py

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

@dataclass
class AgentConfig:
    """Configuration for agent behavior"""
    model_name: str
    temperature: float = 0.3
    max_tokens: int = 4000
    timeout: int = 30
    retry_attempts: int = 3
    fallback_model: Optional[str] = None

class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    
    Design Principles:
    1. Single Responsibility: Each agent has one clear purpose
    2. Fail Gracefully: Always return partial results on error
    3. Observable: Comprehensive logging for debugging
    4. Testable: Mock-friendly interfaces
    5. Composable: Agents can call other agents
    
    FAANG Quality Standards:
    - Type hints on all methods
    - Docstrings with examples
    - Error handling with retries
    - Metrics collection
    - Request/response logging
    """
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.model_client = self._initialize_model()
        self.call_count = 0
        self.error_count = 0
        self.total_cost = 0.0
    
    @abstractmethod
    def _initialize_model(self):
        """Initialize the AI model client"""
        pass
    
    @abstractmethod
    def execute(self, state: ResearchState) -> Dict[str, Any]:
        """Main execution method - must be implemented by subclass"""
        pass
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def call_model(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Call the AI model with retry logic and error handling.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt for model behavior
            **kwargs: Additional model parameters
            
        Returns:
            Model response text
            
        Raises:
            ModelAPIError: If all retries fail
            
        Example:
            >>> agent = SomeAgent(config)
            >>> response = agent.call_model("Analyze this fact: ...")
        """
        try:
            self.call_count += 1
            
            logger.debug(f"Model call #{self.call_count}: {prompt[:100]}...")
            
            response = self._make_api_call(
                prompt=prompt,
                system_prompt=system_prompt,
                **kwargs
            )
            
            # Track cost (estimated)
            self.total_cost += self._estimate_cost(prompt, response)
            
            return response
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Model call failed: {e}")
            
            # Try fallback model if configured
            if self.config.fallback_model:
                logger.info(f"Trying fallback model: {self.config.fallback_model}")
                return self._call_fallback(prompt, system_prompt, **kwargs)
            
            raise
    
    @abstractmethod
    def _make_api_call(
        self,
        prompt: str,
        system_prompt: Optional[str],
        **kwargs
    ) -> str:
        """Actual API call - implemented per model type"""
        pass
    
    def _estimate_cost(self, prompt: str, response: str) -> float:
        """Estimate API call cost based on token count"""
        # Rough estimation: 4 chars = 1 token
        input_tokens = len(prompt) / 4
        output_tokens = len(response) / 4
        
        # Model-specific pricing (example for Claude)
        if "claude" in self.config.model_name.lower():
            input_cost = input_tokens * 0.000015  # $15 per 1M tokens
            output_cost = output_tokens * 0.000075  # $75 per 1M tokens
        else:
            input_cost = output_tokens = 0.00001
        
        return input_cost + output_cost
    
    def get_metrics(self) -> Dict[str, Any]:
        """Return agent performance metrics"""
        return {
            "agent_type": self.__class__.__name__,
            "model": self.config.model_name,
            "total_calls": self.call_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.call_count, 1),
            "total_cost": self.total_cost
        }
```

### 3.2 Research Coordinator Agent

```python
# src/agents/coordinator.py

from typing import Dict, List, Any
from .base_agent import BaseAgent, AgentConfig
from anthropic import Anthropic
import json

class ResearchCoordinator(BaseAgent):
    """
    Master coordinator that plans research strategy.
    
    Uses Claude Opus 4 for superior strategic thinking.
    
    Responsibilities:
    1. Analyze initial query and extract target entity
    2. Generate comprehensive search strategy
    3. Prioritize research areas
    4. Define quality thresholds
    5. Adapt strategy based on findings
    """
    
    def _initialize_model(self):
        return Anthropic(api_key=self.config.api_key)
    
    def execute(self, state: ResearchState) -> Dict[str, Any]:
        """Generate comprehensive research plan"""
        
        # Build strategic prompt
        system_prompt = """You are a master research strategist with expertise in:
- Due diligence and background checks
- Open-source intelligence (OSINT)
- Risk assessment and red flag detection
- Investigative journalism techniques

Your goal is to design a comprehensive research strategy that will uncover:
1. Biographical and professional facts
2. Financial connections and interests
3. Legal and regulatory issues
4. Hidden relationships and patterns
5. Potential risks and red flags

Think deeply about what information exists, where to find it, and how to verify it."""

        user_prompt = f"""
Design a comprehensive research strategy for: {state['query']}

Target Entity: {json.dumps(state['target_entity'], indent=2)}

Generate a search plan that:
1. Starts with basic biographical information
2. Progressively digs deeper into specific areas
3. Cross-references multiple sources
4. Looks for non-obvious connections
5. Identifies potential risks

Provide your response as JSON with this structure:
{{
    "entity_analysis": {{
        "name": "Full name",
        "entity_type": "person|company|organization",
        "known_facts": ["fact1", "fact2"],
        "information_gaps": ["gap1", "gap2"]
    }},
    "research_priorities": [
        {{
            "area": "biographical",
            "priority": 1,
            "rationale": "why important",
            "expected_difficulty": "easy|medium|hard"
        }}
    ],
    "search_queries": [
        {{
            "query": "search query text",
            "purpose": "what we're looking for",
            "priority": 1,
            "estimated_depth": 2
        }}
    ],
    "topics": ["topic1", "topic2"],
    "risk_areas": ["area1", "area2"],
    "expected_challenges": ["challenge1", "challenge2"]
}}
"""
        
        # Call model
        response = self.call_model(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.3
        )
        
        # Parse JSON response
        try:
            plan = json.loads(response)
            return self._process_plan(plan)
        except json.JSONDecodeError:
            # Fallback: extract queries manually
            return self._fallback_plan(state['query'])
    
    def _process_plan(self, plan: Dict) -> Dict[str, Any]:
        """Process and validate the research plan"""
        
        # Extract and prioritize queries
        queries = sorted(
            plan['search_queries'],
            key=lambda x: x['priority']
        )
        
        return {
            "queries": [q['query'] for q in queries],
            "topics": plan['topics'],
            "priorities": plan['research_priorities'],
            "risk_areas": plan.get('risk_areas', []),
            "entity_analysis": plan['entity_analysis']
        }
    
    def _fallback_plan(self, query: str) -> Dict[str, Any]:
        """Simple fallback if JSON parsing fails"""
        return {
            "queries": [
                f"{query}",
                f"{query} biography",
                f"{query} professional history",
                f"{query} education background",
                f"{query} controversies",
                f"{query} legal issues",
                f"{query} business connections",
                f"{query} social media",
                f"{query} news articles",
                f"{query} public records"
            ],
            "topics": ["biographical", "professional", "legal", "connections"],
            "priorities": [],
            "risk_areas": ["legal", "financial", "reputational"],
            "entity_analysis": {}
        }
    
    def adapt_strategy(
        self,
        current_state: ResearchState,
        recent_findings: List[Dict]
    ) -> List[str]:
        """
        Adapt search strategy based on what's been found.
        
        This is key to "consecutive search" - each round informs the next.
        """
        
        prompt = f"""
Based on our research so far, suggest 5 new search queries to dig deeper.

Current findings summary:
- {len(current_state['facts'])} facts discovered
- {len(current_state['risk_flags'])} risk flags identified
- Coverage: {current_state['search_coverage']}

Recent discoveries:
{json.dumps(recent_findings, indent=2)}

What should we investigate next? Focus on:
1. Following up on interesting findings
2. Filling information gaps
3. Cross-referencing conflicting information
4. Exploring connections that were mentioned

Return JSON array of search queries: ["query1", "query2", ...]
"""
        
        response = self.call_model(prompt)
        
        try:
            return json.loads(response)
        except:
            # Fallback: generate based on gaps
            return self._generate_followup_queries(current_state)
```

### 3.3 Data Collection Agent

```python
# src/agents/collector.py

from typing import List, Dict, Any
from .base_agent import BaseAgent, AgentConfig
import google.generativeai as genai
from src.search.search_executor import SearchExecutor
import asyncio
from concurrent.futures import ThreadPoolExecutor

class DataCollector(BaseAgent):
    """
    Executes searches and collects data from multiple sources.
    
    Uses Gemini 2.5 for:
    - Large context window (1M tokens)
    - Efficient document processing
    - Multi-source synthesis
    
    Design Decisions:
    - Parallel search execution (3-5 concurrent)
    - Multiple search engines for redundancy
    - Content extraction and cleaning
    - Deduplication at collection time
    """
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.search_executor = SearchExecutor()
        self.executor = ThreadPoolExecutor(max_workers=5)
    
    def _initialize_model(self):
        genai.configure(api_key=self.config.api_key)
        return genai.GenerativeModel(self.config.model_name)
    
    def execute(self, state: ResearchState) -> Dict[str, Any]:
        """Execute multiple searches in parallel"""
        
        # Get next batch of searches
        queries = state['pending_searches'][:5]
        
        # Execute in parallel
        results = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(self.search_and_extract, query)
                for query in queries
            ]
            
            for future in futures:
                try:
                    result = future.result(timeout=30)
                    results.extend(result)
                except Exception as e:
                    logger.error(f"Search failed: {e}")
        
        return {
            "search_results": results,
            "queries_executed": queries
        }
    
    def search_and_extract(self, query: str) -> List[Dict[str, Any]]:
        """
        Execute single search and extract relevant content.
        
        Process:
        1. Search using multiple engines
        2. Fetch full page content
        3. Extract relevant text
        4. Clean and structure
        5. Assess source reliability
        """
        
        # Execute search
        raw_results = self.search_executor.search(query)
        
        # Process each result
        processed = []
        for result in raw_results[:5]:  # Top 5 results
            try:
                # Fetch full content
                content = self.search_executor.fetch_url(result['url'])
                
                # Extract relevant content using Gemini
                extracted = self.extract_relevant_content(
                    content=content,
                    query=query,
                    source_url=result['url']
                )
                
                processed.append({
                    "query": query,
                    "url": result['url'],
                    "title": result['title'],
                    "snippet": result['snippet'],
                    "content": extracted['content'],
                    "relevance_score": extracted['relevance'],
                    "source_reliability": self.assess_source(result['url']),
                    "timestamp": datetime.now(),
                    "search_engine": result['engine']
                })
                
            except Exception as e:
                logger.error(f"Error processing {result['url']}: {e}")
                continue
        
        return processed
    
    def extract_relevant_content(
        self,
        content: str,
        query: str,
        source_url: str
    ) -> Dict[str, Any]:
        """
        Use Gemini to extract only relevant content from page.
        
        This reduces token usage and improves fact extraction quality.
        """
        
        prompt = f"""
Extract ONLY the content relevant to: {query}

Source: {source_url}

Full content:
{content[:50000]}  # Truncate to fit in context

Return JSON:
{{
    "content": "extracted relevant text",
    "relevance": 0.0-1.0,
    "key_points": ["point1", "point2"],
    "entities_mentioned": ["entity1", "entity2"]
}}
"""
        
        response = self.call_model(prompt)
        
        try:
            return json.loads(response)
        except:
            # Fallback: return first 5000 chars
            return {
                "content": content[:5000],
                "relevance": 0.5,
                "key_points": [],
                "entities_mentioned": []
            }
    
    def assess_source(self, url: str) -> float:
        """
        Assess source reliability based on domain.
        
        Tier 1 (0.9-1.0): .gov, .edu, official records
        Tier 2 (0.7-0.89): Major news, verified databases
        Tier 3 (0.5-0.69): Industry sites, smaller news
        Tier 4 (0.3-0.49): Blogs, forums
        Tier 5 (0-0.29): Unknown/questionable
        """
        
        from urllib.parse import urlparse
        domain = urlparse(url).netloc.lower()
        
        # Tier 1
        tier1_domains = ['.gov', '.edu', 'sec.gov', 'courts.gov']
        if any(d in domain for d in tier1_domains):
            return 0.95
        
        # Tier 2
        tier2_domains = [
            'reuters.com', 'bloomberg.com', 'wsj.com',
            'nytimes.com', 'ft.com', 'economist.com'
        ]
        if any(d in domain for d in tier2_domains):
            return 0.85
        
        # Tier 3
        tier3_domains = [
            'techcrunch.com', 'forbes.com', 'businessinsider.com',
            'linkedin.com', 'crunchbase.com'
        ]
        if any(d in domain for d in tier3_domains):
            return 0.65
        
        # Tier 4
        tier4_domains = ['medium.com', 'wordpress.com', 'blogspot.com']
        if any(d in domain for d in tier4_domains):
            return 0.40
        
        # Tier 5 - unknown
        return 0.25
```

---

## 4. Multi-Model Router

```python
# src/models/router.py

from typing import Dict, Any, Optional, List
from enum import Enum
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class TaskType(str, Enum):
    """Different types of tasks that require different models"""
    STRATEGY_PLANNING = "strategy_planning"
    DOCUMENT_PROCESSING = "document_processing"
    STRUCTURED_OUTPUT = "structured_output"
    RISK_ASSESSMENT = "risk_assessment"
    FACT_EXTRACTION = "fact_extraction"
    CONNECTION_ANALYSIS = "connection_analysis"
    VERIFICATION = "verification"

@dataclass
class ModelCapability:
    """Model capabilities and characteristics"""
    name: str
    strengths: List[str]
    context_window: int  # tokens
    cost_per_1k_input: float
    cost_per_1k_output: float
    rate_limit: int  # requests per minute
    reliability_score: float  # 0-1

class MultiModelRouter:
    """
    Intelligently routes requests to optimal AI model.
    
    Design Principles:
    1. Task-based routing: Different tasks need different capabilities
    2. Fallback chains: Always have backup options
    3. Cost optimization: Use cheaper models when appropriate
    4. Rate limit aware: Automatically failover
    5. Performance tracking: Learn from results
    
    Features:
    - Circuit breaker pattern for failing models
    - Latency tracking for optimization
    - Cost monitoring and alerts
    - A/B testing framework ready
    """
    
    def __init__(self):
        self.models = self._initialize_models()
        self.task_routing = self._define_task_routing()
        self.call_counts = {model: 0 for model in self.models}
        self.error_counts = {model: 0 for model in self.models}
        self.total_cost = 0.0
    
    def _initialize_models(self) -> Dict[str, ModelCapability]:
        """Define capabilities of each model"""
        
        return {
            "claude-opus-4": ModelCapability(
                name="claude-opus-4-20250514",
                strengths=[
                    "strategic_thinking",
                    "nuanced_analysis",
                    "ethical_reasoning",
                    "complex_instructions"
                ],
                context_window=200000,
                cost_per_1k_input=0.015,
                cost_per_1k_output=0.075,
                rate_limit=50,
                reliability_score=0.98
            ),
            "gemini-2.5": ModelCapability(
                name="gemini-2.5-pro",
                strengths=[
                    "large_context",
                    "multimodal",
                    "document_processing",
                    "fast_inference"
                ],
                context_window=1000000,
                cost_per_1k_input=0.005,
                cost_per_1k_output=0.015,
                rate_limit=360,
                reliability_score=0.95
            ),
            "gpt-4.1": ModelCapability(
                name="gpt-4-turbo-preview",
                strengths=[
                    "structured_output",
                    "function_calling",
                    "json_mode",
                    "reliable"
                ],
                context_window=128000,
                cost_per_1k_input=0.01,
                cost_per_1k_output=0.03,
                rate_limit=500,
                reliability_score=0.97
            )
        }
    
    def _define_task_routing(self) -> Dict[TaskType, Dict[str, Any]]:
        """
        Define optimal model for each task type.
        
        Based on extensive testing and cost-benefit analysis.
        """
        
        return {
            TaskType.STRATEGY_PLANNING: {
                "primary": "claude-opus-4",
                "fallback": "gpt-4.1",
                "rationale": "Claude excels at strategic thinking"
            },
            TaskType.DOCUMENT_PROCESSING: {
                "primary": "gemini-2.5",
                "fallback": "claude-opus-4",
                "rationale": "Gemini has 1M context window"
            },
            TaskType.STRUCTURED_OUTPUT: {
                "primary": "gpt-4.1",
                "fallback": "claude-opus-4",
                "rationale": "GPT-4 has reliable JSON mode"
            },
            TaskType.RISK_ASSESSMENT: {
                "primary": "claude-opus-4",
                "fallback": "gpt-4.1",
                "rationale": "Claude best at nuanced risk analysis"
            },
            TaskType.FACT_EXTRACTION: {
                "primary": "gemini-2.5",
                "fallback": "gpt-4.1",
                "rationale": "Gemini efficient for large documents"
            },
            TaskType.CONNECTION_ANALYSIS: {
                "primary": "claude-opus-4",
                "fallback": "gemini-2.5",
                "rationale": "Claude best at finding subtle connections"
            },
            TaskType.VERIFICATION: {
                "primary": "gpt-4.1",
                "fallback": "claude-opus-4",
                "rationale": "GPT-4 good at systematic checking"
            }
        }
    
    def route(
        self,
        task_type: TaskType,
        prompt: str,
        prefer_cost: bool = False,
        force_model: Optional[str] = None
    ) -> str:
        """
        Route request to optimal model.
        
        Args:
            task_type: Type of task to perform
            prompt: The actual prompt
            prefer_cost: If True, prefer cheaper models
            force_model: Override routing logic
            
        Returns:
            Model name to use
            
        Example:
            >>> router = MultiModelRouter()
            >>> model = router.route(TaskType.STRATEGY_PLANNING, prompt)
            >>> # Returns: "claude-opus-4"
        """
        
        if force_model:
            return force_model
        
        # Get routing config
        config = self.task_routing[task_type]
        primary = config["primary"]
        fallback = config["fallback"]
        
        # Check if primary model is available
        if self._is_model_available(primary):
            return primary
        
        logger.warning(f"{primary} unavailable, using fallback: {fallback}")
        return fallback
    
    def _is_model_available(self, model_key: str) -> bool:
        """
        Check if model is currently available.
        
        Considers:
        - Rate limits
        - Error rate (circuit breaker)
        - Service status
        """
        
        # Check error rate (circuit breaker)
        if model_key in self.error_counts and model_key in self.call_counts:
            error_rate = self.error_counts[model_key] / max(self.call_counts[model_key], 1)
            if error_rate > 0.5:  # >50% errors
                logger.error(f"{model_key} circuit breaker opened")
                return False
        
        # Check rate limit (simplified - should use Redis)
        model = self.models[model_key]
        current_rpm = self.call_counts.get(model_key, 0)
        if current_rpm >= model.rate_limit:
            logger.warning(f"{model_key} rate limit reached")
            return False
        
        return True
    
    def record_call(
        self,
        model_key: str,
        success: bool,
        cost: float
    ):
        """Record call for metrics and circuit breaking"""
        
        self.call_counts[model_key] += 1
        if not success:
            self.error_counts[model_key] += 1
        self.total_cost += cost
    
    def get_metrics(self) -> Dict[str, Any]:
        """Return routing metrics"""
        
        return {
            "total_cost": self.total_cost,
            "call_counts": self.call_counts,
            "error_counts": self.error_counts,
            "error_rates": {
                model: self.error_counts[model] / max(self.call_counts[model], 1)
                for model in self.models
            }
        }
```

---

## 5. Search Strategy Engine

```python
# src/search/strategy.py

from typing import List, Dict, Any, Set
from dataclasses import dataclass
import re

@dataclass
class SearchQuery:
    """Structured search query"""
    text: str
    purpose: str
    priority: int
    expected_depth: int
    category: str
    dependencies: List[str]  # Queries that should run first

class SearchStrategyEngine:
    """
    Generates and refines search queries using consecutive search strategy.
    
    Key Innovation: Dynamic query refinement based on findings.
    
    Algorithm:
    1. Start with broad queries (depth 1)
    2. Extract entities and topics from results
    3. Generate follow-up queries targeting gaps
    4. Increase specificity progressively
    5. Stop when diminishing returns
    
    Design Decisions:
    - Query templates for common patterns
    - Entity extraction for follow-ups
    - Coverage tracking to avoid redundancy
    - Depth balancing (don't get stuck in rabbit holes)
    """
    
    def __init__(self):
        self.executed_queries: Set[str] = set()
        self.entity_mentions: Dict[str, int] = {}
        self.coverage_map: Dict[str, float] = {}
        
        # Query templates for different depths
        self.templates = {
            1: [  # Depth 1: Basic biographical
                "{name}",
                "{name} biography",
                "{name} background",
                "{name} profile"
            ],
            2: [  # Depth 2: Professional
                "{name} career history",
                "{name} employment",
                "{name} professional experience",
                "{name} education"
            ],
            3: [  # Depth 3: Connections
                "{name} business partners",
                "{name} board positions",
                "{name} investments",
                "{name} affiliations"
            ],
            4: [  # Depth 4: Issues
                "{name} controversy",
                "{name} lawsuit",
                "{name} legal issues",
                "{name} regulatory"
            ],
            5: [  # Depth 5: Deep dive
                "{name} {entity} relationship",
                "{name} {company} connection",
                "{name} {topic} involvement"
            ]
        }
    
    def generate_initial_queries(
        self,
        target_name: str,
        context: Dict[str, Any]
    ) -> List[SearchQuery]:
        """
        Generate initial set of queries covering all major areas.
        
        Strategy:
        - Start broad for context
        - Cover all risk categories
        - Prepare for follow-up queries
        """
        
        queries = []
        
        # Depth 1: Basic info
        queries.append(SearchQuery(
            text=target_name,
            purpose="Basic information and context",
            priority=1,
            expected_depth=1,
            category="biographical",
            dependencies=[]
        ))
        
        queries.append(SearchQuery(
            text=f'"{target_name}" biography',
            purpose="Comprehensive biography",
            priority=1,
            expected_depth=1,
            category="biographical",
            dependencies=[]
        ))
        
        # Depth 2: Professional
        queries.append(SearchQuery(
            text=f'"{target_name}" career history',
            purpose="Professional background",
            priority=2,
            expected_depth=2,
            category="professional",
            dependencies=[]
        ))
        
        queries.append(SearchQuery(
            text=f'"{target_name}" linkedin',
            purpose="Professional profile",
            priority=2,
            expected_depth=2,
            category="professional",
            dependencies=[]
        ))
        
        # Depth 3: Connections
        queries.append(SearchQuery(
            text=f'"{target_name}" business partners',
            purpose="Business relationships",
            priority=3,
            expected_depth=3,
            category="connections",
            dependencies=[]
        ))
        
        # Depth 3-4: Risk areas
        queries.append(SearchQuery(
            text=f'"{target_name}" controversy OR lawsuit OR scandal',
            purpose="Identify potential issues",
            priority=2,
            expected_depth=3,
            category="risk",
            dependencies=[]
        ))
        
        queries.append(SearchQuery(
            text=f'"{target_name}" SEC OR regulatory',
            purpose="Regulatory issues",
            priority=3,
            expected_depth=3,
            category="legal",
            dependencies=[]
        ))
        
        # Depth 2-3: Financial
        queries.append(SearchQuery(
            text=f'"{target_name}" investment OR investor',
            purpose="Financial activities",
            priority=3,
            expected_depth=3,
            category="financial",
            dependencies=[]
        ))
        
        return sorted(queries, key=lambda q: q.priority)
    
    def refine_based_on_findings(
        self,
        findings: List[Dict[str, Any]],
        current_coverage: Dict[str, float]
    ) -> List[SearchQuery]:
        """
        Generate follow-up queries based on what's been found.
        
        This is the heart of "consecutive search strategy".
        
        Logic:
        1. Extract entities mentioned in findings
        2. Identify information gaps
        3. Generate targeted follow-up queries
        4. Prioritize by potential value
        """
        
        # Extract entities mentioned
        entities = self._extract_entities(findings)
        
        # Identify gaps
        gaps = self._identify_gaps(current_coverage)
        
        # Generate follow-ups
        follow_ups = []
        
        # Follow up on frequently mentioned entities
        for entity, count in sorted(
            entities.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]:  # Top 5 entities
            
            follow_ups.append(SearchQuery(
                text=f'"{target_name}" AND "{entity}"',
                purpose=f"Explore connection to {entity}",
                priority=4,
                expected_depth=4,
                category="connections",
                dependencies=[]
            ))
        
        # Fill coverage gaps
        for gap_category in gaps:
            if gap_category == "financial":
                follow_ups.append(SearchQuery(
                    text=f'"{target_name}" financial OR assets OR property',
                    purpose="Financial information",
                    priority=3,
                    expected_depth=3,
                    category="financial",
                    dependencies=[]
                ))
            
            elif gap_category == "legal":
                follow_ups.append(SearchQuery(
                    text=f'"{target_name}" court OR legal OR case',
                    purpose="Legal records",
                    priority=3,
                    expected_depth=3,
                    category="legal",
                    dependencies=[]
                ))
        
        return follow_ups
    
    def _extract_entities(
        self,
        findings: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """
        Extract named entities from findings.
        
        Simple implementation - in production, use NER model.
        """
        
        entities = {}
        
        for finding in findings:
            content = finding.get('content', '')
            
            # Simple pattern: Capitalized words (names)
            potential_entities = re.findall(
                r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',
                content
            )
            
            for entity in potential_entities:
                if len(entity) > 3:  # Filter noise
                    entities[entity] = entities.get(entity, 0) + 1
        
        return entities
    
    def _identify_gaps(
        self,
        current_coverage: Dict[str, float]
    ) -> List[str]:
        """Identify categories with low coverage"""
        
        gaps = []
        target_coverage = 0.7
        
        for category, coverage in current_coverage.items():
            if coverage < target_coverage:
                gaps.append(category)
        
        return gaps
    
    def deduplicate_queries(
        self,
        queries: List[SearchQuery]
    ) -> List[SearchQuery]:
        """
        Remove duplicate or very similar queries.
        
        Uses simple string similarity for now.
        In production, use embeddings for semantic similarity.
        """
        
        unique = []
        seen = set()
        
        for query in queries:
            # Normalize
            normalized = query.text.lower().strip()
            
            # Check if too similar to existing
            is_duplicate = False
            for seen_query in seen:
                similarity = self._string_similarity(normalized, seen_query)
                if similarity > 0.9:  # 90% similar
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique.append(query)
                seen.add(normalized)
        
        return unique
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        """Simple string similarity using Jaccard index"""
        
        set1 = set(s1.split())
        set2 = set(s2.split())
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
```

---
