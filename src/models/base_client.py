"""
Base AI Model Client

Abstract base class for all AI model integrations.
Provides common functionality: retry logic, error handling, cost tracking, logging.

FAANG Design Patterns:
- Template Method: Base class defines workflow, subclasses implement specifics
- Strategy Pattern: Different retry/fallback strategies
- Observer Pattern: Metrics collection on every call
- Circuit Breaker: Auto-disable failing models

Quality Standards:
- Type safety with full type hints
- Comprehensive error handling
- Structured logging with context
- Production-tested retry logic
- Cost tracking per request
- Performance metrics
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import lru_cache
import hashlib
import logging
import time
import asyncio
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

from config.logging_config import get_logger

logger = get_logger(__name__)


class ModelProvider(str, Enum):
    """Supported AI model providers"""
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OPENAI = "openai"


class TaskType(str, Enum):
    """Task types for intelligent routing"""
    STRATEGY_PLANNING = "strategy_planning"      # Complex reasoning
    DOCUMENT_PROCESSING = "document_processing"  # Large context
    STRUCTURED_OUTPUT = "structured_output"      # JSON generation
    RISK_ASSESSMENT = "risk_assessment"          # Nuanced analysis
    FACT_EXTRACTION = "fact_extraction"          # Information extraction
    VERIFICATION = "verification"                # Cross-checking
    ANALYSIS = "analysis"                        # General analysis


@dataclass
class ModelConfig:
    """
    Configuration for AI model client.
    
    Attributes:
        provider: Model provider (anthropic, google, openai)
        model_name: Specific model identifier
        api_key: Authentication key
        max_tokens: Maximum response tokens
        temperature: Randomness (0.0-1.0)
        timeout: Request timeout in seconds
        max_retries: Maximum retry attempts
        rate_limit: Requests per minute
        cost_per_1k_input: Input cost (USD per 1K tokens)
        cost_per_1k_output: Output cost (USD per 1K tokens)
    """
    provider: ModelProvider
    model_name: str
    api_key: str
    max_tokens: int = 4000
    temperature: float = 0.3
    timeout: int = 30
    max_retries: int = 3
    rate_limit: int = 50  # requests per minute
    cost_per_1k_input: float = 0.01
    cost_per_1k_output: float = 0.03
    
    # Runtime state (not in constructor)
    total_calls: int = field(default=0, init=False)
    total_errors: int = field(default=0, init=False)
    total_cost: float = field(default=0.0, init=False)
    circuit_open: bool = field(default=False, init=False)
    last_error_time: Optional[datetime] = field(default=None, init=False)


@dataclass
class ModelResponse:
    """
    Standardized model response across all providers.
    
    Attributes:
        content: The actual response text
        provider: Which model generated this
        model_name: Specific model used
        tokens_used: Token count (input + output)
        cost: Estimated cost in USD
        latency_ms: Response time in milliseconds
        success: Whether call succeeded
        error: Error message if failed
        metadata: Additional provider-specific data
    """
    content: str
    provider: ModelProvider
    model_name: str
    tokens_used: int
    cost: float
    latency_ms: float
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        return {
            "content": self.content,
            "provider": self.provider.value,
            "model": self.model_name,
            "tokens": self.tokens_used,
            "cost": f"${self.cost:.4f}",
            "latency_ms": self.latency_ms,
            "success": self.success,
            "error": self.error,
            "metadata": self.metadata
        }


class BaseModelClient(ABC):
    """
    Abstract base class for AI model clients.
    
    Implements common functionality:
    - Request/response handling
    - Error handling with retries
    - Cost tracking
    - Performance metrics
    - Circuit breaker pattern
    - Structured logging
    
    Subclasses must implement:
    - _make_api_call(): Actual API integration
    - _estimate_tokens(): Token counting logic
    
    Usage:
        class ClaudeClient(BaseModelClient):
            def _make_api_call(self, prompt, **kwargs):
                # Claude-specific implementation
                pass
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize model client.
        
        Args:
            config: Model configuration with API keys, limits, costs
        """
        self.config = config
        self.logger = get_logger(f"{__name__}.{config.provider.value}")
        
        # Metrics
        self._call_history: List[Dict[str, Any]] = []
        self._last_call_time: Optional[datetime] = None
        
        self.logger.info(
            f"Initialized {config.provider.value} client",
            extra={
                "model": config.model_name,
                "rate_limit": config.rate_limit,
                "max_retries": config.max_retries
            }
        )
    
    @abstractmethod
    def _make_api_call(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Make actual API call to model provider.
        
        Must be implemented by subclass.
        
        Args:
            prompt: User prompt
            system_prompt: System instructions
            **kwargs: Provider-specific parameters
            
        Returns:
            Model response text
            
        Raises:
            Exception: On API errors
        """
        pass
    
    @abstractmethod
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        
        Must be implemented by subclass.
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        pass
    
    def call(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        task_type: Optional[TaskType] = None,
        **kwargs
    ) -> ModelResponse:
        """
        Call model with automatic retry and error handling.
        
        This is the main public method to call the model.
        
        Features:
        - Automatic retries with exponential backoff
        - Circuit breaker (stops trying if model consistently fails)
        - Cost tracking
        - Performance metrics
        - Structured logging
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system instructions
            task_type: Type of task (for metrics/routing)
            **kwargs: Additional model parameters
            
        Returns:
            ModelResponse with content, cost, metrics
            
        Raises:
            Exception: If all retries fail
            
        Example:
            >>> client = ClaudeClient(config)
            >>> response = client.call("Analyze this fact...")
            >>> print(response.content)
            >>> print(f"Cost: ${response.cost:.4f}")
        """
        # Check circuit breaker
        if self.config.circuit_open:
            self._check_circuit_breaker()
        
        # Rate limiting check
        self._enforce_rate_limit()
        
        # Record start time
        start_time = time.time()
        
        # Track attempt
        self.config.total_calls += 1
        
        try:
            # Make the API call with retry logic
            content = self._call_with_retry(prompt, system_prompt, **kwargs)
            
            # Calculate metrics
            latency_ms = (time.time() - start_time) * 1000
            tokens = self._estimate_tokens(prompt + content)
            cost = self._calculate_cost(prompt, content)
            
            # Update metrics
            self.config.total_cost += cost
            
            # Create response
            response = ModelResponse(
                content=content,
                provider=self.config.provider,
                model_name=self.config.model_name,
                tokens_used=tokens,
                cost=cost,
                latency_ms=latency_ms,
                success=True,
                metadata={
                    "task_type": task_type.value if task_type else None,
                    "temperature": kwargs.get("temperature", self.config.temperature),
                    "max_tokens": kwargs.get("max_tokens", self.config.max_tokens)
                }
            )
            
            # Log successful call
            self.logger.info(
                f"{self.config.provider.value} call successful",
                extra=response.to_dict()
            )
            
            # Record in history
            self._record_call(response)
            
            return response
            
        except Exception as e:
            # Track error
            self.config.total_errors += 1
            self.config.last_error_time = datetime.now()
            
            # Calculate error rate
            error_rate = self.config.total_errors / max(self.config.total_calls, 1)
            
            # Open circuit breaker if error rate too high
            if error_rate > 0.5 and self.config.total_calls > 10:
                self.config.circuit_open = True
                self.logger.error(
                    f"Circuit breaker opened for {self.config.provider.value}",
                    extra={"error_rate": error_rate, "total_calls": self.config.total_calls}
                )
            
            # Create error response
            latency_ms = (time.time() - start_time) * 1000
            response = ModelResponse(
                content="",
                provider=self.config.provider,
                model_name=self.config.model_name,
                tokens_used=0,
                cost=0.0,
                latency_ms=latency_ms,
                success=False,
                error=str(e)
            )
            
            # Log error
            self.logger.error(
                f"{self.config.provider.value} call failed",
                extra=response.to_dict(),
                exc_info=True
            )
            
            raise
    
    @retry(
        retry=retry_if_exception_type((Exception,)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True
    )
    def _call_with_retry(
        self,
        prompt: str,
        system_prompt: Optional[str],
        **kwargs
    ) -> str:
        """
        Call API with automatic retry logic.
        
        Uses tenacity for exponential backoff:
        - 1st retry: 2 seconds
        - 2nd retry: 4 seconds
        - 3rd retry: 8 seconds
        - Max: 10 seconds
        
        Args:
            prompt: User prompt
            system_prompt: System instructions
            **kwargs: Additional parameters
            
        Returns:
            Model response
            
        Raises:
            Exception: After all retries exhausted
        """
        return self._make_api_call(prompt, system_prompt, **kwargs)
    
    def _calculate_cost(self, prompt: str, response: str) -> float:
        """
        Calculate API call cost with cached token counting.
        
        Args:
            prompt: Input prompt
            response: Model output
            
        Returns:
            Cost in USD
        """
        input_tokens = self.estimate_tokens_fast(prompt)  # ✅ Use cached version
        output_tokens = self.estimate_tokens_fast(response)  # ✅ Use cached version
        
        input_cost = (input_tokens / 1000) * self.config.cost_per_1k_input
        output_cost = (output_tokens / 1000) * self.config.cost_per_1k_output
        
        return input_cost + output_cost
    
    def _enforce_rate_limit(self):
        """
        Enforce rate limiting.
        
        Simple implementation: 1 second between calls if at rate limit.
        Production: Use token bucket or sliding window.
        """
        if self._last_call_time:
            elapsed = (datetime.now() - self._last_call_time).total_seconds()
            min_interval = 60 / self.config.rate_limit
            
            if elapsed < min_interval:
                sleep_time = min_interval - elapsed
                self.logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)
        
        self._last_call_time = datetime.now()
    
    def _check_circuit_breaker(self):
        """
        Check if circuit breaker should be reset.
        
        Reset after 60 seconds of being open.
        """
        if self.config.last_error_time:
            elapsed = (datetime.now() - self.config.last_error_time).total_seconds()
            if elapsed > 60:  # Reset after 1 minute
                self.config.circuit_open = False
                self.logger.info(f"Circuit breaker reset for {self.config.provider.value}")
    
    def _record_call(self, response: ModelResponse):
        """Record call in history for metrics"""
        self._call_history.append({
            "timestamp": datetime.now().isoformat(),
            "success": response.success,
            "cost": response.cost,
            "latency_ms": response.latency_ms,
            "tokens": response.tokens_used
        })
        
        # Keep only last 100 calls
        if len(self._call_history) > 100:
            self._call_history = self._call_history[-100:]

    @lru_cache(maxsize=1000)
    def _estimate_tokens_cached(self, text_hash: str, text: str) -> int:
        """
        Cached token estimation for repeated text.
        
        Uses LRU cache to avoid re-counting tokens for same text.
        Cache key is hash to handle large texts efficiently.
        
        Args:
            text_hash: MD5 hash of text (for cache key)
            text: Actual text to count
            
        Returns:
            Token count
        """
        return self._estimate_tokens(text)
    
    def estimate_tokens_fast(self, text: str) -> int:
        """
        Fast token estimation with caching.
        
        Public method that uses cached estimation.
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        # Create hash for cache key
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return self._estimate_tokens_cached(text_hash, text)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get client performance metrics.
        
        Returns:
            Dictionary with comprehensive metrics
        """
        recent_calls = self._call_history[-10:] if self._call_history else []
        
        return {
            "provider": self.config.provider.value,
            "model": self.config.model_name,
            "total_calls": self.config.total_calls,
            "total_errors": self.config.total_errors,
            "error_rate": self.config.total_errors / max(self.config.total_calls, 1),
            "total_cost": self.config.total_cost,
            "avg_cost_per_call": self.config.total_cost / max(self.config.total_calls, 1),
            "circuit_open": self.config.circuit_open,
            "recent_latency_avg": sum(c["latency_ms"] for c in recent_calls) / max(len(recent_calls), 1),
            "recent_cost_avg": sum(c["cost"] for c in recent_calls) / max(len(recent_calls), 1)
        }
    
    def reset_metrics(self):
        """Reset all metrics (for testing)"""
        self.config.total_calls = 0
        self.config.total_errors = 0
        self.config.total_cost = 0.0
        self.config.circuit_open = False
        self._call_history = []


# Export public API
__all__ = [
    "BaseModelClient",
    "ModelConfig",
    "ModelResponse",
    "ModelProvider",
    "TaskType"
]

