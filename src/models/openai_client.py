"""
OpenAI (GPT-4) Model Client

Integration with GPT-4 Turbo for structured output and reliable JSON generation.

Use Cases:
- Structured data extraction (JSON mode)
- Function calling
- Reliable formatted output
- Fast iteration cycles

Model: gpt-4-turbo-preview
Context: 128K tokens
Strengths: Structured output, reliability, speed

Features:
- JSON mode for guaranteed valid JSON
- Function calling support
- Response format validation
- Streaming support (future)
"""

from typing import Optional, Dict, Any, List
from openai import OpenAI, APIError, RateLimitError, APIConnectionError
import tiktoken
import json

from src.models.base_client import (
    BaseModelClient,
    ModelConfig,
    ModelProvider
)
from config.settings import settings
from config.logging_config import get_logger

logger = get_logger(__name__)


class OpenAIClient(BaseModelClient):
    """
    GPT-4 Turbo client implementation.
    
    Features:
    - Reliable structured output (JSON mode)
    - Function calling capabilities
    - Fast inference
    - 128K context window
    
    Performance:
    - Fastest response times (avg 2-3s)
    - Highest reliability for JSON generation
    - Best for structured data extraction
    
    Usage:
        >>> client = OpenAIClient()
        >>> response = client.call(
        ...     "Extract facts as JSON",
        ...     response_format="json"
        ... )
        >>> data = json.loads(response.content)
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize OpenAI client.
        
        Args:
            config: Optional custom configuration
                   If None, uses default from settings
        """
        if config is None:
            config = ModelConfig(
                provider=ModelProvider.OPENAI,
                model_name=settings.OPENAI_MODEL,
                api_key=settings.OPENAI_API_KEY,
                max_tokens=4000,
                temperature=0.3,
                rate_limit=settings.OPENAI_RATE_LIMIT,
                cost_per_1k_input=settings.OPENAI_INPUT_COST_PER_1M / 1000,
                cost_per_1k_output=settings.OPENAI_OUTPUT_COST_PER_1M / 1000
            )
        
        super().__init__(config)
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=config.api_key)
        
        # Token encoder (OpenAI provides exact encoder)
        try:
            # Use model-specific encoder
            if "gpt-4" in config.model_name:
                self.encoder = tiktoken.encoding_for_model("gpt-4")
            else:
                self.encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.encoder = None
            self.logger.warning("tiktoken not available, using rough estimation")
        
        self.logger.info(
            "OpenAI client initialized",
            extra={
                "model": config.model_name,
                "supports_json_mode": True,
                "supports_functions": True
            }
        )
    
    def _make_api_call(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Make API call to OpenAI with optional JSON mode.
        
        Features:
        - JSON mode: Guarantees valid JSON output
        - Function calling: Structured tool use
        - Streaming: Real-time response (future)
        
        Args:
            prompt: User message
            system_prompt: System instructions
            **kwargs: Additional parameters
                - response_format: "json" for JSON mode
                - functions: List of function definitions
                - max_tokens: Override default
                - temperature: Override default
            
        Returns:
            GPT-4's response text
            
        Raises:
            RateLimitError: When rate limit exceeded
            APIConnectionError: On network issues
            APIError: On other API errors
        """
        try:
            # Build messages
            messages = []
            
            # Add system message if provided
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            else:
                messages.append({"role": "system", "content": "You are a helpful AI assistant."})
            
            # Add user message
            messages.append({"role": "user", "content": prompt})
            
            # Get parameters
            max_tokens = kwargs.get("max_tokens", self.config.max_tokens)
            temperature = kwargs.get("temperature", self.config.temperature)
            response_format = kwargs.get("response_format", "text")
            
            # Build API call parameters
            api_params = {
                "model": self.config.model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            
            # Enable JSON mode if requested
            if response_format == "json":
                api_params["response_format"] = {"type": "json_object"}
                self.logger.debug("JSON mode enabled")
            
            # Add function calling if provided
            if "functions" in kwargs:
                api_params["functions"] = kwargs["functions"]
                self.logger.debug(f"Function calling enabled ({len(kwargs['functions'])} functions)")
            
            # Make API call
            self.logger.debug(
                "Calling OpenAI API",
                extra={
                    "prompt_length": len(prompt),
                    "max_tokens": max_tokens,
                    "json_mode": response_format == "json"
                }
            )
            
            response = self.client.chat.completions.create(**api_params)
            
            # Extract content
            content = response.choices[0].message.content
            
            # Validate JSON if in JSON mode
            if response_format == "json":
                try:
                    json.loads(content)  # Validate
                    self.logger.debug("Valid JSON response received")
                except json.JSONDecodeError as e:
                    self.logger.error(f"Invalid JSON in response: {e}")
                    # GPT-4 JSON mode should always return valid JSON
                    # If this happens, it's a bug
                    raise
            
            return content
            
        except RateLimitError as e:
            self.logger.error(
                "OpenAI rate limit exceeded",
                extra={"error": str(e), "rate_limit": self.config.rate_limit},
                exc_info=True
            )
            raise
        except APIConnectionError as e:
            self.logger.error(
                "OpenAI API connection error",
                extra={"error": str(e)},
                exc_info=True
            )
            raise
        except APIError as e:
            self.logger.error(
                "OpenAI API error",
                extra={"error": str(e)},
                exc_info=True
            )
            raise
        except Exception as e:
            self.logger.error(
                "Unexpected OpenAI error",
                extra={"error": str(e)},
                exc_info=True
            )
            raise
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        
        Uses OpenAI's tiktoken library for exact counting.
        
        Args:
            text: Input text
            
        Returns:
            Exact token count
        """
        if self.encoder:
            try:
                return len(self.encoder.encode(text))
            except Exception as e:
                self.logger.warning(f"tiktoken encoding failed: {e}")
                return len(text) // 4
        else:
            # Rough estimation: 4 characters ≈ 1 token
            return len(text) // 4
    
    def call_with_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Call GPT-4 with JSON mode and parse response.
        
        Convenience method that:
        1. Enables JSON mode
        2. Makes API call
        3. Parses JSON response
        4. Returns parsed dictionary
        
        Args:
            prompt: User message (should request JSON output)
            system_prompt: System instructions
            **kwargs: Additional parameters
            
        Returns:
            Parsed JSON as dictionary
            
        Example:
            >>> client = OpenAIClient()
            >>> data = client.call_with_json(
            ...     "Extract person details as JSON: Sarah Chen is CEO of TechCorp"
            ... )
            >>> print(data["name"])  # "Sarah Chen"
            >>> print(data["title"])  # "CEO"
        """
        # Ensure JSON mode is enabled
        kwargs["response_format"] = "json"
        
        # Make call
        response = self.call(prompt, system_prompt, **kwargs)
        
        # Parse JSON
        try:
            return json.loads(response.content)
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {e}")
            raise


# Convenience function
def create_openai_client() -> OpenAIClient:
    """
    Create OpenAI client with default settings.
    
    Returns:
        Configured OpenAIClient
        
    Example:
        >>> client = create_openai_client()
        >>> response = client.call("Extract facts from this text...")
    """
    return OpenAIClient()


# Export
__all__ = ["OpenAIClient", "create_openai_client"]