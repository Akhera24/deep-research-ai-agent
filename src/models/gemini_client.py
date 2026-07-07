"""
Gemini (Google) Model Client

Integration with Gemini Flash models for document processing.

Use Cases:
- Large document processing (large context window)
- Multi-source data synthesis
- Efficient fact extraction
- Cost-effective bulk operations

Model: gemini-3.1-flash-lite (settings.GEMINI_MODEL)
Strengths: Speed, large context, cost-effective

Features:
- Large context window
- Fast inference (~2-3s)
- Relaxed safety settings for research
- Cost-effective processing
- Comprehensive error handling
- Token estimation with fallback
- Production-ready configuration
- Implements all required abstract methods

"""

from typing import Optional, Dict, Any

from google import genai
from google.genai import types

# ============================================================================
# INTERNAL IMPORTS
# ============================================================================
from src.models.base_client import (
    BaseModelClient,
    ModelConfig,
    ModelProvider
)
from config.settings import settings
from config.logging_config import get_logger

# ============================================================================
# OPTIONAL: TIKTOKEN FOR TOKEN COUNTING
# ============================================================================
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    tiktoken = None

logger = get_logger(__name__)


class GeminiClient(BaseModelClient):
    """
    Gemini Flash client implementation (model from settings.GEMINI_MODEL).

    Implements BaseModelClient abstract methods:
    - _make_api_call() -> str (required by base class)
    - _estimate_tokens() -> int (required by base class)

    Features:
    - Large context window
    - Fast inference (~2-3s average)
    - Multimodal capabilities (text, images, video)
    - Cost-effective for large documents
    - Relaxed safety settings for research

    Performance:
    - Speed: 2-3s per request
    - Cost: $0.25/$1.50 per 1M tokens (gemini-3.1-flash-lite; vs Claude's $5/$25)
    - Use case: Document processing, bulk extraction
    
    Safety Settings:
    - BLOCK_NONE for all categories (research agent needs flexibility)
    - Can be adjusted based on use case
    - Important: Monitor outputs for policy compliance
    
    Usage:
        >>> client = GeminiClient()
        >>> response = client.call("Extract facts from this document...")
        >>> print(f"Cost: ${response.cost:.4f}")
        >>> print(response.content)
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize Gemini client.
        
        Args:
            config: Optional model configuration
        
        Raises:
            ValueError: If API key not configured
        """
        # Default config
        if config is None:
            config = ModelConfig(
                provider=ModelProvider.GOOGLE,
                model_name=settings.GEMINI_MODEL,
                api_key=settings.GOOGLE_API_KEY,
                max_tokens=16384,  # Gemini output limit updated from 8000 for more comphresensive reporting and details 
                temperature=0.3,
                rate_limit=settings.GEMINI_RATE_LIMIT,
                max_retries=3,
                cost_per_1k_input=settings.GEMINI_INPUT_COST_PER_1M / 1000,
                cost_per_1k_output=settings.GEMINI_OUTPUT_COST_PER_1M / 1000
            )
        
        super().__init__(config)
        
        if not self.config.api_key:
            raise ValueError("Google API key not configured")
        
        # ====================================================================
        # INITIALIZE SDK (google-genai)
        # ====================================================================
        try:
            self.client = genai.Client(api_key=self.config.api_key)
        except Exception as e:
            logger.error(f"Failed to initialize Gemini SDK: {e}")
            raise

        # Relaxed safety settings (research agent needs flexibility)
        self.safety_settings = [
            types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH",
                threshold="BLOCK_NONE"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="BLOCK_NONE"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="BLOCK_NONE"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold="BLOCK_NONE"
            )
        ]
        
        # Token encoder (optional, for better cost estimation)
        if TIKTOKEN_AVAILABLE:
            try:
                self.encoder = tiktoken.get_encoding("cl100k_base")
                logger.debug("tiktoken encoder loaded")
            except Exception as e:
                self.encoder = None
                logger.debug(f"tiktoken loading failed: {e}, using estimation")
        else:
            self.encoder = None
            logger.debug("tiktoken not available, using token estimation")
        
        logger.info(
            "Gemini client initialized",
            extra={
                "model": self.config.model_name,
                "safety_settings": "relaxed",
                "sdk": "google.genai"
            }
        )
    
    def _make_api_call(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Make API call to Gemini.
        
        CRITICAL: This method is required by BaseModelClient (abstract method).
        
        Note: Gemini doesn't have a separate system prompt parameter.
        We prepend system instructions to the user prompt.
        
        Args:
            prompt: User message
            system_prompt: System instructions (prepended to prompt)
            **kwargs: Additional parameters (ignored for now)
            
        Returns:
            Gemini's response text (str, not ModelResponse)
            
        Raises:
            Exception: On API errors, rate limits, or network issues
            
        Example:
            >>> response = client._make_api_call(
            ...     prompt="What is AI?",
            ...     system_prompt="You are a helpful assistant."
            ... )
            >>> isinstance(response, str)
            True
        """
        # Build full prompt (Gemini doesn't separate system/user)
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        logger.debug(
            "Calling Gemini API",
            extra={
                "prompt_length": len(full_prompt),
                "has_system_prompt": bool(system_prompt),
                "model": self.config.model_name
            }
        )
        
        try:
            response = self.client.models.generate_content(
                model=self.config.model_name,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    temperature=self.config.temperature,
                    max_output_tokens=self.config.max_tokens,
                    safety_settings=self.safety_settings
                )
            )

            # Extract text from response
            content = response.text if hasattr(response, 'text') else str(response)

            logger.debug(
                "Gemini response received",
                extra={
                    "response_length": len(content),
                    "model": self.config.model_name
                }
            )
            
            return content
            
        except Exception as e:
            logger.error(
                "Gemini API error",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "model": self.config.model_name
                },
                exc_info=True
            )
            raise
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        
        CRITICAL: This method is required by BaseModelClient (abstract method).
        
        Uses tiktoken if available for accurate counting.
        Falls back to rough estimation (4 chars = 1 token).
        
        Gemini uses similar tokenization to GPT models,
        so cl100k_base encoder is a good approximation.
        
        Args:
            text: Input text to count tokens for
            
        Returns:
            Estimated token count (int)
            
        Example:
            >>> client._estimate_tokens("Hello, world!")
            3  # Approximate
        """
        if not text:
            return 0
        
        if self.encoder:
            try:
                tokens = len(self.encoder.encode(text))
                logger.debug(f"Token count (tiktoken): {tokens}")
                return tokens
            except Exception as e:
                logger.debug(f"tiktoken encoding failed: {e}, using fallback")
                # Fall through to rough estimation
        
        # Rough estimation: 4 characters ≈ 1 token
        # This is approximate but good enough for cost estimation
        estimated = len(text) // 4
        logger.debug(f"Token count (estimated): {estimated}")
        return estimated


def create_gemini_client() -> GeminiClient:
    """
    Create Gemini client with default settings.
    
    Convenience function for quick initialization.
    Uses settings from config/settings.py.
    
    Returns:
        Configured GeminiClient ready to use
        
    Example:
        >>> client = create_gemini_client()
        >>> response = client.call("Process this document...")
        >>> print(response.content)
    """
    return GeminiClient()

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = ["GeminiClient", "create_gemini_client"]