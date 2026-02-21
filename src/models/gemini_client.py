"""
Gemini (Google) Model Client

Integration with Gemini 2.5 Flash for document processing.

MIGRATION: Upgraded to google.genai (modern SDK) with backward compatibility

Use Cases:
- Large document processing (2M token context)
- Multi-source data synthesis
- Efficient fact extraction
- Cost-effective bulk operations

Model: gemini-2.5-flash
Context: 2M tokens (largest available)
Strengths: Speed, large context, cost-effective

Features:
- 2M token context window
- Fast inference (~2-3s)
- Relaxed safety settings for research
- Cost-effective processing
- Backward compatibility with old SDK
- Comprehensive error handling
- Modern SDK with graceful fallback
- Token estimation with fallback
- Production-ready configuration
- Implements all required abstract methods

"""

from typing import Optional, Dict, Any

# ============================================================================
# GOOGLE GENAI IMPORTS - Try new, fall back to old
# ============================================================================
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
    logger_msg = "✅ Using NEW google.genai SDK (no warnings!)"
except ImportError:
    # Fallback to old package
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning, module='google.generativeai')
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    GENAI_AVAILABLE = False
    types = None
    logger_msg = "⚠️  Using deprecated google.generativeai"

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
    Gemini 2.5 Flash client implementation.
    
    CORRECTED: Now properly implements BaseModelClient abstract methods:
    - _make_api_call() -> str (required by base class)
    - _estimate_tokens() -> int (required by base class)
    
    Features:
    - 2M token context window (largest available)
    - Fast inference (~2-3s average)
    - Multimodal capabilities (text, images, video)
    - Cost-effective for large documents
    - Relaxed safety settings for research
    - Backward compatibility with old SDK
    
    Performance:
    - Context: 2M tokens (vs Claude's 200K)
    - Speed: 2-3s per request
    - Cost: $0.15/$0.60 per 1M tokens (vs Claude's $15/$75)
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
        # INITIALIZE SDK (New or Old)
        # ====================================================================
        if GENAI_AVAILABLE:
            # NEW SDK: google.genai
            try:
                self.client = genai.Client(api_key=self.config.api_key)
                self.model = None  # Not used in new SDK
                logger.info(logger_msg)
            except Exception as e:
                logger.error(f"Failed to initialize new Gemini SDK: {e}")
                raise
        else:
            # OLD SDK: google.generativeai
            try:
                genai.configure(api_key=self.config.api_key)
                self.client = None  # Not used in old SDK
                self.model = genai.GenerativeModel(
                    model_name=self.config.model_name,
                    generation_config={
                        "temperature": self.config.temperature,
                        "max_output_tokens": self.config.max_tokens,
                    },
                    safety_settings={
                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    }
                )
                logger.info(logger_msg)
                logger.warning("   To remove warnings: pip install google-genai")
            except Exception as e:
                logger.error(f"Failed to initialize old Gemini SDK: {e}")
                raise
        
        # Safety settings for new SDK (prepared but only used if new SDK)
        if GENAI_AVAILABLE:
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
        else:
            self.safety_settings = None
        
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
                "context_window": "2M tokens",
                "safety_settings": "relaxed",
                "sdk": "google.genai" if GENAI_AVAILABLE else "google.generativeai (deprecated)"
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
            # ================================================================
            # CALL API (Different for new vs old SDK)
            # ================================================================
            if GENAI_AVAILABLE:
                # NEW SDK: google.genai
                response = self.client.models.generate_content(
                    model=self.config.model_name,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        temperature=self.config.temperature,
                        max_output_tokens=self.config.max_tokens,
                        safety_settings=self.safety_settings
                    )
                )
                
                # Extract text from new SDK response
                content = response.text if hasattr(response, 'text') else str(response)
                
            else:
                # OLD SDK: google.generativeai
                response = self.model.generate_content(full_prompt)
                content = response.text
            
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