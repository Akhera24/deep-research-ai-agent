"""
Application Settings & Configuration

Centralized configuration management using Pydantic.
All settings loaded from environment variables with type validation.

Features:
- Type-safe configuration with Pydantic
- Environment-based settings (12-factor app)
- Validation on startup (fail fast)
- Security-focused (mask sensitive values)
- Helper methods for common operations
- Comprehensive model configurations
- All AI model configurations (Claude, Gemini, GPT-4)
- Search engine settings (Brave, Serper)
- Database configuration (PostgreSQL, Redis)
- Rate limiting and cost tracking
- CORS configuration with parsing
- Security helpers (mask sensitive data)
- Validation (auto-run on import)

Usage:
    >>> from config.settings import settings
    >>> print(settings.CLAUDE_MODEL)
    'claude-opus-4-20250514'
    >>> print(settings.mask_sensitive('ANTHROPIC_API_KEY'))
    'sk-ant-api...xyz'
"""

from pydantic_settings import BaseSettings
from pydantic import Field, model_validator
from typing import List, Optional
from functools import lru_cache


class Settings(BaseSettings):
    """
    Main application settings loaded from environment variables.
    
    Model Pricing (as of Jan 2026):
    - Claude Opus 4: $15/$75 per 1M input/output tokens
    - Gemini 2.0 Flash: $5/$15 per 1M input/output tokens  
    - GPT-4 Turbo: $10/$30 per 1M input/output tokens
    
    All settings can be overridden via environment variables.
    """
    
    # ========================================================================
    # ENVIRONMENT
    # ========================================================================
    ENVIRONMENT: str = Field(default="development", description="Runtime environment")
    DEBUG: bool = Field(default=True, description="Enable debug mode")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    
    # ========================================================================
    # API KEYS - AI MODELS
    # ========================================================================
    ANTHROPIC_API_KEY: str = Field(..., description="Anthropic API key for Claude")
    GOOGLE_API_KEY: str = Field(..., description="Google API key for Gemini")
    OPENAI_API_KEY: str = Field(..., description="OpenAI API key for GPT-4")
    PERPLEXITY_API_KEY: Optional[str] = Field(default=None, description="Perplexity API key (optional)")
    
    # ========================================================================
    # API KEYS - SEARCH ENGINES
    # ========================================================================
    BRAVE_API_KEY: str = Field(..., description="Brave Search API key")
    SERPER_API_KEY: Optional[str] = Field(default=None, description="Serper API key (optional)")
    SERPAPI_KEY: Optional[str] = Field(default=None, description="SerpAPI key (optional)")
    
    # ========================================================================
    # DATABASE
    # ========================================================================
    DATABASE_URL: str = Field(..., description="PostgreSQL connection URL")
    REDIS_URL: str = Field(default="redis://localhost:6379/0", description="Redis connection URL")
    
    # ========================================================================
    # CLAUDE (ANTHROPIC) CONFIGURATION
    # ========================================================================
    CLAUDE_MODEL: str = Field(
        default="claude-opus-4-20250514",
        description="Claude model identifier"
    )
    CLAUDE_RATE_LIMIT: int = Field(
        default=50,
        description="Claude rate limit (requests per minute)",
        ge=1
    )
    CLAUDE_MAX_TOKENS: int = Field(default=4000, ge=1, le=100000)
    CLAUDE_TEMPERATURE: float = Field(default=0.3, ge=0.0, le=2.0)
    CLAUDE_TIMEOUT: int = Field(default=60, ge=5, le=300)
    
    # Claude Pricing (per 1 million tokens)
    # These names MUST match what claude_client.py expects!
    CLAUDE_INPUT_COST_PER_1M: float = Field(
        default=15.0,
        description="Claude input cost per 1M tokens (USD)"
    )
    CLAUDE_OUTPUT_COST_PER_1M: float = Field(
        default=75.0,
        description="Claude output cost per 1M tokens (USD)"
    )
    
    # ========================================================================
    # GEMINI (GOOGLE) CONFIGURATION
    # ========================================================================
    GEMINI_MODEL: str = Field(
        default="gemini-2.5-flash",
        description="Gemini model identifier (2.5 Flash - stable, recommended)"
    )
    GEMINI_RATE_LIMIT: int = Field(
        default=360,
        description="Gemini rate limit (requests per minute)",
        ge=1
    )
    GEMINI_MAX_TOKENS: int = Field(default=8000, ge=1, le=1000000)
    GEMINI_TEMPERATURE: float = Field(default=0.3, ge=0.0, le=2.0)
    GEMINI_TIMEOUT: int = Field(default=60, ge=5, le=300)
    
    # Gemini Pricing (per 1 million tokens)
    GEMINI_INPUT_COST_PER_1M: float = Field(
        default=0.15,
        description="Gemini 2.5 Flash input cost per 1M tokens (USD)"
    )
    GEMINI_OUTPUT_COST_PER_1M: float = Field(
        default=0.60,
        description="Gemini 2.5 Flash output cost per 1M tokens (USD)"
    )
    
    # ========================================================================
    # OPENAI (GPT-4) CONFIGURATION
    # ========================================================================
    OPENAI_MODEL: str = Field(
        default="gpt-4-turbo-preview",
        description="OpenAI model identifier"
    )
    OPENAI_RATE_LIMIT: int = Field(
        default=500,
        description="OpenAI rate limit (requests per minute)",
        ge=1
    )
    OPENAI_MAX_TOKENS: int = Field(default=4000, ge=1, le=128000)
    OPENAI_TEMPERATURE: float = Field(default=0.3, ge=0.0, le=2.0)
    OPENAI_TIMEOUT: int = Field(default=60, ge=5, le=300)
    
    # OpenAI Pricing (per 1 million tokens)
    OPENAI_INPUT_COST_PER_1M: float = Field(
        default=10.0,
        description="OpenAI input cost per 1M tokens (USD)"
    )
    OPENAI_OUTPUT_COST_PER_1M: float = Field(
        default=30.0,
        description="OpenAI output cost per 1M tokens (USD)"
    )
    
    # ========================================================================
    # SEARCH CONFIGURATION
    # ========================================================================
    MAX_SEARCH_ITERATIONS: int = Field(
        default=50,
        description="Maximum search iterations per research session",
        ge=1,
        le=100
    )
    MAX_CONCURRENT_SEARCHES: int = Field(
        default=5,
        description="Maximum concurrent search requests",
        ge=1,
        le=10
    )
    SEARCH_TIMEOUT: int = Field(
        default=30,
        description="Search timeout in seconds",
        ge=5,
        le=60
    )
    DEFAULT_SEARCH_ENGINE: str = Field(
        default="brave",
        description="Default search engine to use"
    )
    
    # ========================================================================
    # APPLICATION SETTINGS
    # ========================================================================
    MAX_CONCURRENT_REQUESTS: int = Field(
        default=10,
        description="Maximum concurrent API requests",
        ge=1,
        le=50
    )
    REQUEST_TIMEOUT: int = Field(
        default=300,
        description="Request timeout in seconds",
        ge=30,
        le=600
    )
    
    # Rate limiting
    RATE_LIMIT_ENABLED: bool = Field(
        default=True,
        description="Enable rate limiting"
    )
    RATE_LIMIT_PER_MINUTE: int = Field(
        default=60,
        description="Global rate limit per minute",
        ge=1
    )
    
    # ========================================================================
    # CORS (for API)
    # ========================================================================
    # IMPORTANT: Load as string, parse in validator (Pydantic limitation)
    cors_origins_raw: str = Field(
        default="http://localhost:3000,http://localhost:5173,http://localhost:8000",
        alias="CORS_ORIGINS",
        description="Comma-separated CORS origins"
    )
    cors_origins_list: List[str] = []
    
    CORS_ALLOW_CREDENTIALS: bool = Field(default=True)
    CORS_ALLOW_METHODS: List[str] = Field(default=["*"])
    CORS_ALLOW_HEADERS: List[str] = Field(default=["*"])
    
    # ========================================================================
    # MONITORING & OBSERVABILITY
    # ========================================================================
    SENTRY_DSN: Optional[str] = Field(
        default=None,
        description="Sentry DSN for error tracking"
    )
    ENABLE_METRICS: bool = Field(
        default=True,
        description="Enable metrics collection"
    )
    METRICS_PORT: int = Field(
        default=9090,
        description="Metrics server port",
        ge=1024,
        le=65535
    )
    
    # ========================================================================
    # SECURITY
    # ========================================================================
    SECRET_KEY: str = Field(
        default="dev-secret-key-change-in-production",
        description="Application secret key"
    )
    API_KEY_HEADER: str = Field(
        default="X-API-Key",
        description="API key header name"
    )
    REQUIRE_API_KEY: bool = Field(
        default=False,
        description="Require API key for requests"
    )
    
    # ========================================================================
    # PYDANTIC VALIDATORS
    # ========================================================================
    
    @model_validator(mode='after')
    def parse_cors_origins(self):
        """
        Parse CORS_ORIGINS from comma-separated string to list.
        
        Pydantic can't directly parse comma-separated env vars to List[str],
        so we load as string and parse in this validator.
        
        Process:
        1. Load cors_origins_raw as string from CORS_ORIGINS env var
        2. Split by comma and strip whitespace
        3. Store in cors_origins_list
        4. Access via CORS_ORIGINS property
        
        Returns:
            Self (required by Pydantic)
        """
        raw = self.cors_origins_raw
        
        if raw and raw.strip():
            self.cors_origins_list = [
                origin.strip() 
                for origin in raw.split(",") 
                if origin.strip()
            ]
        else:
            # Default fallback
            self.cors_origins_list = [
                "http://localhost:3000",
                "http://localhost:5173",
                "http://localhost:8000"
            ]
        
        return self
    
    # ========================================================================
    # HELPER METHODS (SECURITY & UTILITY)
    # ========================================================================
    
    @property
    def CORS_ORIGINS(self) -> List[str]:
        """
        Get CORS origins as a list.
        
        Returns:
            List of CORS origin URLs
            
        Example:
            >>> settings.CORS_ORIGINS
            ['http://localhost:3000', 'http://localhost:5173']
        """
        return self.cors_origins_list
    
    def mask_sensitive(self, key: str) -> str:
        """
        Mask sensitive values for safe logging.
        
        Shows first 10 and last 4 characters, masks the middle.
        Critical for security - never log full API keys!
        
        Args:
            key: Setting attribute name (e.g., 'ANTHROPIC_API_KEY')
            
        Returns:
            Masked string (e.g., "sk-ant-api...xyz")
            
        Example:
            >>> settings.mask_sensitive('ANTHROPIC_API_KEY')
            'sk-ant-api...4xyz'
        """
        value = getattr(self, key, None)
        if value and isinstance(value, str) and len(value) > 14:
            return f"{value[:10]}...{value[-4:]}"
        return "***"
    
    def get_database_url(self, hide_password: bool = True) -> str:
        """
        Get database URL, optionally masking the password.
        
        Args:
            hide_password: If True, replace password with ***
            
        Returns:
            Database URL (safe for logging if hide_password=True)
            
        Example:
            >>> settings.get_database_url(hide_password=True)
            'postgresql://user:***@localhost:5432/db'
        """
        if not hide_password:
            return self.DATABASE_URL
        
        url = self.DATABASE_URL
        
        # Parse URL to mask password
        if "@" in url and "://" in url:
            try:
                scheme = url.split("://")[0]
                rest = url.split("://")[1]
                
                if "@" in rest:
                    credentials, host = rest.split("@", 1)
                    if ":" in credentials:
                        user = credentials.split(":")[0]
                        return f"{scheme}://{user}:***@{host}"
            except Exception:
                pass
        
        return url
    
    def get_all_api_keys_masked(self) -> dict:
        """
        Get all API keys in masked format for logging.
        
        Returns:
            Dictionary of all API keys (masked)
            
        Example:
            >>> settings.get_all_api_keys_masked()
            {
                'ANTHROPIC_API_KEY': 'sk-ant-api...xyz',
                'GOOGLE_API_KEY': 'AIza...abc',
                ...
            }
        """
        api_key_fields = [
            'ANTHROPIC_API_KEY',
            'GOOGLE_API_KEY', 
            'OPENAI_API_KEY',
            'PERPLEXITY_API_KEY',
            'BRAVE_API_KEY',
            'SERPER_API_KEY',
            'SERPAPI_KEY'
        ]
        
        return {
            field: self.mask_sensitive(field)
            for field in api_key_fields
            if getattr(self, field, None)
        }
    
    def get_model_costs(self) -> dict:
        """
        Get all model cost configurations.
        
        Returns:
            Dictionary of model costs per 1M tokens
            
        Example:
            >>> settings.get_model_costs()
            {
                'claude': {'input': 15.0, 'output': 75.0},
                'gemini': {'input': 5.0, 'output': 15.0},
                'openai': {'input': 10.0, 'output': 30.0}
            }
        """
        return {
            'claude': {
                'input_per_1m': self.CLAUDE_INPUT_COST_PER_1M,
                'output_per_1m': self.CLAUDE_OUTPUT_COST_PER_1M,
                'input_per_1k': self.CLAUDE_INPUT_COST_PER_1M / 1000,
                'output_per_1k': self.CLAUDE_OUTPUT_COST_PER_1M / 1000
            },
            'gemini': {
                'input_per_1m': self.GEMINI_INPUT_COST_PER_1M,
                'output_per_1m': self.GEMINI_OUTPUT_COST_PER_1M,
                'input_per_1k': self.GEMINI_INPUT_COST_PER_1M / 1000,
                'output_per_1k': self.GEMINI_OUTPUT_COST_PER_1M / 1000
            },
            'openai': {
                'input_per_1m': self.OPENAI_INPUT_COST_PER_1M,
                'output_per_1m': self.OPENAI_OUTPUT_COST_PER_1M,
                'input_per_1k': self.OPENAI_INPUT_COST_PER_1M / 1000,
                'output_per_1k': self.OPENAI_OUTPUT_COST_PER_1M / 1000
            }
        }
    
    # ========================================================================
    # PYDANTIC CONFIGURATION
    # ========================================================================
    
    class Config:
        """Pydantic configuration"""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "allow"  # Allow extra fields for future additions
        populate_by_name = True  # Allow alias population


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance (singleton).
    
    Uses @lru_cache to ensure only one Settings instance per application.
    This is the recommended Pydantic pattern.
    
    Returns:
        Settings instance
        
    Example:
        >>> settings = get_settings()
        >>> print(settings.CLAUDE_MODEL)
        'claude-opus-4-20250514'
    """
    return Settings()


def validate_settings() -> bool:
    """
    Validate that all required settings are present and valid.
    
    Called automatically on import to fail fast if configuration is invalid.
    This is FAANG best practice - fail at startup, not during execution!
    
    Returns:
        True if all required settings are present
        
    Raises:
        ValueError: If any required setting is missing or empty
        
    Example:
        >>> validate_settings()
        True
    """
    required_keys = [
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "OPENAI_API_KEY",
        "BRAVE_API_KEY",
        "DATABASE_URL"
    ]
    
    missing = []
    for key in required_keys:
        value = getattr(settings, key, None)
        if not value or (isinstance(value, str) and value.strip() == ""):
            missing.append(key)
    
    if missing:
        raise ValueError(
            f"\n❌ Missing required settings: {', '.join(missing)}\n"
            f"   Please set these in your .env file\n"
        )
    
    return True


# ============================================================================
# INITIALIZATION & EXPORTS
# ============================================================================

# Global singleton instance
settings = get_settings()

# Auto-validate on import (fail fast!)
try:
    validate_settings()
    # Only print in development
    if settings.ENVIRONMENT == "development":
        print("✅ Settings validated successfully")
except ValueError as e:
    print(f"\n⚠️  Configuration Error:\n{e}")
    raise
except Exception as e:
    print(f"\n⚠️  Unexpected settings error: {e}")
    raise


# Export public API
__all__ = [
    "Settings",
    "settings",
    "get_settings",
    "validate_settings"
]