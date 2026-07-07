"""
Test that all imports work correctly.

This verifies the module structure is correct.

Rewritten as a proper pytest module (2026-07-07): the old version ran at
module level and called sys.exit(), which aborted pytest collection for
the entire tests/ directory.

Run with: pytest tests/test_imports.py -v
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import importlib.util

import pytest


def test_config_imports():
    from config.settings import settings
    from config.logging_config import get_logger, logger
    assert settings.CLAUDE_MODEL
    assert get_logger("test") is not None


@pytest.mark.skipif(
    importlib.util.find_spec("psycopg2") is None,
    reason="psycopg2 not installed in this venv (requirements drift; "
           "src/database is not on the runtime path — repin lands in Phase 2, "
           "Postgres wiring in Phase 3)",
)
def test_database_imports():
    from src.database.models import ResearchSession, Fact, RiskFlag, Connection
    from src.database.connection import get_db, init_db, check_connection
    from src.database.repository import (
        ResearchSessionRepository,
        FactRepository,
        RiskFlagRepository,
        ConnectionRepository,
    )


def test_state_manager_imports():
    from src.core.state_manager import ResearchState, StateManager


def test_model_client_imports():
    from src.models.base_client import (
        BaseModelClient,
        ModelConfig,
        ModelResponse,
        ModelProvider,
        TaskType,
    )
    from src.models.claude_client import ClaudeClient, create_claude_client
    from src.models.gemini_client import GeminiClient, create_gemini_client
    from src.models.openai_client import OpenAIClient, create_openai_client
    from src.models.router import ModelRouter, create_router


def test_client_instances_create():
    """Create every client (no API calls) and the router."""
    from src.models.claude_client import ClaudeClient
    from src.models.gemini_client import GeminiClient
    from src.models.openai_client import OpenAIClient
    from src.models.router import ModelRouter

    assert ClaudeClient() is not None
    assert GeminiClient() is not None
    assert OpenAIClient() is not None

    router = ModelRouter()
    assert len(router.clients) == 3


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
