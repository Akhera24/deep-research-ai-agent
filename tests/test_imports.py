"""
Test that all imports work correctly.

This verifies the module structure is correct.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

print("Testing imports...")

# Test 1: Config imports
print("\n1. Testing config imports...")
try:
    from config.settings import settings
    print("   ✅ settings imported")
except Exception as e:
    print(f"   ❌ settings failed: {e}")
    sys.exit(1)

try:
    from config.logging_config import get_logger, logger
    print("   ✅ logging_config imported")
except Exception as e:
    print(f"   ❌ logging_config failed: {e}")
    sys.exit(1)

# Test 2: Database imports
print("\n2. Testing database imports...")
try:
    from src.database.models import ResearchSession, Fact, RiskFlag, Connection
    print("   ✅ database models imported")
except Exception as e:
    print(f"   ❌ database models failed: {e}")
    sys.exit(1)

try:
    from src.database.connection import get_db, init_db, check_connection
    print("   ✅ database connection imported")
except Exception as e:
    print(f"   ❌ database connection failed: {e}")
    sys.exit(1)

try:
    from src.database.repository import (
        ResearchSessionRepository,
        FactRepository,
        RiskFlagRepository,
        ConnectionRepository
    )
    print("   ✅ database repositories imported")
except Exception as e:
    print(f"   ❌ database repositories failed: {e}")
    sys.exit(1)

# Test 3: State manager imports
print("\n3. Testing state manager imports...")
try:
    from src.core.state_manager import ResearchState, StateManager
    print("   ✅ state manager imported")
except Exception as e:
    print(f"   ❌ state manager failed: {e}")
    sys.exit(1)

# Test 4: Model client imports
print("\n4. Testing model client imports...")
try:
    from src.models.base_client import (
        BaseModelClient,
        ModelConfig,
        ModelResponse,
        ModelProvider,
        TaskType
    )
    print("   ✅ base_client imported")
except Exception as e:
    print(f"   ❌ base_client failed: {e}")
    sys.exit(1)

try:
    from src.models.claude_client import ClaudeClient, create_claude_client
    print("   ✅ claude_client imported")
except Exception as e:
    print(f"   ❌ claude_client failed: {e}")
    sys.exit(1)

try:
    from src.models.gemini_client import GeminiClient, create_gemini_client
    print("   ✅ gemini_client imported")
except Exception as e:
    print(f"   ❌ gemini_client failed: {e}")
    sys.exit(1)

try:
    from src.models.openai_client import OpenAIClient, create_openai_client
    print("   ✅ openai_client imported")
except Exception as e:
    print(f"   ❌ openai_client failed: {e}")
    sys.exit(1)

try:
    from src.models.router import ModelRouter, create_router
    print("   ✅ router imported")
except Exception as e:
    print(f"   ❌ router failed: {e}")
    sys.exit(1)

# Test 5: Actually create instances
print("\n5. Testing instance creation...")
try:
    logger_test = get_logger("test")
    logger_test.info("Test log message")
    print("   ✅ Logger instance created and working")
except Exception as e:
    print(f"   ❌ Logger instance failed: {e}")
    sys.exit(1)

try:
    # Don't actually call API, just create client
    claude = ClaudeClient()
    print("   ✅ Claude client instance created")
except Exception as e:
    print(f"   ❌ Claude client instance failed: {e}")
    sys.exit(1)

try:
    gemini = GeminiClient()
    print("   ✅ Gemini client instance created")
except Exception as e:
    print(f"   ❌ Gemini client instance failed: {e}")
    sys.exit(1)

try:
    openai = OpenAIClient()
    print("   ✅ OpenAI client instance created")
except Exception as e:
    print(f"   ❌ OpenAI client instance failed: {e}")
    sys.exit(1)

try:
    router = ModelRouter()
    print("   ✅ Router instance created")
    print(f"   ✅ Router has {len(router.clients)} model clients")
except Exception as e:
    print(f"   ❌ Router instance failed: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("🎉 ALL IMPORTS WORKING!")
print("=" * 70)
print("\n✅ Configuration: OK")
print("✅ Database: OK")
print("✅ State Management: OK")
print("✅ Model Clients: OK")
print("✅ Router: OK")
print("\n✅✅✅ Ready for full model tests! ✅✅✅")