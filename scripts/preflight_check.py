"""
Preflight model check — probe every configured model with a tiny live call.

PLAN.md edge case #1: the July 2026 outage was silent because retired model
IDs only fail at runtime, and the router degrades through fallbacks instead
of crashing. This script fails fast and loudly instead.

Probes go through the project's own client classes (not raw SDKs), so they
also exercise the per-provider payload shape (no temperature on Claude,
max_completion_tokens on OpenAI).

Usage:
    python scripts/preflight_check.py          # probe all three providers
Exit code 0 = all configured models respond; 1 = at least one failed.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from config.settings import settings


def probe_all() -> bool:
    from src.models.claude_client import ClaudeClient
    from src.models.gemini_client import GeminiClient
    from src.models.openai_client import OpenAIClient

    probes = [
        ("anthropic", settings.CLAUDE_MODEL, ClaudeClient),
        ("google", settings.GEMINI_MODEL, GeminiClient),
        ("openai", settings.OPENAI_MODEL, OpenAIClient),
    ]

    ok = True
    for provider, model, client_cls in probes:
        try:
            client = client_cls()
            response = client.call(
                prompt="Reply with the single word: OK",
                max_tokens=16,
                use_cache=False,
            )
            snippet = (response.content or "").strip().replace("\n", " ")[:40]
            print(f"  ✅ {provider:<10} {model:<28} -> {snippet!r} "
                  f"(${response.cost:.6f}, {response.latency_ms:.0f}ms)")
        except Exception as e:
            ok = False
            print(f"  ❌ {provider:<10} {model:<28} -> {type(e).__name__}: {e}")

    return ok


if __name__ == "__main__":
    print(f"Preflight model check — probing configured models with tiny live calls")
    all_ok = probe_all()
    if not all_ok:
        print("\nFAILED: at least one configured model is unreachable. "
              "Check model IDs in config/settings.py / .env "
              "(providers retire model IDs — see PLAN.md edge case #1).")
        sys.exit(1)
    print("\nAll configured models respond.")
    sys.exit(0)
