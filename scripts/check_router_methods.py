#!/usr/bin/env python3
"""Diagnostic: Check ModelRouter methods"""

import sys
from pathlib import Path

script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

print("=" * 70)
print("MODELROUTER DIAGNOSTIC")
print("=" * 70)

try:
    from src.models.router import ModelRouter, TaskType
    
    # Create router
    router = ModelRouter()
    
    print("\n✅ ModelRouter imported successfully")
    print("\n📋 Available methods:")
    print("-" * 70)
    
    # Get all methods
    methods = [m for m in dir(router) if not m.startswith('_')]
    for method in methods:
        print(f"   • {method}")
    
    print("\n📋 TaskType values:")
    print("-" * 70)
    for task in TaskType:
        print(f"   • {task.name} = {task.value}")
    
    print("\n✅ Correct usage examples:")
    print("-" * 70)
    print("   # Route a task:")
    print("   model = router.route_task(TaskType.STRATEGY_PLANNING)")
    print()
    print("   # Call a model directly:")
    print("   response = await router.call_model(")
    print("       task_type=TaskType.STRATEGY_PLANNING,")
    print("       prompt='Your prompt here'")
    print("   )")
    
    print("\n" + "=" * 70)
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()