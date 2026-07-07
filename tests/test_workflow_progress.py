"""
Tests for the ResearchOrchestrator progress hook (PHASE3_DESIGN §11.R3).

The graph is mocked — these verify the astream plumbing, callback contract,
and abort propagation without any API spend.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from unittest.mock import MagicMock

import pytest

from src.core.workflow import ResearchOrchestrator


def _make_orchestrator(snapshots):
    orch = ResearchOrchestrator(max_iterations=3, enable_checkpoints=False)

    class FakeGraph:
        def astream(self, initial_state, stream_mode):
            assert stream_mode == "values"

            async def gen():
                for s in snapshots:
                    yield s
            return gen()

    orch.workflow = FakeGraph()
    orch._format_results = MagicMock(side_effect=lambda state: {
        "final": state,
        "facts": state.get("facts", []),
        "metadata": {"iterations": state.get("iteration", 0), "duration_seconds": 0.0},
    })
    return orch


SNAPSHOTS = [
    {"stage": "initialization", "iteration": 0, "facts": []},
    {"stage": "extracting_facts", "iteration": 1, "facts": [{"a": 1}, {"b": 2}],
     "coverage": {"average": 0.4}},
    {"stage": "complete", "iteration": 2, "facts": [{"a": 1}, {"b": 2}, {"c": 3}]},
]


@pytest.mark.asyncio
async def test_no_callback_matches_old_behavior():
    orch = _make_orchestrator(SNAPSHOTS)
    result = await orch.research("Test Person")
    # Final state is the LAST streamed snapshot — same as ainvoke's return.
    orch._format_results.assert_called_once_with(SNAPSHOTS[-1])
    assert result["final"] == SNAPSHOTS[-1]


@pytest.mark.asyncio
async def test_callback_invoked_per_node_with_contract_fields():
    orch = _make_orchestrator(SNAPSHOTS)
    seen = []

    async def cb(p):
        seen.append(p)

    await orch.research("Test Person", progress_callback=cb)

    assert len(seen) == len(SNAPSHOTS)
    assert seen[1] == {
        "node": "extracting_facts",
        "iteration": 1,
        "max_iterations": 3,
        "facts": 2,
        "coverage": {"average": 0.4},
    }


@pytest.mark.asyncio
async def test_callback_exception_aborts_run():
    """A budget-abort raised inside the callback must propagate out of
    research() (jobs.py relies on this for the $1 per-job cap)."""

    class BudgetExceeded(Exception):
        pass

    orch = _make_orchestrator(SNAPSHOTS)

    async def cb(p):
        if p["iteration"] >= 1:
            raise BudgetExceeded("cap hit")

    with pytest.raises(BudgetExceeded):
        await orch.research("Test Person", progress_callback=cb)
