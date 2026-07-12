"""
Truncated-JSON repair in _parse_json_array_response (risks + connections).

Live failure 2026-07-11 (Phase B gate run): the connections response hit the
output-token limit and ended mid-string, inside an opening ```json fence with
no closing fence. The parser required a COMPLETE fence pair and a closing ']'
→ 0 connections → a 20-point quality-score drop. The repair recovers every
complete leading object, mirroring the extractor's _parse_ai_response design.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from src.core.workflow import ResearchOrchestrator


def _parser():
    return ResearchOrchestrator.__new__(ResearchOrchestrator)


def _obj(i):
    return ('{"entity_1": "T", "entity_2": "E%d", '
            '"relationship_type": "employer", "strength": 0.8}' % i)


def test_truncated_fenced_response_recovers_complete_objects():
    content = '```json\n[\n' + ',\n'.join(_obj(i) for i in range(1, 6)) \
              + ',\n{"entity_1": "T", "entity_2": "cut off mid str'
    out = _parser()._parse_json_array_response(content, "T", "connections")
    assert len(out) == 5
    assert out[4]["entity_2"] == "E5"


def test_truncated_bare_array_recovers():
    content = '[' + ','.join(_obj(i) for i in range(1, 4)) + ',{"entity'
    out = _parser()._parse_json_array_response(content, "T", "connections")
    assert len(out) == 3


def test_complete_fenced_response_unchanged():
    content = '```json\n[' + _obj(1) + ']\n```'
    out = _parser()._parse_json_array_response(content, "T", "connections")
    assert len(out) == 1


def test_complete_bare_array_unchanged():
    out = _parser()._parse_json_array_response('[' + _obj(1) + ']', "T", "risks")
    assert len(out) == 1


def test_prose_around_array_still_parses():
    content = 'Here are the connections:\n[' + _obj(1) + ']\nHope this helps!'
    out = _parser()._parse_json_array_response(content, "T", "connections")
    assert len(out) == 1


def test_no_array_returns_empty():
    assert _parser()._parse_json_array_response("no json here", "T", "c") == []
    assert _parser()._parse_json_array_response("", "T", "c") == []


def test_unrepairable_truncation_returns_empty():
    out = _parser()._parse_json_array_response('[{"never closes', "T", "c")
    assert out == []


def test_non_list_json_returns_empty():
    out = _parser()._parse_json_array_response('{"a": 1}', "T", "c")
    assert out == []
