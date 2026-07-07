"""Pytest configuration for the tests/ directory."""

# test_quality_score.py is a script with a home-rolled runner: its test_*
# functions take a custom `t: TestResult` argument, so pytest collection
# errors on every one. Run it directly instead:
#     python tests/test_quality_score.py
collect_ignore = ["test_quality_score.py"]
