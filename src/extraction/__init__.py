"""Extraction module"""
try:
    from .extractor import FactExtractor, Fact
    __all__ = ["FactExtractor", "Fact"]
except ImportError:
    __all__ = []
