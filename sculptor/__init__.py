"""
Sculptor: A library for extracting structured data from text using LLMs.
"""

from .sculptor import Sculptor, ALLOWED_TYPES, DEFAULT_SYSTEM_PROMPT
from .sculptor_pipeline import SculptorPipeline

__version__ = "0.1.0"

__all__ = [
    "Sculptor",
    "SculptorPipeline",
]