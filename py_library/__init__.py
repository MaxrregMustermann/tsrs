# SPDX-License-Identifier: MIT
# Copyright (c) 2026 MaxrregMustermann
#
# TRACE-SRS - Two-Rate Adaptive Consolidation Engine for Spaced Repetition
"""
TRACE-SRS: Two-Rate Adaptive Consolidation Engine for Spaced Repetition
========================================================================

A spaced repetition algorithm based on the Memory Chain Model (MCM) from
cognitive neuroscience, modeling hippocampal (fast) and neocortical (slow)
memory traces separately.

Quick Start
-----------
    >>> from trace_srs import TraceSRS, Card, Scheduler, Grade
    >>> 
    >>> # Create a scheduler with default parameters
    >>> scheduler = Scheduler()
    >>> 
    >>> # Create a new card
    >>> card = Card()
    >>> 
    >>> # After a successful review (Good grade)
    >>> card = scheduler.schedule(card, ok=True, t_days=0, grade=Grade.GOOD)
    >>> print(f"Next interval: {card.interval:.1f} days")
    >>> print(f"Retrievability: {card.retrievability:.1%}")

Main Classes
------------
- :class:`TraceSRS` - Core algorithm with low-level API
- :class:`Card` - High-level card representation with state
- :class:`Scheduler` - Scheduling logic using TRACE-SRS
- :class:`Grade` - Enum for review grades (AGAIN, HARD, GOOD, EASY)

For full documentation, see the docstrings of individual classes.
"""

from .tsrs import (
    TraceSRS,
    DEFAULT_PARAMETERS,
    Card,
    Scheduler,
    Grade,
)

__version__ = "0.1.0"
__author__ = "Claude, MaxrregMustermann"

__all__ = [
    "Card",
    "DEFAULT_PARAMETERS",
    "Grade",
    "Scheduler",
    "TraceSRS",
    "__version__",
]
