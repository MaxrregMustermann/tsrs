# SPDX-License-Identifier: MIT
# Copyright (c) 2026 MaxrregMustermann
#
# TRACE-SRS - Two-Rate Adaptive Consolidation Engine for Spaced Repetition
"""
TRACE-SRS: Two-Rate Adaptive Consolidation Engine for Spaced Repetition

A spaced repetition algorithm based on the Memory Chain Model (MCM) from
cognitive neuroscience, modeling hippocampal (fast) and neocortical (slow)
memory traces separately.
"""

import math
from typing import Optional, Tuple


DEFAULT_PARAMETERS = [
    # w0-w3: initial S per grade (same as FSRS-6)
    0.212, 1.2931, 2.3065, 8.2956,
    # w4-w7: difficulty dynamics (same base)
    6.4133, 0.8334, 3.0194, 0.001,
    # w8-w11: SInc base
    1.8722, 0.1666, 0.796, 1.4835,
    # w12-w15: grade/lapse modifiers
    0.0614, 0.2629, 1.6483, 0.6014,
    # w16: fast-to-slow ratio (S_s_init = S_f_init * w16)
    3.2,
    # w17: consolidation speed (w_f decay per successful review)
    0.088,
    # w18: slow-trace lapse retention fraction
    0.45,
    # w19: uncertainty tau (same as NOVA)
    6.0,
    # w20: R-coupling strength for difficulty (NOVA improvement 1)
    0.40,
    # w21: surprise amplifier (NOVA improvement 3)
    0.32,
    # w22: slow-trace SInc multiplier (slow trace grows more per review)
    1.35,
]


class TraceSRS:
    """
    TRACE-SRS: Two-Rate Adaptive Consolidation Engine for Spaced Repetition.

    Based on the Memory Chain Model (MCM) from cognitive neuroscience:
    - Models two memory stores: hippocampal (fast) and neocortical (slow)
    - Forgetting curve: R(t) = w_f * 0.9^(t/S_f) + (1-w_f) * 0.9^(t/S_s)
    - Consolidation shifts weight from fast to slow trace over time

    Key features:
    - R-dependent difficulty updates
    - Uncertainty shrinkage for new cards
    - Surprise-amplified stability increase
    - Partial slow-trace survival on lapse (explains faster relearning)

    Example:
        >>> srs = TraceSRS()
        >>> # First review: grade 3 (good), t=0
        >>> srs.update(ok=True, t=0, grade=3)
        >>> # Predict retrievability after 5 days
        >>> r = srs.predict_R(5)
        >>> print(f"Retrievability: {r:.2%}")
    """

    PRIOR_R = 0.68  # Conservative population prior for uncertainty shrinkage

    def __init__(self, parameters: Optional[list] = None):
        """
        Initialize TRACE-SRS with optional custom parameters.

        Args:
            parameters: List of 23 parameters. If None, uses default parameters.
        """
        base = DEFAULT_PARAMETERS if parameters is None else parameters
        self.w = list(base)

        # Card state variables
        self.S_f: Optional[float] = None  # Fast trace (hippocampal) stability
        self.S_s: Optional[float] = None  # Slow trace (neocortical) stability
        self.w_f: float = 0.72  # Fast trace mixture weight
        self.D: Optional[float] = None  # Difficulty
        self.n: int = 0  # Review count

    def _R_point(self, t: float) -> float:
        """
        Raw dual-store retrievability prediction without uncertainty correction.

        Args:
            t: Elapsed time in days

        Returns:
            Retrievability probability (0 to 1)
        """
        if self.S_f is None:
            return self.PRIOR_R

        w_s = 1.0 - self.w_f
        r_f = math.pow(0.9, t / max(0.1, self.S_f))
        r_s = math.pow(0.9, t / max(0.1, self.S_s))
        return min(1.0, max(1e-6, self.w_f * r_f + w_s * r_s))

    def predict_R(self, t: float) -> float:
        """
        Predict retrievability at time t with uncertainty shrinkage.

        For new cards (n < ~15 reviews), predictions are shrunk toward
        the conservative prior to avoid overconfident early predictions.

        Args:
            t: Elapsed time in days since last review

        Returns:
            Predicted retrievability probability (0 to 1)
        """
        R_raw = self._R_point(t)
        conf = 1.0 - math.exp(-self.n / self.w[19])
        return R_raw * conf + self.PRIOR_R * (1.0 - conf)

    def predict_interval(self, target_R: float = 0.90) -> float:
        """
        Find the interval that achieves target retrievability.

        Uses bisection search to find t such that predict_R(t) = target_R.

        Args:
            target_R: Target retrievability (default 0.90 = 90%)

        Returns:
            Optimal interval in days
        """
        if self.S_f is None:
            return 1.0  # Default for new cards

        lo, hi = 0.0, 3650.0
        for _ in range(50):  # Bisection iterations
            mid = (lo + hi) / 2.0
            if self.predict_R(mid) > target_R:
                lo = mid
            else:
                hi = mid
        return max(1.0, (lo + hi) / 2.0)

    def _D0(self, g: int) -> float:
        """Initial difficulty for a given grade."""
        w = self.w
        return min(10.0, max(1.0, w[4] - math.exp(w[5] * (g - 1)) + 1))

    def _next_D(self, D: float, g: int, R: float) -> float:
        """
        R-dependent difficulty update (NOVA improvement 1).

        Scales difficulty delta by outcome surprise:
        - Recall at low R → difficulty decreases more
        - Lapse at high R → difficulty increases more
        """
        w = self.w
        surprise_f = 1.0 + w[20] * abs(R - 0.5) * 2.0
        delta = -w[6] * (g - 3) * surprise_f
        Dp = D + delta * (10.0 - D) / 9.0
        return min(10.0, max(1.0, w[7] * self._D0(4) + (1.0 - w[7]) * Dp))

    def _S_fast_recall(self, S_f: float, D: float, R: float, g: int) -> float:
        """
        Fast trace (hippocampal) update on successful recall.

        Includes surprise amplifier (NOVA improvement 3):
        - Low R recalls produce larger stability increases
        """
        w = self.w
        hp = w[15] if g == 2 else 1.0  # Hard penalty
        eb = w[16] if g == 4 else 1.0  # Easy bonus

        base_sinc = (math.exp(w[8]) * (11 - D) * pow(max(0.01, S_f), -w[9]) *
                     (math.exp(w[10] * (1 - R)) - 1) * hp * eb + 1)

        surprise_amp = math.exp(w[21] * max(0.0, 1.0 - R))
        sinc = base_sinc * (1.0 + (surprise_amp - 1.0) * (1.0 - R))

        return max(S_f, S_f * sinc)

    def _S_slow_recall(self, S_f: float, S_s: float, D: float, R: float, g: int) -> float:
        """
        Slow trace (neocortical) update on successful recall.

        The slow trace grows faster per review, modeling neocortical
        consolidation. This explains why relearning is faster than
        initial learning.
        """
        w = self.w
        hp = w[15] if g == 2 else 1.0
        eb = w[16] if g == 4 else 1.0

        # Slow trace has weaker S decay and R dependency
        base_sinc = (math.exp(w[8]) * (11 - D) * pow(max(0.01, S_s), -w[9] * 0.7) *
                     (math.exp(w[10] * (1 - R) * 0.8) - 1) * hp * eb + 1)

        slow_sinc = base_sinc * w[22]  # Slow trace consolidation multiplier
        new_S_s = max(S_s, S_s * slow_sinc)

        # Upper bound: slow trace can't exceed fast trace by more than ratio
        return min(new_S_s, S_f * w[16] * 5.0)

    def _S_fast_lapse(self, D: float) -> float:
        """
        Fast trace (hippocampal) reset on lapse.

        Resets almost fully, explaining sudden performance drop.
        """
        w = self.w
        return max(0.1, w[11] * (11 - D) / 10.0)

    def _S_slow_lapse(self, S_s: float, R: float) -> float:
        """
        Slow trace (neocortical) partial survival on lapse.

        Retains 40-70% of slow trace, explaining why relearning
        is faster than initial learning.
        """
        return max(0.1, S_s * self.w[18])

    def update(self, ok: bool, t: float, grade: int = 3) -> None:
        """
        Update card state after a review.

        Args:
            ok: Whether the review was successful (recalled)
            t: Elapsed time in days since last review
            grade: Review grade (1=Again, 2=Hard, 3=Good, 4=Easy)
                   Default is 3 (Good) for binary recall/lapse scenarios.
        """
        g = grade if grade in [1, 2, 3, 4] else (3 if ok else 1)
        R_raw = self._R_point(t)  # Use raw R for updates, not shrunk
        self.n += 1

        if self.S_f is None:
            # First review: initialize both traces
            self.S_f = self.w[g - 1]
            self.S_s = self.S_f * self.w[16]  # Slow trace starts w16x larger
            self.D = self._D0(g)
        else:
            if ok:
                # Successful recall: both traces grow
                new_Sf = max(0.1, self._S_fast_recall(self.S_f, self.D, R_raw, g))
                new_Ss = max(0.1, self._S_slow_recall(new_Sf, self.S_s, self.D, R_raw, g))
                self.S_f = new_Sf
                self.S_s = new_Ss
                # Consolidation: shift weight from fast to slow trace
                self.w_f = max(0.05, self.w_f * (1.0 - self.w[17]))
            else:
                # Lapse: fast trace resets, slow trace persists
                self.S_f = self._S_fast_lapse(self.D)
                self.S_s = self._S_slow_lapse(self.S_s, R_raw)
                # Reconsolidation: lapse temporarily boosts fast trace weight
                self.w_f = min(0.85, self.w_f + 0.20)

            self.D = self._next_D(self.D, g, R_raw)

    def get_state(self) -> dict:
        """
        Get current card state as a dictionary.

        Returns:
            Dictionary with S_f, S_s, w_f, D, n values
        """
        return {
            "S_f": self.S_f,
            "S_s": self.S_s,
            "w_f": self.w_f,
            "D": self.D,
            "n": self.n,
        }

    def set_state(self, state: dict) -> None:
        """
        Set card state from a dictionary.

        Args:
            state: Dictionary with S_f, S_s, w_f, D, n values
        """
        self.S_f = state.get("S_f")
        self.S_s = state.get("S_s")
        self.w_f = state.get("w_f", 0.72)
        self.D = state.get("D")
        self.n = state.get("n", 0)

    def reset(self) -> None:
        """Reset card to initial state (as if new)."""
        self.S_f = None
        self.S_s = None
        self.w_f = 0.72
        self.D = None
        self.n = 0

    @property
    def stability(self) -> Optional[float]:
        """
        Get effective stability (fast trace weighted).

        For compatibility with single-trace models like FSRS.
        """
        if self.S_f is None:
            return None
        return self.w_f * self.S_f + (1.0 - self.w_f) * self.S_s

    @property
    def difficulty(self) -> Optional[float]:
        """Get current difficulty."""
        return self.D

    @property
    def is_new(self) -> bool:
        """Check if card is new (never reviewed)."""
        return self.S_f is None


# ─────────────────────────────────────────────────────────────────────────────
# HIGH-LEVEL API
# ─────────────────────────────────────────────────────────────────────────────

from enum import Enum
from dataclasses import dataclass, field


class Grade(Enum):
    """
    Review grades for spaced repetition.

    Members:
        AGAIN: Failed to recall (resets interval)
        HARD: Recalled with difficulty (small interval increase)
        GOOD: Recalled successfully (normal interval increase)
        EASY: Recalled easily (large interval increase)
    """
    AGAIN = 1
    HARD = 2
    GOOD = 3
    EASY = 4

    def to_int(self) -> int:
        """Convert grade to integer (1-4)."""
        return self.value

    @classmethod
    def from_int(cls, value: int) -> "Grade":
        """Convert integer to Grade."""
        return cls(value)

    @classmethod
    def from_bool(cls, ok: bool) -> "Grade":
        """Convert boolean outcome to Grade (True=GOOD, False=AGAIN)."""
        return cls.GOOD if ok else cls.AGAIN


@dataclass
class Card:
    """
    High-level card representation for spaced repetition.

    A card represents a single flashcard or learning item with its
    current learning state.

    Attributes:
        card_id: Optional unique identifier for the card
        interval: Current interval in days (0 for new cards)
        retrievability: Current retrievability estimate (0-1)
        stability: Effective stability in days
        difficulty: Current difficulty (1-10, higher = harder)
        reviews: Number of successful reviews
        lapses: Number of lapses (failed recalls)
        is_new: True if card has never been reviewed
        state: Raw state dictionary for serialization

    Example:
        >>> card = Card()
        >>> print(f"New card: interval={card.interval}, is_new={card.is_new}")
        New card: interval=0, is_new=True
    """
    card_id: Optional[str] = None
    interval: float = 0.0
    retrievability: float = 1.0
    stability: Optional[float] = None
    difficulty: Optional[float] = None
    reviews: int = 0
    lapses: int = 0
    _state: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_state(cls, state: dict, card_id: Optional[str] = None) -> "Card":
        """
        Create a Card from a state dictionary.

        Args:
            state: State dict from Card.get_state() or Scheduler
            card_id: Optional card identifier

        Returns:
            New Card instance with restored state
        """
        srs = TraceSRS()
        srs.set_state(state)

        # Calculate current interval and retrievability
        n = state.get("n", 0)
        if n == 0:
            interval = 0.0
            retrievability = 1.0
        else:
            # Use the last interval that was scheduled
            interval = state.get("last_interval", 0.0)
            retrievability = srs.predict_R(interval)

        lapses = state.get("lapses", 0)

        return cls(
            card_id=card_id,
            interval=interval,
            retrievability=retrievability,
            stability=srs.stability,
            difficulty=srs.difficulty,
            reviews=n,
            lapses=lapses,
            _state=state,
        )

    def get_state(self) -> dict:
        """
        Get the card's state for serialization.

        Returns:
            Dictionary containing all card state
        """
        state = {
            "S_f": self._state.get("S_f"),
            "S_s": self._state.get("S_s"),
            "w_f": self._state.get("w_f", 0.72),
            "D": self._state.get("D"),
            "n": self.reviews,
            "lapses": self.lapses,
            "last_interval": self.interval,
        }
        return state

    @property
    def is_new(self) -> bool:
        """True if card has never been reviewed."""
        return self.reviews == 0


class Scheduler:
    """
    High-level scheduler for TRACE-SRS spaced repetition.

    The Scheduler provides a simple interface for scheduling cards
    using the TRACE-SRS algorithm.

    Attributes:
        srs: Underlying TraceSRS instance
        target_R: Target retrievability for scheduling (default 0.90)

    Example:
        >>> scheduler = Scheduler(target_R=0.90)
        >>> card = Card()
        >>> 
        >>> # Schedule after a review
        >>> card = scheduler.schedule(card, ok=True, t_days=0, grade=Grade.GOOD)
        >>> print(f"Next interval: {card.interval:.1f} days")
        >>> print(f"Expected retrievability: {card.retrievability:.1%}")
    """

    def __init__(self, target_R: float = 0.90, parameters: Optional[list] = None):
        """
        Initialize the scheduler.

        Args:
            target_R: Target retrievability for interval scheduling (default 0.90)
            parameters: Optional custom parameters for TRACE-SRS
        """
        self.target_R = target_R
        self.srs = TraceSRS(parameters=parameters)

    def schedule(
        self,
        card: Card,
        ok: bool,
        t_days: float,
        grade: Optional[Grade] = None,
    ) -> Card:
        """
        Schedule a card after a review.

        Args:
            card: The card being reviewed
            ok: Whether the review was successful (recalled)
            t_days: Days since the last review
            grade: Optional grade (defaults to GOOD if ok, AGAIN if not)

        Returns:
            New Card instance with updated state and scheduling info
        """
        if grade is None:
            grade = Grade.from_bool(ok)

        # Create new SRS instance and restore state
        srs = TraceSRS(parameters=self.srs.w)
        state = card.get_state()
        if not card.is_new:
            srs.set_state(state)

        # Update with review result
        srs.update(ok=ok, t=t_days, grade=grade.to_int())

        # Calculate new interval
        if ok:
            new_interval = srs.predict_interval(target_R=self.target_R)
            new_retrievability = srs.predict_R(new_interval)
            new_reviews = card.reviews + 1
            new_lapses = card.lapses
        else:
            new_interval = 1.0  # Reset to 1 day on lapse
            new_retrievability = srs.predict_R(1.0)
            new_reviews = card.reviews
            new_lapses = card.lapses + 1

        # Create new card with updated state
        new_state = srs.get_state()
        new_state["lapses"] = new_lapses
        new_state["last_interval"] = new_interval

        return Card(
            card_id=card.card_id,
            interval=new_interval,
            retrievability=new_retrievability,
            stability=srs.stability,
            difficulty=srs.difficulty,
            reviews=new_reviews,
            lapses=new_lapses,
            _state=new_state,
        )

    def predict(self, card: Card, days_ahead: float) -> float:
        """
        Predict retrievability at a future time.

        Args:
            card: The card to predict
            days_ahead: Days from now to predict

        Returns:
            Predicted retrievability (0-1)
        """
        if card.is_new:
            return self.srs.PRIOR_R

        srs = TraceSRS(parameters=self.srs.w)
        srs.set_state(card.get_state())
        return srs.predict_R(days_ahead)

    def get_optimal_interval(self, card: Card) -> float:
        """
        Get the optimal review interval for a card.

        Args:
            card: The card to schedule

        Returns:
            Optimal interval in days for target retrievability
        """
        if card.is_new:
            return 1.0

        srs = TraceSRS(parameters=self.srs.w)
        srs.set_state(card.get_state())
        return srs.predict_interval(target_R=self.target_R)
