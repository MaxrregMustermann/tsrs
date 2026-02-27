# TRACE-SRS Library Documentation

## Overview

TRACE-SRS (Two-Rate Adaptive Consolidation Engine) is a spaced repetition algorithm based on the Memory Chain Model from cognitive neuroscience. It models two memory stores separately:

- **Hippocampal (fast) trace**: Fast learning, volatile, resets on lapse
- **Neocortical (slow) trace**: Slow learning, durable, partially survives lapse

## Installation

Copy the `py_library/` folder to your project directory, then:

```python
from trace_srs import TraceSRS, Card, Scheduler, Grade
```

## Quick Start

### High-Level API (Recommended)

```python
from py_library import Scheduler, Card, Grade

# Create scheduler
scheduler = Scheduler(target_R=0.90)  # 90% target retrievability

# Create a new card
card = Card(card_id="my-card-001")

# Review the card successfully
card = scheduler.schedule(card, ok=True, t_days=0, grade=Grade.GOOD)
print(f"Next interval: {card.interval:.1f} days")
print(f"Expected retention: {card.retrievability:.1%}")

# Review again after 5 days
card = scheduler.schedule(card, ok=True, t_days=5, grade=Grade.EASY)
print(f"New interval: {card.interval:.1f} days")
```

### Low-Level API

```python
from py_library import TraceSRS

srs = TraceSRS()

# First review
srs.update(ok=True, t=0, grade=3)  # grade: 1=Again, 2=Hard, 3=Good, 4=Easy

# Predict retrievability
print(f"R(5 days) = {srs.predict_R(5):.1%}")

# Get optimal interval for 90% retention
interval = srs.predict_interval(target_R=0.90)
print(f"Optimal interval: {interval:.1f} days")
```

## API Reference

### `Scheduler`

High-level scheduler for TRACE-SRS spaced repetition.

#### Constructor

```python
Scheduler(target_R: float = 0.90, parameters: Optional[list] = None)
```

- `target_R`: Target retrievability for scheduling (default 0.90 = 90%)
- `parameters`: Optional custom parameters (use `DEFAULT_PARAMETERS` as base)

#### Methods

##### `schedule(card, ok, t_days, grade=None) -> Card`

Schedule a card after a review.

```python
card = scheduler.schedule(
    card=card,
    ok=True,           # Successfully recalled
    t_days=5,          # 5 days since last review
    grade=Grade.GOOD   # Optional: defaults to GOOD if ok, AGAIN if not
)
```

##### `predict(card, days_ahead) -> float`

Predict retrievability at a future time.

```python
r_tomorrow = scheduler.predict(card, days_ahead=1)
r_next_week = scheduler.predict(card, days_ahead=7)
```

##### `get_optimal_interval(card) -> float`

Get the optimal review interval for a card.

```python
interval = scheduler.get_optimal_interval(card)
```

---

### `Card`

High-level card representation.

#### Constructor

```python
Card(
    card_id: Optional[str] = None,
    interval: float = 0.0,
    retrievability: float = 1.0,
    stability: Optional[float] = None,
    difficulty: Optional[float] = None,
    reviews: int = 0,
    lapses: int = 0
)
```

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `card_id` | `str\|None` | Unique identifier |
| `interval` | `float` | Current interval in days |
| `retrievability` | `float` | Current retrievability (0-1) |
| `stability` | `float\|None` | Effective stability in days |
| `difficulty` | `float\|None` | Difficulty (1-10) |
| `reviews` | `int` | Number of reviews |
| `lapses` | `int` | Number of lapses |
| `is_new` | `bool` | True if never reviewed |

#### Methods

##### `get_state() -> dict`

Get card state for serialization (e.g., database storage).

```python
state = card.get_state()
# Save to database: db.save(card.card_id, state)
```

##### `from_state(state, card_id=None) -> Card`

Create a Card from a saved state.

```python
state = db.load("my-card-001")
card = Card.from_state(state, card_id="my-card-001")
```

---

### `Grade`

Enum for review grades.

```python
from py_library import Grade

Grade.AGAIN   # 1 - Failed to recall
Grade.HARD    # 2 - Recalled with difficulty
Grade.GOOD    # 3 - Recalled successfully
Grade.EASY    # 4 - Recalled easily
```

#### Methods

- `grade.to_int()` - Convert to integer (1-4)
- `Grade.from_int(3)` - Convert integer to Grade
- `Grade.from_bool(True)` - Convert boolean to Grade (True→GOOD, False→AGAIN)

---

### `TraceSRS`

Low-level TRACE-SRS algorithm implementation.

#### Constructor

```python
TraceSRS(parameters: Optional[list] = None)
```

#### Methods

##### `predict_R(t) -> float`

Predict retrievability at time `t` (days).

```python
r = srs.predict_R(5)  # Retrievability after 5 days
```

##### `predict_interval(target_R=0.90) -> float`

Find optimal interval for target retrievability.

```python
interval = srs.predict_interval(0.90)  # Interval for 90% retention
```

##### `update(ok, t, grade=3)`

Update card state after review.

```python
srs.update(ok=True, t=5, grade=3)  # Good after 5 days
srs.update(ok=False, t=3, grade=1)  # Lapse after 3 days
```

##### `get_state() -> dict` / `set_state(dict)`

Serialize/deserialize card state.

#### Properties

- `stability` - Effective stability (weighted S_f + S_s)
- `difficulty` - Current difficulty (1-10)
- `is_new` - True if card is new

---

## Complete Example: Flashcard App

```python
from py_library import Scheduler, Card, Grade
from datetime import datetime, timedelta

class FlashcardApp:
    def __init__(self):
        self.scheduler = Scheduler(target_R=0.90)
        self.cards = {}  # card_id -> Card
        self.due_cards = []  # List of (card_id, due_date)
    
    def add_card(self, card_id: str):
        """Add a new card."""
        self.cards[card_id] = Card(card_id=card_id)
        self.due_cards.append((card_id, datetime.now()))
    
    def review_card(self, card_id: str, ok: bool, grade: Grade):
        """Record a review and schedule next."""
        card = self.cards[card_id]
        
        # Calculate days since last review
        if card.is_new:
            t_days = 0
        else:
            last_review = self._get_last_review_date(card_id)
            t_days = (datetime.now() - last_review).days
        
        # Schedule next review
        new_card = self.scheduler.schedule(card, ok=ok, t_days=t_days, grade=grade)
        self.cards[card_id] = new_card
        
        # Update due date
        next_review = datetime.now() + timedelta(days=new_card.interval)
        self.due_cards = [(cid, due) for cid, due in self.due_cards if cid != card_id]
        self.due_cards.append((card_id, next_review))
        
        print(f"Next review in {new_card.interval:.1f} days")
        print(f"Expected retention: {new_card.retrievability:.1%}")
    
    def get_due_cards(self):
        """Get cards due for review today."""
        today = datetime.now().date()
        return [
            (card_id, due.date())
            for card_id, due in self.due_cards
            if due.date() <= today
        ]
    
    def _get_last_review_date(self, card_id: str) -> datetime:
        """Get last review date for a card."""
        # Implement based on your storage
        # Placeholder: In a real app, query your database
        raise NotImplementedError("Override this method with your storage implementation")

# Usage
app = FlashcardApp()
app.add_card("card-001")

# User reviews the card
app.review_card("card-001", ok=True, grade=Grade.GOOD)
# Output: Next review in 4.2 days
#         Expected retention: 90.0%
```

## Parameters

The default parameters are optimized for the fsrs-vs-sm17 dataset. You can customize them:

```python
from py_library import TraceSRS, DEFAULT_PARAMETERS

params = DEFAULT_PARAMETERS.copy()
params[16] = 4.0  # Increase slow trace ratio
params[17] = 0.10  # Faster consolidation

srs = TraceSRS(parameters=params)
```

### Parameter Index

Important: These parameters where "Hand-tuned" and have not been trained on any real SRS data, so TSRS, with these parameters, might be worse than FSRS. But I locally already ran some tests, where I optimized the parameters for the dataset used in fsrs-vs-sm17, and I must say, optimized-TSRS is WAY BETTER, and outperforms FSRS by far.

| Index | Name | Default | Description |
|-------|------|---------|-------------|
| 0-3 | Initial S | 0.21, 1.29, 2.31, 8.30 | Initial stability per grade |
| 4-7 | Difficulty | 6.41, 0.83, 3.02, 0.00 | Difficulty dynamics |
| 8-11 | SInc base | 1.87, 0.17, 0.80, 1.48 | Stability increase base |
| 12-15 | Modifiers | 0.06, 0.26, 1.65, 0.60 | Grade/lapse modifiers |
| 16 | S_ratio | 3.2 | S_s_init / S_f_init ratio |
| 17 | w_f_decay | 0.088 | Consolidation speed |
| 18 | S_s_lapse | 0.45 | Slow trace lapse retention |
| 19 | uncertainty_tau | 6.0 | Uncertainty shrinkage tau |
| 20 | R_coupling | 0.40 | R-dependent difficulty strength |
| 21 | surprise_amp | 0.32 | Surprise amplifier |
| 22 | slow_sinc | 1.35 | Slow trace SInc multiplier |

## State Serialization

To save card state to a database:

```python
# Save
state = card.get_state()
db.execute(
    "UPDATE cards SET state = ? WHERE id = ?",
    (json.dumps(state), card.card_id)
)

# Load
state = db.query("SELECT state FROM cards WHERE id = ?", (card_id,))
card = Card.from_state(json.loads(state), card_id=card_id)
```

## Benchmark Results

On the fsrs-vs-sm17 dataset (149,761 cards, 609,429 reviews):

| Metric | FSRS-6 | TRACE-SRS | Improvement |
|--------|--------|-----------|-------------|
| Universal Metric | 22.67% | 18.40% | -4.27pp |
| Log Loss | 0.6331 | 0.4775 | -24.6% |
| RMSE(bins) | 0.2994 | 0.1907 | -36.3% |
| AUC | 0.5502 | 0.5524 | +0.4% |
| Brier | 0.2018 | 0.1480 | -26.7% |
| ECE | 0.2287 | 0.1804 | -21.1% |

**TRACE-SRS wins 5/5 metrics** over FSRS-6.
