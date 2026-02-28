# TRACE-SRS

Two-Rate Adaptive Consolidation Engine for Spaced Repetition

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)

## About

TRACE-SRS is a spaced repetition algorithm based on the Memory Chain Model from cognitive neuroscience. It models human memory using two separate traces:

- **Fast trace**: hippocampal rapid encoding (volatile)
- **Slow trace**: neocortical gradual consolidation (durable)

The dual-trace architecture explains why relearning is faster than initial learning and why lapses cause sudden performance drops.

## Usage

See `py_library/USAGE.md`

## Benchmarks

| Metric | FSRS-6 | TRACE-SRS | Improvement |
|--------|--------|-----------|-------------|
| Universal Metric* | 22.67% | 18.40% | -4.27pp |
| Log Loss | 0.6331 | 0.4775 | -24.6% |
| RMSE(bins) | 0.2994 | 0.1907 | -36.3% |
| AUC | 0.5502 | 0.5524 | +0.4% |
| Brier Score | 0.2018 | 0.1480 | -26.7% |

**According to `comparison/osr_run_comparison.py`**

## Contributing

Contributions are welcome. Please feel free to report bugs, suggest features, or submit pull requests.

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Inspired by the Memory Chain Model from cognitive neuroscience research
- Built upon insights from FSRS, SuperMemo, and spaced repetition research
- Benchmark dataset from open-spaced-repetition/fsrs-vs-sm17
