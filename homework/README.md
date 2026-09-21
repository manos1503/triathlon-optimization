# Course Assignments

Coursework for **Linear and Combinatorial Optimization** (University of Patras,
Prof. S. Daskalaki) — two problem sets of seven exercises each, set during the
semester.

These are **independent of the triathlon project** in the rest of this
repository; they live here only to keep all coursework for the module in one
place.

| | | |
| :--- | :--- | :--- |
| [`HW1/HW1_up1100830.pdf`](HW1/HW1_up1100830.pdf) | Assignment 1 — report | 41 pp. |
| [`HW1/code/`](HW1/code) | `ex_1.py` … `ex_7.py` | |
| [`HW2/HW2_up1100830.pdf`](HW2/HW2_up1100830.pdf) | Assignment 2 — report | 38 pp. |
| [`HW2/code/`](HW2/code) | `ex_1.py` … `ex_7.py` | |
| `*/outputs/` | captured console output of every script | |

Both PDFs open with a note listing, exercise by exercise, what was revised
relative to the first submission.

## Running

```bash
pip install -r requirements.txt
cd HW1/code && python3 ex_1.py      # likewise for every exercise
```

Each script is self-contained and prints exactly the results shown in the
corresponding exercise of the report. `ex_1.py` and `ex_2.py` of Assignment 1
additionally write their figures (`.png`) to the working directory.

## Implementation notes

- Wherever degeneracy has to be detected (zero basic variables, coincident
  vertices), the arithmetic is **exact rational** via `fractions.Fraction`, so
  no conclusion depends on floating-point rounding.
- **Simplex, dual Simplex, Phase I and Branch & Bound are implemented
  explicitly**, printing every tableau and every node. The solver (PuLP/CBC) is
  called only for the LP relaxations at the B&B nodes and as an independent
  check on the final answers.
- Libraries: numpy, scipy, matplotlib, pulp (CBC) — as used in the lectures.
  `fractions` and `itertools` are Python standard library.
