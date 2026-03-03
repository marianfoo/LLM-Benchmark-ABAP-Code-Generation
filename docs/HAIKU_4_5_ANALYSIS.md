# Claude Haiku 4.5 — Benchmark Analysis

Date: 2026-03-01

## Summary

`claude-haiku-4-5-20251001` achieves only **~5.7% success** across all feedback rounds, making it the lowest-performing model in the benchmark. This is not a general capability issue — it is caused by a single systematic ABAP syntax mistake the model cannot recover from.

---

## Results at a Glance

| Metric | Value |
|---|---|
| Success (unit tests passed) | 103 / 1800 (5.7%) |
| Maxed out (all 6 rounds failed) | 1697 / 1800 (94.3%) |
| Error stage — syntax | 95.1% of all rounds |
| Error stage — unit test | 4.2% of all rounds |
| Error stage — success | 0.7% of all rounds |

---

## Root Cause: CDS Type Notation in Classical ABAP

Haiku consistently generates `abap.*` type annotations inside classical ABAP class implementations, for example:

```abap
DATA lt_supervisors TYPE TABLE OF abap.char(20).
DATA lv_ceo_id      TYPE abap.char(20).
```

`abap.char(20)` is valid syntax in **ABAP CDS** (Core Data Services — the SQL-layer annotation language). It is **illegal** in classical ABAP class implementation where the correct equivalent would be `c LENGTH 20`, `string`, or a custom DDIC type.

When the SAP NetWeaver 7.5 parser encounters `abap.char(20)` inside a `METHOD` body, it loses track of the class structure. The reported error is:

```
"DEFINITION" or "IMPLEMENTATION" expected after "METHODS". (line 3)
```

This is a **misleading cascading error** — the real problem is inside the method body, but the parser reports it as a structural issue in the method declaration on line 3.

---

## Why Haiku Never Escapes the Feedback Loop

Haiku reads the error `"DEFINITION" or "IMPLEMENTATION" expected after "METHODS"` and correctly infers the problem is with the method declaration. Across all 6 rounds it only adjusts the formatting of the `RETURNING VALUE(...)` clause:

| Round | Change made | Still fails? |
|---|---|---|
| 0 → 1 | Moves `.` to a new line after `TYPE string` | Yes |
| 1 → 2 | Reverts `.` placement | Yes |
| 2 → 3 | One-lines the `RETURNING` clause | Yes |
| 3 → 4 | Splits `RETURNING` onto its own line | Yes |
| 4 → 5 | Moves `.` again | Yes |

The `abap.char(20)` inside the method body is **never touched** because the error message does not point there. Haiku lacks the reasoning to connect the misleading parser error to its true origin deeper in the code.

---

## Scale of the Problem

Roughly 3.7% of Round 0 responses directly contain `abap.*` types. However, many more responses contain other classical ABAP syntax errors that also prevent passing syntax check, explaining the overall 95.1% syntax failure rate across all rounds.

---

## Comparison with Claude Opus 4.5

| Model | R0 success | R5 success |
|---|---|---|
| `claude-opus-4-5-20251101` | ~46% | ~79% |
| `claude-haiku-4-5-20251001` | ~4% | ~6% |

Opus 4.5 generates valid classical ABAP from round 0 and uses feedback rounds effectively. Haiku generates structurally plausible but syntactically invalid ABAP that does not improve under feedback.

---

## Conclusion

Haiku 4.5 conflates CDS type notation with classical ABAP types. This is a training/fine-tuning gap specific to classical ABAP — Haiku's general coding ability is strong, but it has insufficient exposure to or disambiguation between the two ABAP dialects (CDS vs. classical). The feedback loop mechanism cannot compensate because the error message from SAP misdirects the model away from the actual bug.
