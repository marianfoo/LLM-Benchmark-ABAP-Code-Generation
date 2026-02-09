# Benchmark Dataset

This directory contains the full benchmark inputs used to evaluate LLM ABAP
code generation, as described in
[arXiv:2601.15188](https://arxiv.org/abs/2601.15188).

## Structure

| Directory | Files | Description |
|-----------|------:|-------------|
| `prompts/` | 180 `.txt` | Natural-language task descriptions sent to the LLM as the initial user message. File naming: `<N>.txt` for HumanEval-derived tasks, `erp_<N>.txt` for SAP/ERP-specific tasks. |
| `abap_canonical_solution/` | 180 `.abap` | Reference ABAP implementations (one per prompt). These are **not** shown to the LLM; they exist for manual inspection and future automated comparison. |
| `abap_unittests/` | 180 `.abap` | ABAP unit-test classes uploaded to the SAP system to verify LLM-generated code. Each file corresponds to one prompt. |
| `abap_tables/` | 13 `.abap` | CDS-style table definitions required by the ERP-specific tasks. These tables are created in the SAP Docker environment before running tests. |

## Task categories

Tasks are classified into five categories (see `data/prompt_classification.csv`):

- **String Handling** -- string manipulation, parsing, formatting
- **List or Array Operation** -- internal table / array logic
- **Mathematical Calculation** -- numeric computation, math functions
- **Logical Condition** -- boolean / conditional logic
- **ABAP Database Operation** -- SQL, CDS views, table operations (ERP tasks)

## Provenance

- HumanEval-derived prompts (`0.txt` -- `162.txt`) are ABAP adaptations of
  selected problems from the
  [OpenAI HumanEval](https://github.com/openai/human-eval) benchmark,
  rewritten as SAP global-class specifications.
- ERP-specific prompts (`erp_000.txt` -- `erp_014.txt`) are original tasks
  created for this benchmark, exercising SAP-specific database and business
  logic patterns.
- All canonical solutions, unit tests, and table definitions were written by
  the benchmark authors.

## License

This dataset is distributed under the same MIT license as the rest of this
repository. See [LICENSE](../LICENSE) for details.
