#!/usr/bin/env python3
"""Build objective ABAP understanding benchmark items from existing dataset assets.

Each prompt becomes one composite understanding item with deterministic answers
extracted from:
- canonical solution code
- ABAP unit test code

Output format: JSONL (one item per line)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "dataset"
PROMPTS_DIR = DATASET_DIR / "prompts"
CANON_DIR = DATASET_DIR / "abap_canonical_solution"
TESTS_DIR = DATASET_DIR / "abap_unittests"
CLASSIFICATION_CSV = REPO_ROOT / "data" / "prompt_classification.csv"


def _prompt_sort_key(stem: str) -> tuple[int, int]:
    if stem.startswith("erp_"):
        try:
            return (1, int(stem.split("_", 1)[1]))
        except (ValueError, IndexError):
            return (1, 999999)
    try:
        return (0, int(stem))
    except ValueError:
        return (0, 999999)


def _class_stem_from_prompt_id(prompt_id: str) -> str:
    if prompt_id.startswith("erp_"):
        return f"z_humaneval_{prompt_id}"
    return f"z_humaneval_{int(prompt_id):03d}"


def _load_categories() -> dict[str, list[str]]:
    if not CLASSIFICATION_CSV.exists():
        return {}

    category_map = {
        "String Handling": "String Handling",
        "List & Array Operations": "List or Array Operation",
        "List or Array Operation": "List or Array Operation",
        "Mathematical Calculations": "Mathematical Calculation",
        "Mathematical Calculation": "Mathematical Calculation",
        "Logical Checks": "Logical Condition",
        "Logical Condition": "Logical Condition",
        "Database Operations (ABAP)": "ABAP Database Operation",
        "ABAP Database Operation": "ABAP Database Operation",
    }

    out: dict[str, list[str]] = {}
    with CLASSIFICATION_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            raw_id = row.get("HumanEval/Nr") or row.get("Nr")
            if raw_id is None:
                continue
            prompt_id = str(raw_id).strip()
            if prompt_id.isdigit():
                prompt_id = str(int(prompt_id))

            labels: list[str] = []
            for src_col, normalized in category_map.items():
                if src_col in row and str(row[src_col]).strip().upper() == "X":
                    labels.append(normalized)

            # Stable unique order
            deduped = []
            seen = set()
            for label in labels:
                if label not in seen:
                    deduped.append(label)
                    seen.add(label)
            out[prompt_id] = deduped

    return out


def _extract_method_name(src: str) -> str:
    # Primary: class-methods declaration
    m = re.search(r"CLASS-METHODS\s*:?\s*([A-Za-z0-9_]+)", src, re.IGNORECASE)
    if m:
        return m.group(1).lower()

    # Fallback: first method implementation name
    m = re.search(r"METHOD\s+([A-Za-z0-9_]+)\s*\.", src, re.IGNORECASE)
    if m:
        return m.group(1).lower()

    return ""


def _extract_return_param_name(src: str) -> str:
    m = re.search(
        r"RETURNING\s+VALUE\s*\(\s*([A-Za-z0-9_]+)\s*\)",
        src,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).lower()
    return ""


def _extract_loop_count(src: str) -> int:
    # Count LOOP AT occurrences to avoid counting ENDLOOP.
    return len(re.findall(r"\bLOOP\s+AT\b", src, re.IGNORECASE))


def _extract_has_select(src: str) -> bool:
    return bool(re.search(r"\bSELECT\b", src, re.IGNORECASE))


def _extract_test_method_count(test_src: str) -> int:
    # "FOR TESTING" appears once on the class definition line and then once per
    # test method declaration. We want only method declarations here.
    total_for_testing = len(re.findall(r"\bFOR\s+TESTING\b", test_src, re.IGNORECASE))
    return max(total_for_testing - 1, 0)


def _extract_assert_count(test_src: str) -> int:
    return len(re.findall(r"cl_abap_unit_assert=>", test_src, re.IGNORECASE))


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def build_items(limit: int | None = None) -> tuple[list[dict[str, Any]], list[str]]:
    categories = _load_categories()

    prompt_files = sorted(PROMPTS_DIR.glob("*.txt"), key=lambda p: _prompt_sort_key(p.stem))
    if limit is not None:
        prompt_files = prompt_files[:limit]

    items: list[dict[str, Any]] = []
    warnings: list[str] = []

    for prompt_path in prompt_files:
        prompt_id = prompt_path.stem
        class_stem = _class_stem_from_prompt_id(prompt_id)

        canonical_path = CANON_DIR / f"{class_stem}.abap"
        unittest_path = TESTS_DIR / f"{class_stem}_test.abap"

        if not canonical_path.exists() or not unittest_path.exists():
            warnings.append(
                f"Missing asset(s) for prompt {prompt_id}: canonical={canonical_path.exists()}, unittest={unittest_path.exists()}"
            )
            continue

        canonical_src = _read(canonical_path)
        unittest_src = _read(unittest_path)

        method_name = _extract_method_name(canonical_src)
        return_param_name = _extract_return_param_name(canonical_src)

        if not method_name:
            warnings.append(f"No method name extracted for {prompt_id}")
        if not return_param_name:
            warnings.append(f"No return parameter extracted for {prompt_id}")

        expected = {
            "method_name": method_name,
            "return_param_name": return_param_name,
            "loop_count": _extract_loop_count(canonical_src),
            "has_select": _extract_has_select(canonical_src),
            "test_method_count": _extract_test_method_count(unittest_src),
            "assert_call_count": _extract_assert_count(unittest_src),
        }

        item = {
            "item_id": prompt_id,
            "prompt_id": prompt_id,
            "prompt_file": str(prompt_path.relative_to(REPO_ROOT)),
            "canonical_file": str(canonical_path.relative_to(REPO_ROOT)),
            "unittest_file": str(unittest_path.relative_to(REPO_ROOT)),
            "categories": categories.get(prompt_id, []),
            "question_type": "composite_abap_understanding_v1",
            "expected": expected,
            "answer_schema": {
                "method_name": "string",
                "return_param_name": "string",
                "loop_count": "integer",
                "has_select": "boolean",
                "test_method_count": "integer",
                "assert_call_count": "integer",
            },
        }
        items.append(item)

    return items, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description="Build ABAP understanding benchmark items from existing dataset assets.")
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "data" / "understanding_items.jsonl"),
        help="Output JSONL file path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for quick tests.",
    )
    args = parser.parse_args()

    items, warnings = build_items(limit=args.limit)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Wrote {output_path} ({len(items)} items)")
    if warnings:
        print(f"Warnings: {len(warnings)}")
        for warning in warnings[:20]:
            print(f"  - {warning}")
        if len(warnings) > 20:
            print(f"  ... and {len(warnings) - 20} more")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
