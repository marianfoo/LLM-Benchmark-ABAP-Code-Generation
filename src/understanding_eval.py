#!/usr/bin/env python3
"""Run ABAP understanding benchmark with iterative feedback rounds.

This benchmark is objective and auto-scored. Each item asks the model to extract
structured facts from ABAP canonical code + ABAP unit tests.

Output:
    data/<model>_understanding_predictions.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from dotenv import load_dotenv

# Ensure src/ is on sys.path so we can import siblings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

from llms import API_PROVIDERS, MODELS_TO_RUN, RunnableModel, create_anthropic_client, get_provider_api_key
from abap1_orchestration import ABAP1OrchestrationClient


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ITEMS_FILE = REPO_ROOT / "data" / "understanding_items.jsonl"

SYSTEM_PROMPT = (
    "You are an expert ABAP code analyst. "
    "Answer only with strict JSON. No markdown fences, no explanation."
)

FEEDBACK_PROMPT = (
    "The previous JSON answer was incorrect for one or more fields. "
    "Re-check the same context and return corrected JSON only with exactly the required keys."
)

OUTPUT_KEYS = [
    "method_name",
    "return_param_name",
    "loop_count",
    "has_select",
    "test_method_count",
    "assert_call_count",
]


def _normalize_model_name(name: str) -> str:
    return name.replace(":", "_")


def _read_text(repo_relative_path: str, cache: dict[str, str]) -> str:
    if repo_relative_path in cache:
        return cache[repo_relative_path]
    path = REPO_ROOT / repo_relative_path
    content = path.read_text(encoding="utf-8")
    cache[repo_relative_path] = content
    return content


def _get_model_info(model_name: str) -> RunnableModel | None:
    model_name_lower = model_name.lower()
    for model in MODELS_TO_RUN:
        n = model["name"].lower()
        if n == model_name_lower or model_name_lower in n:
            return model
    return None


def _load_items(items_file: Path, limit_items: int | None = None) -> list[dict[str, Any]]:
    if not items_file.exists():
        raise FileNotFoundError(f"Items file not found: {items_file}")

    items: list[dict[str, Any]] = []
    with items_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))

    if limit_items is not None:
        items = items[:limit_items]

    return items


def _load_existing_keys(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()

    keys = set()
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            item_id = str(record.get("item_id", ""))
            repetition = int(record.get("repetition", -1))
            if item_id and repetition >= 0:
                keys.add(f"{item_id}|{repetition}")
    return keys


def _append_record(output_path: Path, record: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_\-]*\n", "", t)
        t = re.sub(r"\n```$", "", t)
    return t.strip()


def _extract_json_dict(raw: str) -> dict[str, Any] | None:
    cleaned = _strip_code_fences(raw)

    # Direct parse
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Parse largest object-like span
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = cleaned[start : end + 1]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    # Fallback: key: value lines
    parsed: dict[str, Any] = {}
    for line in cleaned.splitlines():
        m = re.match(r"^\s*\"?([A-Za-z0-9_]+)\"?\s*[:=]\s*(.+?)\s*$", line)
        if not m:
            continue
        key = m.group(1)
        value = m.group(2).strip().strip(",")
        parsed[key] = value.strip().strip("\"")

    return parsed or None


def _to_norm_str(value: Any) -> str:
    if value is None:
        return ""
    s = str(value).strip().lower()
    s = s.strip("\"'")
    s = re.sub(r"\s+", "", s)
    return s


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)

    s = str(value)
    m = re.search(r"-?\d+", s)
    if not m:
        return None
    try:
        return int(m.group(0))
    except ValueError:
        return None


def _to_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)

    s = str(value).strip().lower()
    s = s.strip("\"'")

    true_values = {"true", "yes", "y", "1"}
    false_values = {"false", "no", "n", "0"}

    if s in true_values:
        return True
    if s in false_values:
        return False
    return None


def _evaluate(parsed: dict[str, Any] | None, expected: dict[str, Any]) -> tuple[dict[str, bool], bool, dict[str, Any]]:
    parsed = parsed or {}

    method_name = _to_norm_str(parsed.get("method_name"))
    return_param_name = _to_norm_str(parsed.get("return_param_name"))
    loop_count = _to_int(parsed.get("loop_count"))
    has_select = _to_bool(parsed.get("has_select"))
    test_method_count = _to_int(parsed.get("test_method_count"))
    assert_call_count = _to_int(parsed.get("assert_call_count"))

    normalized = {
        "method_name": method_name,
        "return_param_name": return_param_name,
        "loop_count": loop_count,
        "has_select": has_select,
        "test_method_count": test_method_count,
        "assert_call_count": assert_call_count,
    }

    field_correct = {
        "method_name": method_name == _to_norm_str(expected.get("method_name")),
        "return_param_name": return_param_name == _to_norm_str(expected.get("return_param_name")),
        "loop_count": loop_count == _to_int(expected.get("loop_count")),
        "has_select": has_select == _to_bool(expected.get("has_select")),
        "test_method_count": test_method_count == _to_int(expected.get("test_method_count")),
        "assert_call_count": assert_call_count == _to_int(expected.get("assert_call_count")),
    }

    overall = all(field_correct.values())
    return field_correct, overall, normalized


def _build_initial_user_prompt(item: Mapping[str, Any], prompt_text: str, canonical_src: str, unittest_src: str) -> str:
    expected_schema = {
        "method_name": "string",
        "return_param_name": "string",
        "loop_count": "integer",
        "has_select": "boolean",
        "test_method_count": "integer",
        "assert_call_count": "integer",
    }

    return (
        "ABAP understanding benchmark task.\n"
        "You will analyze a task prompt, canonical ABAP implementation, and ABAP unit test.\n"
        "Extract the requested structured facts.\n\n"
        "Output requirements:\n"
        "- Return exactly one JSON object\n"
        "- No prose, no markdown, no code fences\n"
        "- Keys must match exactly: method_name, return_param_name, loop_count, has_select, test_method_count, assert_call_count\n"
        "- has_select must be true or false\n"
        "- *_count fields must be integers\n\n"
        f"Task Prompt:\n{prompt_text}\n\n"
        f"Canonical ABAP Code:\n{canonical_src}\n\n"
        f"ABAP Unit Test:\n{unittest_src}\n\n"
        f"Expected JSON schema:\n{json.dumps(expected_schema, ensure_ascii=False)}"
    )


def _mock_response(expected: dict[str, Any], round_idx: int, model_name: str, item_id: str, repetition: int) -> str:
    payload = dict(expected)
    seed = sum(ord(c) for c in f"{model_name}|{item_id}|{repetition}")

    # Intentionally wrong on round 0 for around half of cases to exercise feedback rounds.
    if round_idx == 0 and seed % 2 == 0:
        payload["loop_count"] = int(payload["loop_count"]) + 1

    return json.dumps(payload, ensure_ascii=False)


def _parse_retry_delay(exc: Exception) -> float | None:
    """Extract the suggested retry delay (seconds) from a 429 response, if present."""
    try:
        body = getattr(exc, "body", None) or {}
        # OpenAI SDK: body is a dict with 'error' → 'details' list
        for detail in (body.get("error") or {}).get("details", []):
            delay_str = detail.get("retryDelay", "")
            if delay_str:
                # Format is e.g. "6s" or "6.642320681s"
                return float(delay_str.rstrip("s"))
    except Exception:
        pass
    return None


async def _ask_provider(client: Any, model_info: RunnableModel, chat_history: list[dict[str, str]]) -> str:
    """Call the provider API with automatic 429 retry/backoff.

    On RateLimitError the function respects the API's suggested retryDelay
    (parsed from the response body) and falls back to exponential backoff
    (5 s → 10 s → 20 s → 40 s → 80 s, capped at 120 s) for up to 8 attempts.
    """
    provider = model_info["provider"]
    max_attempts = 8

    for attempt in range(max_attempts):
        try:
            if provider == "SAP_AICORE":
                return await client.complete(chat_history)

            if provider == "ANTHROPIC":
                system_prompt = ""
                messages: list[dict[str, str]] = []
                for msg in chat_history:
                    role = msg["role"]
                    if role == "system":
                        system_prompt = msg["content"]
                    elif role in {"user", "assistant"}:
                        messages.append({"role": role, "content": msg["content"]})

                response = await client.messages.create(
                    model=model_info["name"],
                    system=system_prompt,
                    messages=messages,
                    temperature=model_info["temperature"],
                    max_tokens=model_info["max_tokens"],
                )

                parts: list[str] = []
                for block in response.content:
                    txt = getattr(block, "text", None)
                    if isinstance(txt, str):
                        parts.append(txt)
                return "\n".join(parts).strip()

            # OpenAI Responses API (/v1/responses) — for codex/reasoning models
            if provider == "OPENAI_RESPONSES":
                kwargs: dict = {
                    "model": model_info["name"],
                    "input": list(chat_history),
                    "max_output_tokens": model_info["max_tokens"],
                    "store": False,
                }
                effort = model_info.get("reasoning_effort")
                if effort:
                    kwargs["reasoning"] = {"effort": effort}
                response = await client.responses.create(**kwargs)
                return (response.output_text or "").strip()

            # OpenAI and OpenAI-compatible providers (chat completions)
            if "gpt-5" in model_info["name"]:
                response = await client.chat.completions.create(
                    model=model_info["name"],
                    messages=chat_history,
                    max_completion_tokens=model_info["max_tokens"],
                )
            else:
                response = await client.chat.completions.create(
                    model=model_info["name"],
                    messages=chat_history,
                    temperature=model_info["temperature"],
                    max_tokens=model_info["max_tokens"],
                )

            return (response.choices[0].message.content or "").strip()

        except Exception as exc:
            is_last = attempt == max_attempts - 1
            status = getattr(exc, "status_code", None)
            exc_type = type(exc).__name__
            is_rate_limit = status == 429 or "RateLimitError" in exc_type
            # 503 = server overloaded (high demand) — also retriable
            is_server_overload = status == 503 or "InternalServerError" in exc_type

            if is_last or not (is_rate_limit or is_server_overload):
                raise

            # Honour the API's suggested retry delay; fall back to exponential backoff
            suggested = _parse_retry_delay(exc) if is_rate_limit else None
            delay = suggested if suggested is not None else min(5 * (2 ** attempt), 120)
            # Add a small buffer on top of the suggested delay to avoid immediately re-hitting the limit
            delay = delay + 2
            label = "RATE LIMIT" if is_rate_limit else "SERVER OVERLOAD"
            print(
                f"[{label}] {model_info['name']} — attempt {attempt + 1}/{max_attempts}, "
                f"retrying in {delay:.1f}s"
            )
            await asyncio.sleep(delay)


async def _build_client(model_info: RunnableModel) -> Any:
    provider = model_info["provider"]

    if provider == "SAP_AICORE":
        return ABAP1OrchestrationClient.from_env(
            model_name=model_info["name"],
            temperature=model_info["temperature"],
            max_tokens=model_info["max_tokens"],
        )

    if provider == "ANTHROPIC":
        return create_anthropic_client(async_client=True)

    import openai

    api_key = get_provider_api_key(provider)
    # OPENAI_RESPONSES uses the same AsyncOpenAI client; the difference is in _ask_provider
    base_url = API_PROVIDERS[provider].get("base_url")
    if base_url:
        return openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
    return openai.AsyncOpenAI(api_key=api_key)


async def _run_single_item(
    *,
    client: Any,
    model_info: RunnableModel,
    item: dict[str, Any],
    repetition: int,
    max_rounds: int,
    mock: bool,
    include_conversation: bool,
    cache: dict[str, str],
) -> dict[str, Any]:
    prompt_text = _read_text(item["prompt_file"], cache)
    canonical_src = _read_text(item["canonical_file"], cache)
    unittest_src = _read_text(item["unittest_file"], cache)

    expected: dict[str, Any] = item["expected"]

    conversation: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _build_initial_user_prompt(item, prompt_text, canonical_src, unittest_src),
        },
    ]

    rounds: list[dict[str, Any]] = []
    success_round: int | None = None

    for round_idx in range(max_rounds):
        if mock:
            raw_response = _mock_response(expected, round_idx, model_info["name"], item["item_id"], repetition)
        else:
            raw_response = await _ask_provider(client, model_info, conversation)

        parsed = _extract_json_dict(raw_response)
        field_correct, overall_correct, normalized = _evaluate(parsed, expected)

        rounds.append(
            {
                "round": round_idx,
                "raw_response": raw_response,
                "parsed": parsed,
                "normalized": normalized,
                "field_correct": field_correct,
                "overall_correct": overall_correct,
            }
        )

        conversation.append({"role": "assistant", "content": raw_response})

        if overall_correct:
            success_round = round_idx
            break

        if round_idx < max_rounds - 1:
            conversation.append({"role": "user", "content": FEEDBACK_PROMPT})

    record: dict[str, Any] = {
        "model": model_info["name"],
        "provider": model_info["provider"],
        "item_id": item["item_id"],
        "prompt_id": item["prompt_id"],
        "repetition": repetition,
        "question_type": item.get("question_type", "composite_abap_understanding_v1"),
        "categories": item.get("categories", []),
        "expected": expected,
        "max_rounds": max_rounds,
        "success_round": success_round,
        "success": success_round is not None,
        "rounds": rounds,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    if include_conversation:
        record["conversation"] = conversation

    return record


def _status(
    *,
    model_name: str,
    items: list[dict[str, Any]],
    repetitions: int,
    output_path: Path,
) -> None:
    expected_total = len(items) * repetitions
    completed = 0
    success = 0

    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                completed += 1
                if row.get("success_round") is not None:
                    success += 1

    pending = max(expected_total - completed, 0)
    success_pct = (100.0 * success / completed) if completed else 0.0

    print(f"Model: {model_name}")
    print(f"Output file: {output_path}")
    print(f"Items: {len(items)}, repetitions: {repetitions}, expected runs: {expected_total}")
    print(f"Completed: {completed}")
    print(f"Pending: {pending}")
    print(f"Completed-run success: {success} ({success_pct:.2f}%)")


async def _run(args: argparse.Namespace) -> int:
    model_info = _get_model_info(args.model)
    if model_info is None:
        print(f"Error: model not found in src/llms.py: {args.model}")
        return 1

    limit_items = args.limit_items
    repetitions = args.repetitions
    max_rounds = args.max_rounds

    if args.smoke:
        limit_items = 3 if limit_items is None else min(limit_items, 3)
        repetitions = min(repetitions, 1)
        max_rounds = min(max_rounds, 2)

    items = _load_items(Path(args.items_file), limit_items=limit_items)
    if not items:
        print("Error: no items loaded.")
        return 1

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = REPO_ROOT / "data" / f"{_normalize_model_name(model_info['name'])}_understanding_predictions.jsonl"

    if args.mode == "status":
        _status(model_name=model_info["name"], items=items, repetitions=repetitions, output_path=output_path)
        return 0

    # ── Batch modes (OpenAI / Anthropic / Mistral where available) ──
    if args.mode.startswith("batch"):
        from understanding_batch import (
            batch_collect_and_advance,
            batch_run_all_rounds,
            batch_status,
            batch_submit_round,
        )

        if args.mode == "batch":
            try:
                return batch_run_all_rounds(
                    model_info=model_info,
                    items=items,
                    repetitions=repetitions,
                    max_rounds=max_rounds,
                    predictions_path=output_path,
                    poll_interval=30,
                )
            except Exception as exc:
                # Mistral free-trial often rejects batch jobs with HTTP 402 even
                # for very small smoke batches. Fallback to sequential mode.
                msg = str(exc)
                is_mistral_quota = (
                    model_info["provider"] == "MISTRAL"
                    and ("Status 402" in msg or "free trial" in msg.lower())
                )
                if not is_mistral_quota:
                    raise

                print(
                    "[WARN] Mistral batch mode rejected by provider quota "
                    f"({msg}). Falling back to sequential run mode."
                )
                # Continue below in normal run mode.
        if args.mode == "batch-submit":
            return batch_submit_round(
                model_info=model_info,
                items=items,
                repetitions=repetitions,
                max_rounds=max_rounds,
                predictions_path=output_path,
            )
        if args.mode == "batch-collect":
            return batch_collect_and_advance(
                model_info=model_info,
                max_rounds=max_rounds,
            )
        if args.mode == "batch-status":
            return batch_status(model_info)

    existing_keys = _load_existing_keys(output_path) if args.resume else set()

    expected_total = len(items) * repetitions
    todo = expected_total - len(existing_keys)

    concurrency = args.concurrency
    # Gemini 3.1 Pro Preview has only 25 RPM — cap concurrency automatically
    # unless the user explicitly raised it above the default.
    if model_info["provider"] == "GOOGLE" and concurrency == 10:
        concurrency = 3
    print(f"Model: {model_info['name']} ({model_info['provider']})")
    print(f"Items: {len(items)}, repetitions: {repetitions}, max_rounds: {max_rounds}")
    print(f"Output: {output_path}")
    print(f"Resume: {args.resume}, already completed keys: {len(existing_keys)}, todo: {todo}")
    print(f"Mock mode: {args.mock}, concurrency: {concurrency}")

    if todo <= 0:
        print("No pending runs.")
        return 0

    client: Any = None
    if not args.mock:
        client = await _build_client(model_info)

    # Shared mutable state — safe because asyncio is single-threaded cooperative.
    file_cache: dict[str, str] = {}
    done_count = 0
    errors = 0
    semaphore = asyncio.Semaphore(concurrency)

    async def _run_one(item: dict, repetition: int) -> None:
        nonlocal done_count, errors
        item_id = str(item["item_id"])
        key = f"{item_id}|{repetition}"
        if key in existing_keys:
            return
        async with semaphore:
            try:
                record = await _run_single_item(
                    client=client,
                    model_info=model_info,
                    item=item,
                    repetition=repetition,
                    max_rounds=max_rounds,
                    mock=args.mock,
                    include_conversation=args.include_conversation,
                    cache=file_cache,
                )
                _append_record(output_path, record)
                done_count += 1
                if done_count % 10 == 0 or done_count == todo:
                    print(f"Progress: {done_count}/{todo} new runs")
            except Exception as exc:
                errors += 1
                print(f"[ERROR] item={item_id} rep={repetition}: {exc!r}")
                if not args.continue_on_error:
                    raise

    tasks = [
        _run_one(item, repetition)
        for item in items
        for repetition in range(repetitions)
    ]
    try:
        await asyncio.gather(*tasks)
    except Exception:
        # continue_on_error=False: first error already printed; stop here
        print(f"Stopped after error. Completed: {done_count}/{todo}")
        return 1

    if errors:
        print(f"Completed new runs: {done_count}/{todo} ({errors} errors)")
    else:
        print(f"Completed new runs: {done_count}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ABAP understanding benchmark with iterative feedback rounds.")
    parser.add_argument("--model", "-m", required=True, help="Model name (e.g. gpt-5.2, claude-opus-4-5-20251101, sap-abap-1)")
    parser.add_argument(
        "--mode",
        default="run",
        choices=["run", "status", "batch", "batch-submit", "batch-collect", "batch-status"],
        help=(
            "run = sequential API calls (default); "
            "status = print progress; "
            "batch = full automated batch pipeline (OpenAI/Anthropic only); "
            "batch-submit = submit one batch round and exit; "
            "batch-collect = collect completed batch and evaluate; "
            "batch-status = show batch status"
        ),
    )
    parser.add_argument("--items-file", default=str(DEFAULT_ITEMS_FILE), help="Path to understanding items JSONL")
    parser.add_argument("--output", default=None, help="Optional output JSONL path")
    parser.add_argument("--repetitions", type=int, default=3, help="Repetitions per item (default: 3)")
    parser.add_argument("--max-rounds", type=int, default=6, help="Maximum feedback rounds (including round 0)")
    parser.add_argument("--limit-items", type=int, default=None, help="Optional limit for quick runs")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from existing output file")
    parser.add_argument("--no-resume", action="store_false", dest="resume", help="Do not resume; append fresh runs")
    parser.add_argument("--mock", action="store_true", help="Use deterministic mock responses (no API calls)")
    parser.add_argument("--smoke", action="store_true", help="Shortcut: very small run (<=3 items, 1 repetition, <=2 rounds)")
    parser.add_argument("--include-conversation", action="store_true", help="Store full chat history per run (larger output)")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue on per-item errors")
    parser.add_argument(
        "--concurrency", type=int, default=10,
        help="Max concurrent API requests in sequential run mode (default: 10). "
             "Increase for faster providers; decrease if you hit rate-limit errors.",
    )

    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
