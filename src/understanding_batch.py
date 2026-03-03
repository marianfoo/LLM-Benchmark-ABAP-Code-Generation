#!/usr/bin/env python3
"""Batch mode for ABAP understanding benchmark (OpenAI + Anthropic + Google Gemini).

Submits all (item, repetition) pairs per round as a single batch job,
waits for completion, evaluates locally, then submits next-round batches
for items that failed.  ABAP-1 / SAP_AICORE is not supported (no batch API).

Usage (via understanding_eval.py):

    # Full automated pipeline – submit round 0, wait, evaluate, round 1, …
    .venv/bin/python src/understanding_eval.py --model gpt-5.2 --mode batch --repetitions 10

    # Submit one round and exit (async workflow)
    .venv/bin/python src/understanding_eval.py --model gpt-5.2 --mode batch-submit --repetitions 10

    # Collect completed batch, evaluate, prepare next round
    .venv/bin/python src/understanding_eval.py --model gpt-5.2 --mode batch-collect

    # Show batch status
    .venv/bin/python src/understanding_eval.py --model gpt-5.2 --mode batch-status
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llms import API_PROVIDERS, RunnableModel, get_provider_api_key
from understanding_eval import (
    FEEDBACK_PROMPT,
    REPO_ROOT,
    SYSTEM_PROMPT,
    _append_record,
    _build_initial_user_prompt,
    _evaluate,
    _extract_json_dict,
    _load_existing_keys,
    _normalize_model_name,
    _read_text,
)

TRACKING_DIR = REPO_ROOT / "data"
BATCH_PROVIDERS = {"OPENAI", "ANTHROPIC", "MISTRAL", "GOOGLE"}


# ---------------------------------------------------------------------------
# Tracking persistence (batch job history)
# ---------------------------------------------------------------------------

def _tracking_path(provider: str) -> Path:
    if provider == "MISTRAL":
        tag = "mistral"
    elif provider == "ANTHROPIC":
        tag = "anthropic"
    elif provider == "GOOGLE":
        tag = "google"
    else:
        tag = "openai"
    return TRACKING_DIR / f"understanding_{tag}_batch_tracking.json"


def _load_tracking(provider: str) -> Dict:
    path = _tracking_path(provider)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"batches": []}


def _save_tracking(provider: str, tracking: Dict) -> None:
    path = _tracking_path(provider)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tracking, indent=2, ensure_ascii=False), encoding="utf-8")


def _log_batch(
    provider: str,
    batch_id: str,
    model_name: str,
    round_num: int,
    batch_input_file: str,
    item_count: int,
) -> None:
    tracking = _load_tracking(provider)
    tracking["batches"].append({
        "batch_id": batch_id,
        "model_name": model_name,
        "provider": provider,
        "round_num": round_num,
        "batch_input_file": batch_input_file,
        "item_count": item_count,
        "status": "pending",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
    })
    _save_tracking(provider, tracking)
    print(f"[BATCH LOGGED] ID: {batch_id}, Model: {model_name}, Round: {round_num}, Items: {item_count}")


def _update_tracking_status(provider: str, batch_id: str, status: str) -> None:
    tracking = _load_tracking(provider)
    for batch in tracking["batches"]:
        if batch["batch_id"] == batch_id:
            batch["status"] = status
            if status != "pending":
                batch["completed_at"] = datetime.now(timezone.utc).isoformat()
            break
    _save_tracking(provider, tracking)


def get_pending_batches(provider: str, model_name: Optional[str] = None) -> List[Dict]:
    tracking = _load_tracking(provider)
    pending = [b for b in tracking["batches"] if b["status"] == "pending"]
    if model_name:
        pending = [b for b in pending if b["model_name"] == model_name]
    return pending


# ---------------------------------------------------------------------------
# Batch state file (in-progress items across rounds)
# ---------------------------------------------------------------------------

def _state_path(model_name: str) -> Path:
    safe = _normalize_model_name(model_name)
    return TRACKING_DIR / f"{safe}_understanding_batch_state.json"


def _load_state(model_name: str) -> Dict[str, Any]:
    path = _state_path(model_name)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _save_state(model_name: str, state: Dict[str, Any]) -> None:
    path = _state_path(model_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")


def _clear_state(model_name: str) -> None:
    path = _state_path(model_name)
    if path.exists():
        path.unlink()


# ---------------------------------------------------------------------------
# Synchronous client constructors (batch APIs are synchronous)
# ---------------------------------------------------------------------------

def _build_sync_client(model_info: RunnableModel):
    provider = model_info["provider"]

    if provider == "ANTHROPIC":
        import anthropic

        return anthropic.Anthropic(api_key=get_provider_api_key(provider))

    if provider == "MISTRAL":
        from mistralai import Mistral

        return Mistral(api_key=get_provider_api_key(provider))

    import openai

    api_key = get_provider_api_key(provider)
    base_url = API_PROVIDERS[provider].get("base_url")
    if base_url:
        return openai.OpenAI(api_key=api_key, base_url=base_url)
    return openai.OpenAI(api_key=api_key)


# ---------------------------------------------------------------------------
# OpenAI batch helpers
# ---------------------------------------------------------------------------

def _openai_build_request(
    model_info: RunnableModel, custom_id: str, conversation: List[Dict]
) -> Dict:
    body: Dict[str, Any] = {
        "model": model_info["name"],
        "messages": conversation,
        "max_completion_tokens": model_info["max_tokens"],
    }
    if "gpt-5" not in model_info["name"]:
        body["temperature"] = model_info["temperature"]
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }


def _openai_submit(client, requests: List[Dict], batch_input_path: str) -> str:
    Path(batch_input_path).parent.mkdir(parents=True, exist_ok=True)
    with open(batch_input_path, "w", encoding="utf-8") as f:
        for req in requests:
            f.write(json.dumps(req, ensure_ascii=False) + "\n")

    batch_input_file = client.files.create(
        file=open(batch_input_path, "rb"), purpose="batch"
    )
    batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "ABAP Understanding Benchmark"},
    )
    return batch.id


def _openai_poll(client, batch_id: str) -> tuple[str, str]:
    """Returns (status_str, progress_str)."""
    status = client.batches.retrieve(batch_id)
    rc = status.request_counts
    progress = f"({rc.completed}/{rc.total})" if rc else ""
    return status.status, progress


def _openai_collect(client, batch_id: str) -> Dict[str, str]:
    """Returns {custom_id: response_text}."""
    status = client.batches.retrieve(batch_id)
    if not status.output_file_id:
        return {}
    content = client.files.content(status.output_file_id).text
    results: Dict[str, str] = {}
    for line in content.strip().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        cid = entry["custom_id"]
        text = (
            entry["response"]["body"]["choices"][0]["message"]["content"] or ""
        ).strip()
        results[cid] = text
    return results


# ---------------------------------------------------------------------------
# Anthropic batch helpers
# ---------------------------------------------------------------------------

def _anthropic_build_request(
    model_info: RunnableModel, custom_id: str, conversation: List[Dict]
):
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    system_prompt = ""
    messages = []
    for msg in conversation:
        if msg["role"] == "system":
            system_prompt = msg["content"]
        else:
            messages.append({"role": msg["role"], "content": msg["content"]})

    return Request(
        custom_id=custom_id,
        params=MessageCreateParamsNonStreaming(
            model=model_info["name"],
            max_tokens=model_info["max_tokens"],
            temperature=model_info["temperature"],
            system=system_prompt,
            messages=messages,
        ),
    )


def _anthropic_submit(client, requests, batch_input_path: str) -> str:
    Path(batch_input_path).parent.mkdir(parents=True, exist_ok=True)
    with open(batch_input_path, "w", encoding="utf-8") as f:
        for req in requests:
            f.write(json.dumps(req, ensure_ascii=False) + "\n")

    batch = client.messages.batches.create(requests=requests)
    return batch.id


def _anthropic_poll(client, batch_id: str) -> tuple[str, str]:
    status = client.messages.batches.retrieve(batch_id)
    counts = status.request_counts
    progress = ""
    if counts:
        done = counts.succeeded + counts.errored
        total = done + counts.processing + counts.canceled
        progress = f"({done}/{total})"
    return status.processing_status, progress


def _anthropic_collect(client, batch_id: str) -> Dict[str, str]:
    from anthropic.types import TextBlock

    results: Dict[str, str] = {}
    failed = 0
    for entry in client.messages.batches.results(batch_id):
        if entry.result.type == "succeeded":
            text = " ".join(
                c.text
                for c in entry.result.message.content
                if isinstance(c, TextBlock)
            ).strip()
            results[entry.custom_id] = text
        else:
            failed += 1
            print(f"  [WARN] Request {entry.custom_id} result type: {entry.result.type}")
    if failed:
        print(f"  {failed} Anthropic request(s) failed")
    return results


# ---------------------------------------------------------------------------
# Mistral batch helpers
# ---------------------------------------------------------------------------

def _mistral_build_request(
    model_info: RunnableModel, custom_id: str, conversation: List[Dict]
) -> Dict:
    """Build a Mistral batch request line (no method/url; model set at job level)."""
    body: Dict[str, Any] = {
        "messages": conversation,
        "max_tokens": model_info["max_tokens"],
        "temperature": model_info["temperature"],
    }
    return {"custom_id": custom_id, "body": body}


def _mistral_submit(client, model_info: RunnableModel, requests: List[Dict], batch_input_path: str) -> str:
    """Upload JSONL and create a Mistral batch job. Returns job ID."""
    Path(batch_input_path).parent.mkdir(parents=True, exist_ok=True)
    with open(batch_input_path, "w", encoding="utf-8") as f:
        for req in requests:
            f.write(json.dumps(req, ensure_ascii=False) + "\n")

    batch_data = client.files.upload(
        file={"file_name": Path(batch_input_path).name, "content": open(batch_input_path, "rb")},
        purpose="batch",
    )
    job = client.batch.jobs.create(
        input_files=[batch_data.id],
        model=model_info["name"],
        endpoint="/v1/chat/completions",
        metadata={"description": "ABAP Understanding Benchmark"},
    )
    return job.id


def _mistral_poll(client, job_id: str) -> tuple[str, str]:
    job = client.batch.jobs.get(job_id=job_id)
    total = getattr(job, "total_requests", 0) or 0
    succeeded = getattr(job, "succeeded_requests", 0) or 0
    failed = getattr(job, "failed_requests", 0) or 0
    done = succeeded + failed
    progress = f"({done}/{total})" if total else ""
    return job.status, progress


def _mistral_collect(client, job_id: str) -> Dict[str, str]:
    """Download Mistral batch output and parse into {custom_id: response_text}."""
    job = client.batch.jobs.get(job_id=job_id)
    output_file_id = getattr(job, "output_file", None)
    if not output_file_id:
        return {}

    stream = client.files.download(file_id=output_file_id)
    raw_bytes = stream.read() if hasattr(stream, "read") else b"".join(stream.stream)
    content = raw_bytes.decode("utf-8")

    results: Dict[str, str] = {}
    failed = 0
    for line in content.strip().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        cid = entry["custom_id"]
        resp = entry.get("response", {})
        if resp.get("status_code") == 200:
            text = (
                resp["body"]["choices"][0]["message"]["content"] or ""
            ).strip()
            results[cid] = text
        else:
            failed += 1
            err = entry.get("error") or resp.get("body", {}).get("message", "unknown")
            print(f"  [WARN] Request {cid} failed: {err}")
    if failed:
        print(f"  {failed} Mistral request(s) failed")
    return results


# ---------------------------------------------------------------------------
# Unified provider dispatch
# ---------------------------------------------------------------------------

def _build_request(
    model_info: RunnableModel, custom_id: str, conversation: List[Dict]
):
    if model_info["provider"] == "ANTHROPIC":
        return _anthropic_build_request(model_info, custom_id, conversation)
    if model_info["provider"] == "MISTRAL":
        return _mistral_build_request(model_info, custom_id, conversation)
    return _openai_build_request(model_info, custom_id, conversation)


def _submit_batch(
    client, model_info: RunnableModel, requests, batch_input_path: str
) -> str:
    if model_info["provider"] == "ANTHROPIC":
        return _anthropic_submit(client, requests, batch_input_path)
    if model_info["provider"] == "MISTRAL":
        return _mistral_submit(client, model_info, requests, batch_input_path)
    return _openai_submit(client, requests, batch_input_path)


def _poll_batch(
    client, model_info: RunnableModel, batch_id: str
) -> tuple[str, str]:
    if model_info["provider"] == "ANTHROPIC":
        return _anthropic_poll(client, batch_id)
    if model_info["provider"] == "MISTRAL":
        return _mistral_poll(client, batch_id)
    return _openai_poll(client, batch_id)


def _is_done(provider: str, status_str: str) -> bool:
    if provider == "ANTHROPIC":
        return status_str == "ended"
    if provider == "MISTRAL":
        return status_str == "SUCCESS"
    return status_str == "completed"


def _is_failed(provider: str, status_str: str) -> bool:
    if provider == "ANTHROPIC":
        return status_str in ("canceled", "expired")
    if provider == "MISTRAL":
        return status_str in ("FAILED", "TIMEOUT_EXCEEDED", "CANCELLATION_REQUESTED", "CANCELLED")
    return status_str in ("failed", "cancelled", "expired")


def _collect_responses(
    client, model_info: RunnableModel, batch_id: str
) -> Dict[str, str]:
    if model_info["provider"] == "ANTHROPIC":
        return _anthropic_collect(client, batch_id)
    if model_info["provider"] == "MISTRAL":
        return _mistral_collect(client, batch_id)
    return _openai_collect(client, batch_id)


# ---------------------------------------------------------------------------
# Item initialization
# ---------------------------------------------------------------------------

def _init_pending_items(
    items: List[Dict[str, Any]],
    repetitions: int,
    existing_keys: set[str],
    file_cache: Dict[str, str],
) -> Dict[str, Dict[str, Any]]:
    """Build dict of (item_id, repetition) pairs that still need processing.

    Key: "item_id|repetition"
    Value: dict with item_id, prompt_id, repetition, conversation, rounds,
           expected, categories, question_type
    """
    pending: Dict[str, Dict[str, Any]] = {}

    for item in items:
        item_id = str(item["item_id"])
        prompt_text = _read_text(item["prompt_file"], file_cache)
        canonical_src = _read_text(item["canonical_file"], file_cache)
        unittest_src = _read_text(item["unittest_file"], file_cache)

        for rep in range(repetitions):
            key = f"{item_id}|{rep}"
            if key in existing_keys:
                continue

            conversation = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _build_initial_user_prompt(
                        item, prompt_text, canonical_src, unittest_src
                    ),
                },
            ]

            pending[key] = {
                "item_id": item_id,
                "prompt_id": item["prompt_id"],
                "repetition": rep,
                "conversation": conversation,
                "rounds": [],
                "expected": item["expected"],
                "categories": item.get("categories", []),
                "question_type": item.get(
                    "question_type", "composite_abap_understanding_v1"
                ),
            }

    return pending


def _build_prediction_record(
    model_info: RunnableModel,
    item_state: Dict[str, Any],
    max_rounds: int,
) -> Dict[str, Any]:
    """Build the final prediction record for writing to the predictions JSONL."""
    rounds = item_state["rounds"]
    success_round = None
    for r in rounds:
        if r["overall_correct"]:
            success_round = r["round"]
            break

    return {
        "model": model_info["name"],
        "provider": model_info["provider"],
        "item_id": item_state["item_id"],
        "prompt_id": item_state["prompt_id"],
        "repetition": item_state["repetition"],
        "question_type": item_state["question_type"],
        "categories": item_state["categories"],
        "expected": item_state["expected"],
        "max_rounds": max_rounds,
        "success_round": success_round,
        "success": success_round is not None,
        "rounds": rounds,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


# ---------------------------------------------------------------------------
# Evaluate one round of batch responses
# ---------------------------------------------------------------------------

def _evaluate_round(
    *,
    pending: Dict[str, Dict[str, Any]],
    responses: Dict[str, str],
    round_idx: int,
    max_rounds: int,
    model_info: RunnableModel,
    predictions_path: Path,
) -> Dict[str, Dict[str, Any]]:
    """Evaluate responses, write successes/final-failures to predictions.

    Returns the still-pending items (with feedback appended to conversations).
    """
    still_pending: Dict[str, Dict[str, Any]] = {}
    round_successes = 0
    completed = 0

    for key, item_state in pending.items():
        custom_id = f"U{item_state['item_id']}-R{item_state['repetition']}"
        raw_response = responses.get(custom_id, "")

        parsed = _extract_json_dict(raw_response)
        field_correct, overall_correct, normalized = _evaluate(
            parsed, item_state["expected"]
        )

        round_data = {
            "round": round_idx,
            "raw_response": raw_response,
            "parsed": parsed,
            "normalized": normalized,
            "field_correct": field_correct,
            "overall_correct": overall_correct,
        }
        item_state["rounds"].append(round_data)
        item_state["conversation"].append(
            {"role": "assistant", "content": raw_response}
        )

        if overall_correct:
            record = _build_prediction_record(model_info, item_state, max_rounds)
            _append_record(predictions_path, record)
            round_successes += 1
            completed += 1
        elif round_idx < max_rounds - 1:
            item_state["conversation"].append(
                {"role": "user", "content": FEEDBACK_PROMPT}
            )
            still_pending[key] = item_state
        else:
            record = _build_prediction_record(model_info, item_state, max_rounds)
            _append_record(predictions_path, record)
            completed += 1

    print(
        f"  Round {round_idx}: {round_successes} succeeded, "
        f"{len(still_pending)} need retry, {completed} written to predictions"
    )
    return still_pending


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def batch_run_all_rounds(
    *,
    model_info: RunnableModel,
    items: List[Dict[str, Any]],
    repetitions: int,
    max_rounds: int,
    predictions_path: Path,
    poll_interval: int = 30,
) -> int:
    """Full automated batch pipeline: round 0 → wait → evaluate → round 1 → …

    Returns 0 on success, 1 on error.
    """
    provider = model_info["provider"]
    if provider not in BATCH_PROVIDERS:
        print(f"Error: batch mode not supported for provider {provider}")
        return 1

    model_name = model_info["name"]
    safe_name = _normalize_model_name(model_name)

    # Check for an already-pending batch from a previous interrupted run
    existing_pending = get_pending_batches(provider, model_name)
    state = _load_state(model_name) if existing_pending else {}

    if state and state.get("items") and state.get("active_batch_id"):
        # Resume from interrupted run
        pending = state["items"]
        round_idx_start = state.get("active_round", 0)
        batch_id = state["active_batch_id"]
        print(f"[RESUME] Found active batch {batch_id} at round {round_idx_start}")
        print(f"  {len(pending)} items in progress. Waiting for batch completion…")

        client = _build_sync_client(model_info)

        # Wait for the interrupted batch
        while True:
            status_str, progress = _poll_batch(client, model_info, batch_id)
            print(f"  Status: {status_str} {progress}")
            if _is_done(provider, status_str):
                _update_tracking_status(provider, batch_id, "completed")
                break
            if _is_failed(provider, status_str):
                _update_tracking_status(provider, batch_id, status_str)
                print(f"[FAILED] Batch {batch_id} ended with: {status_str}")
                return 1
            time.sleep(poll_interval)

        responses = _collect_responses(client, model_info, batch_id)
        print(f"  Collected {len(responses)} / {len(pending)} responses")

        pending = _evaluate_round(
            pending=pending,
            responses=responses,
            round_idx=round_idx_start,
            max_rounds=max_rounds,
            model_info=model_info,
            predictions_path=predictions_path,
        )
        round_idx_start += 1
    else:
        # Fresh start
        existing_keys = _load_existing_keys(predictions_path)
        file_cache: Dict[str, str] = {}
        pending = _init_pending_items(items, repetitions, existing_keys, file_cache)
        round_idx_start = 0

    total = len(pending)
    if total == 0:
        print("No pending runs. All (item, repetition) pairs already completed.")
        _clear_state(model_name)
        return 0

    print(f"Model: {model_name} ({provider}) [BATCH MODE]")
    print(f"Items: {len(items)}, repetitions: {repetitions}, max_rounds: {max_rounds}")
    print(f"Pending: {total}")
    print(f"Output: {predictions_path}")

    if round_idx_start == 0:
        client = _build_sync_client(model_info)

    for round_idx in range(round_idx_start, max_rounds):
        if not pending:
            break

        print(f"\n{'=' * 60}")
        print(f"Round {round_idx}: {len(pending)} items")
        print(f"{'=' * 60}")

        # Build batch requests
        requests = []
        for key, item_state in pending.items():
            custom_id = f"U{item_state['item_id']}-R{item_state['repetition']}"
            req = _build_request(model_info, custom_id, item_state["conversation"])
            requests.append(req)

        # Submit
        batch_input_path = str(
            TRACKING_DIR / f"{safe_name}_understanding_batch_r{round_idx}.jsonl"
        )
        batch_id = _submit_batch(client, model_info, requests, batch_input_path)
        _log_batch(
            provider, batch_id, model_name, round_idx, batch_input_path, len(requests)
        )

        # Save state for crash recovery
        state = {
            "model": model_name,
            "provider": provider,
            "active_batch_id": batch_id,
            "active_round": round_idx,
            "max_rounds": max_rounds,
            "predictions_path": str(predictions_path),
            "items": pending,
        }
        _save_state(model_name, state)

        # Wait for completion
        print(f"[WAITING] Batch {batch_id}…")
        while True:
            status_str, progress = _poll_batch(client, model_info, batch_id)
            print(f"  Status: {status_str} {progress}")
            if _is_done(provider, status_str):
                _update_tracking_status(provider, batch_id, "completed")
                break
            if _is_failed(provider, status_str):
                _update_tracking_status(provider, batch_id, status_str)
                print(f"[FAILED] Batch {batch_id} ended with: {status_str}")
                return 1
            time.sleep(poll_interval)

        # Collect and evaluate
        responses = _collect_responses(client, model_info, batch_id)
        print(f"  Collected {len(responses)} / {len(requests)} responses")

        pending = _evaluate_round(
            pending=pending,
            responses=responses,
            round_idx=round_idx,
            max_rounds=max_rounds,
            model_info=model_info,
            predictions_path=predictions_path,
        )

    # Write any remaining items that exhausted all rounds
    for key, item_state in pending.items():
        record = _build_prediction_record(model_info, item_state, max_rounds)
        _append_record(predictions_path, record)

    _clear_state(model_name)

    print(f"\n{'=' * 60}")
    print(f"Batch pipeline complete.")
    print(f"Output: {predictions_path}")
    return 0


def batch_submit_round(
    *,
    model_info: RunnableModel,
    items: List[Dict[str, Any]],
    repetitions: int,
    max_rounds: int,
    predictions_path: Path,
) -> int:
    """Submit one batch round and exit (for async workflow).

    On first call: submits round 0 for all pending items.
    On subsequent calls: loads state and submits next round for failed items.
    Returns 0 on success, 1 on error.
    """
    provider = model_info["provider"]
    if provider not in BATCH_PROVIDERS:
        print(f"Error: batch mode not supported for provider {provider}")
        return 1

    model_name = model_info["name"]
    safe_name = _normalize_model_name(model_name)

    # Block if there is already a pending batch
    existing_pending = get_pending_batches(provider, model_name)
    if existing_pending:
        print(f"[BLOCKED] Model {model_name} has {len(existing_pending)} pending batch(es).")
        print("  Run --mode batch-collect first to collect results.")
        for pb in existing_pending:
            print(
                f"    Batch {pb['batch_id'][:30]}…  Round {pb['round_num']}  "
                f"submitted {pb['created_at']}"
            )
        return 1

    # Load or initialize state
    state = _load_state(model_name)

    if state and state.get("items"):
        pending = state["items"]
        round_idx = state.get("active_round", -1) + 1
        print(
            f"[RESUME] Loaded state: round {round_idx}, "
            f"{len(pending)} pending items"
        )
    else:
        existing_keys = _load_existing_keys(predictions_path)
        file_cache: Dict[str, str] = {}
        pending = _init_pending_items(items, repetitions, existing_keys, file_cache)
        round_idx = 0

    if not pending:
        print("No pending runs.")
        _clear_state(model_name)
        return 0

    if round_idx >= max_rounds:
        print(f"All {max_rounds} rounds exhausted. Writing remaining failed items.")
        for key, item_state in pending.items():
            record = _build_prediction_record(model_info, item_state, max_rounds)
            _append_record(predictions_path, record)
        _clear_state(model_name)
        return 0

    print(f"Model: {model_name} ({provider}) [BATCH SUBMIT]")
    print(f"Round: {round_idx}, Items: {len(pending)}")

    client = _build_sync_client(model_info)

    requests = []
    for key, item_state in pending.items():
        custom_id = f"U{item_state['item_id']}-R{item_state['repetition']}"
        req = _build_request(model_info, custom_id, item_state["conversation"])
        requests.append(req)

    batch_input_path = str(
        TRACKING_DIR / f"{safe_name}_understanding_batch_r{round_idx}.jsonl"
    )
    batch_id = _submit_batch(client, model_info, requests, batch_input_path)
    _log_batch(
        provider, batch_id, model_name, round_idx, batch_input_path, len(requests)
    )

    state = {
        "model": model_name,
        "provider": provider,
        "active_batch_id": batch_id,
        "active_round": round_idx,
        "max_rounds": max_rounds,
        "predictions_path": str(predictions_path),
        "items": pending,
    }
    _save_state(model_name, state)

    print(f"[SUBMITTED] Batch {batch_id}")
    print("  Run --mode batch-collect when ready to collect results.")
    return 0


def batch_collect_and_advance(
    *,
    model_info: RunnableModel,
    max_rounds: int,
) -> int:
    """Collect a completed batch, evaluate, write results.

    Updates state so a subsequent batch-submit picks up the next round.
    Returns 0 on success, 1 on error.
    """
    provider = model_info["provider"]
    if provider not in BATCH_PROVIDERS:
        print(f"Error: batch mode not supported for provider {provider}")
        return 1

    model_name = model_info["name"]

    state = _load_state(model_name)
    if not state or not state.get("items"):
        print("No batch state found. Nothing to collect.")
        return 0

    pending = state["items"]
    batch_id = state.get("active_batch_id")
    round_idx = state.get("active_round", 0)
    predictions_path = Path(state.get("predictions_path", ""))

    if not batch_id:
        print("No active batch ID in state. Run --mode batch-submit first.")
        return 1

    client = _build_sync_client(model_info)

    status_str, progress = _poll_batch(client, model_info, batch_id)
    print(f"Batch {batch_id}: {status_str} {progress}")

    if _is_failed(provider, status_str):
        _update_tracking_status(provider, batch_id, status_str)
        print(f"[FAILED] Batch ended with: {status_str}")
        return 1

    if not _is_done(provider, status_str):
        print("Batch not yet complete. Try again later.")
        return 0

    _update_tracking_status(provider, batch_id, "completed")

    responses = _collect_responses(client, model_info, batch_id)
    print(f"Collected {len(responses)} / {len(pending)} responses")

    still_pending = _evaluate_round(
        pending=pending,
        responses=responses,
        round_idx=round_idx,
        max_rounds=max_rounds,
        model_info=model_info,
        predictions_path=predictions_path,
    )

    if still_pending:
        next_round = round_idx + 1
        if next_round >= max_rounds:
            print(f"All {max_rounds} rounds exhausted. Writing remaining failed items.")
            for key, item_state in still_pending.items():
                record = _build_prediction_record(model_info, item_state, max_rounds)
                _append_record(predictions_path, record)
            _clear_state(model_name)
        else:
            state["items"] = still_pending
            state["active_batch_id"] = None
            state["active_round"] = round_idx
            _save_state(model_name, state)
            print(
                f"  {len(still_pending)} items ready for round {next_round}. "
                f"Run --mode batch-submit to continue."
            )
    else:
        _clear_state(model_name)
        print("All items complete!")

    return 0


def batch_status(model_info: RunnableModel) -> int:
    """Print status of pending batches and batch state for this model."""
    provider = model_info["provider"]
    model_name = model_info["name"]

    if provider not in BATCH_PROVIDERS:
        print(f"Batch mode not available for provider {provider}")
        return 0

    # Tracking history
    tracking = _load_tracking(provider)
    model_batches = [b for b in tracking["batches"] if b["model_name"] == model_name]

    tag = {"ANTHROPIC": "Anthropic", "MISTRAL": "Mistral", "GOOGLE": "Google"}.get(provider, "OpenAI")
    if model_batches:
        print(f"\n=== {tag} Understanding Batch History ({model_name}) ===")
        for b in model_batches:
            print(
                f"  {b['batch_id'][:30]}… | Round {b['round_num']} | "
                f"{b['status']} | Items: {b['item_count']} | {b['created_at']}"
            )
    else:
        print(f"No batch history for {model_name}")

    # Active state
    state = _load_state(model_name)
    if state and state.get("items"):
        pending = state["items"]
        batch_id = state.get("active_batch_id")
        round_idx = state.get("active_round", "?")
        print(f"\nActive batch state:")
        print(f"  Pending items: {len(pending)}")
        print(f"  Current round: {round_idx}")
        print(f"  Active batch:  {batch_id or 'none (ready for batch-submit)'}")

        if batch_id:
            try:
                client = _build_sync_client(model_info)
                status_str, progress = _poll_batch(client, model_info, batch_id)
                print(f"  Live status:   {status_str} {progress}")
            except Exception as e:
                print(f"  Could not check live status: {e}")
    else:
        print(f"\nNo active batch state for {model_name}")

    return 0
