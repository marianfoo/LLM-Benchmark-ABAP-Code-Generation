"""Google Gemini Batch API implementation.

Google's Gemini API exposes an OpenAI-compatible batch endpoint at:
  https://generativelanguage.googleapis.com/v1beta/openai/

The batch format (JSONL with custom_id / method / url / body) and the
retrieval flow (file upload → batch create → poll → download) are
identical to the OpenAI Batch API, so we reuse the pure utility
functions from generate_llm_answers_batch_openai and only provide
Google-specific client construction and a dedicated tracking file.

Batch benefits: 50% cost reduction, 24-hour completion window.
Obtain your API key at: https://aistudio.google.com/
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional

import openai
from openai.types.chat import ChatCompletionMessageParam

from llms import RunnableModel, get_provider_api_key, API_PROVIDERS
# Re-use pure (non-tracking) helpers from the OpenAI batch module
from generate_llm_answers_batch_openai import (
    _build_single_batch_request,
    _convert_to_json,
    _upload_batch_input_file,
    _create_batch,
    _status_batch,
    _retrieve_batch,
)
from generate_llm_answers import REPETITIONS, PROMPT_FILES, SYSTEM_PROMPT
from chat_state import (
    chat_is_success,
    chat_needs_test,
    chat_waiting_for_llm,
    assistant_count,
    any_needs_test,
    MAX_ASSISTANTS,
)

# Separate tracking file to avoid mixing Google batch IDs with OpenAI batch IDs
BATCH_TRACKING_FILE = "data/google_batch_tracking.json"


# =============================================================================
# Client factory
# =============================================================================

def _make_client() -> openai.OpenAI:
    """Create an OpenAI SDK client routed to Google's Gemini batch endpoint."""
    api_key = get_provider_api_key("GOOGLE")
    base_url = API_PROVIDERS["GOOGLE"]["base_url"]
    return openai.OpenAI(api_key=api_key, base_url=base_url)


# =============================================================================
# Batch tracking helpers (Google-specific tracking file)
# =============================================================================

def _load_batch_tracking() -> Dict:
    """Load the Google batch tracking file."""
    if os.path.exists(BATCH_TRACKING_FILE):
        with open(BATCH_TRACKING_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"batches": []}


def _save_batch_tracking(tracking: Dict):
    """Persist the Google batch tracking file."""
    os.makedirs(os.path.dirname(BATCH_TRACKING_FILE), exist_ok=True)
    with open(BATCH_TRACKING_FILE, "w", encoding="utf-8") as f:
        json.dump(tracking, f, indent=2)


def _log_batch(
    batch_id: str,
    model_name: str,
    save_file: str,
    save_file_batch: str,
    save_file_batch_response: str,
    round_num: int,
):
    """Record a new Google batch job in the tracking file."""
    tracking = _load_batch_tracking()
    tracking["batches"].append(
        {
            "batch_id": batch_id,
            "model_name": model_name,
            "save_file": save_file,
            "save_file_batch": save_file_batch,
            "save_file_batch_response": save_file_batch_response,
            "round_num": round_num,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "completed_at": None,
        }
    )
    _save_batch_tracking(tracking)
    print(f"[BATCH LOGGED] ID: {batch_id}, Model: {model_name}, Round: {round_num}")


def _update_batch_status(
    batch_id: str, status: str, completed_at: Optional[str] = None
):
    """Update the status of an existing batch entry."""
    tracking = _load_batch_tracking()
    for batch in tracking["batches"]:
        if batch["batch_id"] == batch_id:
            batch["status"] = status
            if completed_at:
                batch["completed_at"] = completed_at
            break
    _save_batch_tracking(tracking)


def get_pending_batches(model_name: Optional[str] = None) -> List[Dict]:
    """Return all pending Google batches, optionally filtered by model name."""
    tracking = _load_batch_tracking()
    pending = [b for b in tracking["batches"] if b["status"] == "pending"]
    if model_name:
        pending = [b for b in pending if b["model_name"] == model_name]
    return pending


# =============================================================================
# Public API: check / complete pending batches
# =============================================================================

def check_and_complete_pending_batches(
    client: openai.OpenAI, model_name: Optional[str] = None
) -> List[Dict]:
    """
    Check all pending Google batches and process any that have finished.
    Returns the list of batch records that were completed in this call.
    """
    pending = get_pending_batches(model_name)
    completed = []

    for batch_info in pending:
        batch_id = batch_info["batch_id"]
        print(
            f"[CHECKING] Batch {batch_id} "
            f"(Model: {batch_info['model_name']}, Round: {batch_info['round_num']})"
        )
        try:
            status = _status_batch(client, batch_id)
            print(
                f"  Status: {status.status} "
                f"({status.request_counts.completed}/{status.request_counts.total})"
            )

            if status.status == "completed":
                output_file_id = status.output_file_id
                if output_file_id:
                    content = _retrieve_batch(client, output_file_id)
                    with open(
                        batch_info["save_file_batch_response"], "w", encoding="utf-8"
                    ) as f:
                        f.write(content)
                    _convert_to_json(
                        batch_info["save_file"],
                        batch_info["save_file_batch"],
                        batch_info["save_file_batch_response"],
                    )
                    _update_batch_status(
                        batch_id, "completed", datetime.now().isoformat()
                    )
                    print(f"  [COMPLETED] Results saved to {batch_info['save_file']}")
                    completed.append(batch_info)

            elif status.status in ("failed", "cancelled", "expired"):
                _update_batch_status(batch_id, status.status)
                print(f"  [FAILED] Batch ended with status: {status.status}")
                if status.errors:
                    print(f"  Errors: {status.errors}")

        except Exception as e:
            print(f"  [ERROR] Could not check batch: {e}")

    return completed


# =============================================================================
# Public API: generate batch requests
# =============================================================================

def generate_first_response_batch(
    client: openai.OpenAI,
    model_info: RunnableModel,
    save_file_batch: str,
    save_file: str = None,
) -> Optional[str]:
    """
    Submit a batch for all Round-0 (first) responses.
    Non-destructive: skips prompt+rep combos that already have an assistant response.
    Returns the batch ID, or None if there is nothing to submit.
    """
    if save_file is None:
        save_file = save_file_batch.replace("_batch.jsonl", ".json")
    save_file_batch_response = save_file_batch[:-6] + "_response.jsonl"

    # Load existing data to enable resume behaviour
    existing: Dict = {}
    if os.path.exists(save_file):
        with open(save_file, "r", encoding="utf-8") as f:
            existing = json.load(f)

    conversations = []
    skipped = 0
    for prompt_file in PROMPT_FILES:
        with open(f"dataset/prompts/{prompt_file}", "r", encoding="utf-8") as f:
            prompt_content = f.read()
        for i in range(REPETITIONS):
            # Skip if this prompt+repetition already has an assistant response
            if prompt_file in existing:
                reps = existing[prompt_file]
                if i < len(reps) and reps[i] and assistant_count(reps[i]) > 0:
                    skipped += 1
                    continue

            conversation: List[ChatCompletionMessageParam] = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_content},
            ]
            custom_id = f"P{prompt_file}-R{i}"
            conversations.append(
                _build_single_batch_request(model_info, custom_id, conversation)
            )

    if skipped > 0:
        print(f"[RESUME] Skipped {skipped} already-generated repetitions")

    if not conversations:
        print(f"[SKIP] All first-round responses already exist in {save_file}")
        return None

    with open(save_file_batch, "w", encoding="utf-8") as f:
        for line in conversations:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    batch_input_file = _upload_batch_input_file(client, save_file_batch)
    batch = _create_batch(client, batch_input_file.id)
    _log_batch(
        batch.id,
        model_info["name"],
        save_file,
        save_file_batch,
        save_file_batch_response,
        round_num=1,
    )
    return batch.id


def generate_next_response_batch(
    client: openai.OpenAI,
    model_info: RunnableModel,
    save_file: str,
    save_file_batch: str,
    round_num: int = 0,
) -> Optional[str]:
    """
    Submit a batch for one correction round (WaitingForLLM conversations only).
    Refuses to run if any conversation is still in NeedsSAPTest state.
    Returns the batch ID, or None if there is nothing to submit.
    """
    save_file_batch_response = save_file_batch[:-6] + "_response.jsonl"

    with open(save_file, "r", encoding="utf-8") as f:
        current: Dict = json.load(f)

    # Guard: refuse to generate next round while SAP tests are still pending
    if any_needs_test(current):
        needs = sum(
            1 for chats in current.values() for c in chats if chat_needs_test(c)
        )
        print(
            f"[BLOCKED] {needs} conversation(s) still need SAP testing "
            f"(last message is assistant without feedback).\n"
            f"  Run SAP tests first:  python src/abap_test.py --model {model_info['name']} --mode resume\n"
            f"  Then retry:           python src/abap_test.py --model {model_info['name']} --mode retry --max-attempts 3"
        )
        return None

    conversations = []
    for prompt_file in current:
        for i, conversation in enumerate(current[prompt_file]):
            if not conversation:
                continue
            if chat_is_success(conversation):
                continue
            if not chat_waiting_for_llm(conversation):
                # Covers: NeedsSAPTest, InfraRetriable, empty, etc.
                continue
            if assistant_count(conversation) >= MAX_ASSISTANTS:
                # Already at Round 5 (6 assistant messages) – no more rounds
                continue

            custom_id = f"P{prompt_file}-R{i}"
            conversations.append(
                _build_single_batch_request(model_info, custom_id, conversation)
            )

    if not conversations:
        print(f"[SKIP] No conversations need processing for round {round_num}")
        return None

    with open(save_file_batch, "w", encoding="utf-8") as f:
        for line in conversations:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    batch_input_file = _upload_batch_input_file(client, save_file_batch)
    batch = _create_batch(client, batch_input_file.id)
    _log_batch(
        batch.id,
        model_info["name"],
        save_file,
        save_file_batch,
        save_file_batch_response,
        round_num,
    )
    return batch.id


# =============================================================================
# Public API: wait / poll for completion
# =============================================================================

def wait_for_batch_and_save(
    client: openai.OpenAI,
    batch_id: str,
    save_file: str,
    save_file_batch: str,
    save_file_batch_response: str,
) -> bool:
    """
    Poll the Google batch endpoint every 30 seconds until the batch completes,
    then download and merge results into save_file.
    Returns True on success, False on failure.
    """
    print(f"[WAITING] Batch {batch_id}...")
    while True:
        status = _status_batch(client, batch_id)
        print(
            f"  Status: {status.status} "
            f"({status.request_counts.completed}/{status.request_counts.total})"
        )
        if status.status == "completed":
            break
        elif status.status in ("failed", "cancelled", "expired"):
            _update_batch_status(batch_id, status.status)
            print(f"[FAILED] Batch ended with status: {status.status}")
            if status.errors:
                print(f"  Errors: {status.errors}")
            return False
        time.sleep(30)

    output_file_id = status.output_file_id
    if output_file_id:
        content = _retrieve_batch(client, output_file_id)
        with open(save_file_batch_response, "w", encoding="utf-8") as f:
            f.write(content)
        _convert_to_json(save_file, save_file_batch, save_file_batch_response)
        _update_batch_status(batch_id, "completed", datetime.now().isoformat())
        print(f"[COMPLETED] Results saved to {save_file}")
        return True
    return False


def list_all_batches():
    """Print a summary of all tracked Google batches."""
    tracking = _load_batch_tracking()
    if not tracking["batches"]:
        print("No Google batches tracked yet.")
        return

    print("\n=== Google Batch Tracking ===")
    for batch in tracking["batches"]:
        print(
            f"  {batch['batch_id'][:20]}... | {batch['model_name']} "
            f"| Round {batch['round_num']} | {batch['status']} | {batch['created_at']}"
        )
