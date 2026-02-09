import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional
from openai.types.chat import ChatCompletionMessageParam
import openai

from llms import RunnableModel
from generate_llm_answers import (
    SYSTEM_PROMPT,
    REPETITIONS,
    PROMPT_FILES,
    remove_code_blocks,
)
from chat_state import (
    chat_is_success,
    chat_needs_test,
    chat_waiting_for_llm,
    assistant_count,
    any_needs_test,
    MAX_ASSISTANTS,
)

# Batch tracking file to persist batch IDs across script runs
BATCH_TRACKING_FILE = "data/openai_batch_tracking.json"


def _load_batch_tracking() -> Dict:
    """Load the batch tracking file."""
    if os.path.exists(BATCH_TRACKING_FILE):
        with open(BATCH_TRACKING_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"batches": []}


def _save_batch_tracking(tracking: Dict):
    """Save the batch tracking file."""
    os.makedirs(os.path.dirname(BATCH_TRACKING_FILE), exist_ok=True)
    with open(BATCH_TRACKING_FILE, "w", encoding="utf-8") as f:
        json.dump(tracking, f, indent=2)


def _log_batch(batch_id: str, model_name: str, save_file: str, save_file_batch: str, 
               save_file_batch_response: str, round_num: int):
    """Log a new batch to the tracking file."""
    tracking = _load_batch_tracking()
    tracking["batches"].append({
        "batch_id": batch_id,
        "model_name": model_name,
        "save_file": save_file,
        "save_file_batch": save_file_batch,
        "save_file_batch_response": save_file_batch_response,
        "round_num": round_num,
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "completed_at": None,
    })
    _save_batch_tracking(tracking)
    print(f"[BATCH LOGGED] ID: {batch_id}, Model: {model_name}, Round: {round_num}")


def _update_batch_status(batch_id: str, status: str, completed_at: Optional[str] = None):
    """Update the status of a batch in the tracking file."""
    tracking = _load_batch_tracking()
    for batch in tracking["batches"]:
        if batch["batch_id"] == batch_id:
            batch["status"] = status
            if completed_at:
                batch["completed_at"] = completed_at
            break
    _save_batch_tracking(tracking)


def get_pending_batches(model_name: Optional[str] = None) -> List[Dict]:
    """Get all pending batches, optionally filtered by model name."""
    tracking = _load_batch_tracking()
    pending = [b for b in tracking["batches"] if b["status"] == "pending"]
    if model_name:
        pending = [b for b in pending if b["model_name"] == model_name]
    return pending


def check_and_complete_pending_batches(client: openai.OpenAI, model_name: Optional[str] = None) -> List[Dict]:
    """
    Check all pending batches and complete any that are finished.
    Returns list of batches that were completed.
    """
    pending = get_pending_batches(model_name)
    completed = []
    
    for batch_info in pending:
        batch_id = batch_info["batch_id"]
        print(f"[CHECKING] Batch {batch_id} (Model: {batch_info['model_name']}, Round: {batch_info['round_num']})")
        
        try:
            status = _status_batch(client, batch_id)
            print(f"  Status: {status.status} ({status.request_counts.completed}/{status.request_counts.total})")
            
            if status.status == "completed":
                # Retrieve and save results
                output_file_id = status.output_file_id
                if output_file_id:
                    content = _retrieve_batch(client, output_file_id)
                    with open(batch_info["save_file_batch_response"], "w", encoding="utf-8") as file:
                        file.write(content)
                    _convert_to_json(
                        batch_info["save_file"], 
                        batch_info["save_file_batch"], 
                        batch_info["save_file_batch_response"]
                    )
                    _update_batch_status(batch_id, "completed", datetime.now().isoformat())
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


def _build_single_batch_request(
    model_info: RunnableModel, custom_id: str, conversation: List
):
    conversation_request = {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model_info["name"],
            "messages": conversation,
            "temperature": model_info["temperature"],
            "max_completion_tokens": model_info["max_tokens"],
        },
    }
    return conversation_request


def _upload_batch_input_file(client: openai.OpenAI, file_name: str):
    batch_input_file = client.files.create(file=open(file_name, "rb"), purpose="batch")
    return batch_input_file


def _create_batch(client: openai.OpenAI, batch_input_file_id: str):
    batch_info = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "ABAP LLM Benchmark"},
    )
    return batch_info


def _status_batch(client: openai.OpenAI, batch_id: str):
    batch = client.batches.retrieve(batch_id)
    return batch


def _retrieve_batch(client: openai.OpenAI, file_content: str):
    file_response = client.files.content(file_content)
    return file_response.text


def _convert_to_json(save_file: str, batch_file: str, batch_response_file: str):

    result: Dict[str, List] = {
        prompt: [[] for repetition in range(REPETITIONS)] for prompt in PROMPT_FILES
    }
    if os.path.exists(save_file):
        with open(save_file, "r", encoding="utf-8") as file:
            result = json.load(file)

    with open(batch_file, "r", encoding="utf-8") as file:
        batch_entries = [json.loads(m) for m in file.readlines()]

    with open(batch_response_file, "r", encoding="utf-8") as file:
        batch_responses = [json.loads(m) for m in file.readlines()]

    for batch_entry in batch_entries:
        for batch_response in batch_responses:
            if batch_entry["custom_id"] == batch_response["custom_id"]:

                id: str = batch_entry["custom_id"]
                prompt = id[1 : id.rindex("-R")]
                repetition = int(id[id.rindex("-R") + 2 :])

                messages = batch_entry["body"]["messages"]
                new_message = batch_response["response"]["body"]["choices"][0][
                    "message"
                ]
                new_message = {
                    "role": new_message["role"],
                    "content": remove_code_blocks(new_message["content"]),
                }
                messages.append(new_message)
                result[prompt][repetition] = messages

                break

    with open(save_file, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=4, ensure_ascii=False)


def generate_next_response_batch(
    client: openai.OpenAI,
    model_info: RunnableModel,
    save_file: str,
    save_file_batch: str,
    round_num: int = 0,
):
    save_file_batch_response = save_file_batch[:-6] + "_response.jsonl"
    conversations = []

    current = {}
    with open(save_file, "r", encoding="utf-8") as file:
        current = json.loads(file.read())

    # ── GUARD: refuse to generate next round if any chats still need SAP testing ──
    if any_needs_test(current):
        needs = sum(
            1 for chats in current.values()
            for c in chats if chat_needs_test(c)
        )
        print(
            f"[BLOCKED] {needs} conversation(s) still need SAP testing "
            f"(last message is assistant without feedback).\n"
            f"  Run SAP tests first:  python src/abap_test.py --model {model_info['name']} --mode resume\n"
            f"  Then retry:           python src/abap_test.py --model {model_info['name']} --mode retry --max-attempts 3"
        )
        return None

    for prompt_file in current:
        for i, repetition in enumerate(current[prompt_file]):
            custom_id = f"P{prompt_file}-R{i}"
            conversation = current[prompt_file][i]

            # Only generate for conversations that are WaitingForLLM
            # (last role is user feedback, not success, not infra, not maxed out)
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

            conversation_request = _build_single_batch_request(
                model_info, custom_id, conversation
            )
            conversations.append(conversation_request)

    if not conversations:
        print(f"[SKIP] No conversations need processing for round {round_num}")
        return None

    with open(save_file_batch, "w", encoding="utf-8") as file:
        for line in conversations:
            file.write(json.dumps(line, ensure_ascii=False) + "\n")

    batch_input_file = _upload_batch_input_file(client, save_file_batch)
    batch = _create_batch(client, batch_input_file.id)
    
    # Log the batch for tracking
    _log_batch(batch.id, model_info["name"], save_file, save_file_batch, 
               save_file_batch_response, round_num)
    
    return batch.id


def generate_first_response_batch(
    client, model_info: RunnableModel, save_file_batch: str, save_file: str = None
) -> str:
    if save_file is None:
        save_file = save_file_batch.replace("_batch.jsonl", ".json")
    save_file_batch_response = save_file_batch[:-6] + "_response.jsonl"

    # ── Non-destructive: load existing data and skip already-generated reps ──
    existing = {}
    if os.path.exists(save_file):
        with open(save_file, "r", encoding="utf-8") as f:
            existing = json.load(f)

    conversations = []
    skipped = 0
    for prompt_file in PROMPT_FILES:
        with open(f"dataset/prompts/{prompt_file}", "r", encoding="utf-8") as file:
            prompt_content = file.read()
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
            conversation_request = _build_single_batch_request(
                model_info, custom_id, conversation
            )
            conversations.append(conversation_request)

    if skipped > 0:
        print(f"[RESUME] Skipped {skipped} already-generated repetitions")

    if not conversations:
        print(f"[SKIP] All first-round responses already exist in {save_file}")
        return None

    with open(save_file_batch, "w", encoding="utf-8") as file:
        for line in conversations:
            file.write(json.dumps(line, ensure_ascii=False) + "\n")

    batch_input_file = _upload_batch_input_file(client, save_file_batch)
    batch = _create_batch(client, batch_input_file.id)
    
    # Log the batch for tracking
    _log_batch(batch.id, model_info["name"], save_file, save_file_batch, 
               save_file_batch_response, round_num=1)
    
    return batch.id


def wait_for_batch_and_save(
    client: openai.OpenAI,
    batch_id: str,
    save_file: str,
    save_file_batch: str,
    save_file_batch_response: str,
):
    print(f"[WAITING] Batch {batch_id}...")
    while True:
        status = _status_batch(client, batch_id)
        print(f"  Status: {status.status} ({status.request_counts.completed}/{status.request_counts.total})")
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
        with open(save_file_batch_response, "w", encoding="utf-8") as file:
            file.write(content)
        _convert_to_json(save_file, save_file_batch, save_file_batch_response)
        _update_batch_status(batch_id, "completed", datetime.now().isoformat())
        print(f"[COMPLETED] Results saved to {save_file}")
        return True
    return False


def list_all_batches():
    """Print a summary of all tracked batches."""
    tracking = _load_batch_tracking()
    if not tracking["batches"]:
        print("No batches tracked yet.")
        return
    
    print("\n=== OpenAI Batch Tracking ===")
    for batch in tracking["batches"]:
        print(f"  {batch['batch_id'][:20]}... | {batch['model_name']} | Round {batch['round_num']} | {batch['status']} | {batch['created_at']}")