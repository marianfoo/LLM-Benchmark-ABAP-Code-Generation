"""Mistral batch pipeline for ABAP code-generation benchmark.

Mirrors generate_llm_answers_batch_openai.py but uses the mistralai SDK
(files.upload → batch.jobs.create → batch.jobs.get → files.download).
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from mistralai import Mistral

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

BATCH_TRACKING_FILE = "data/mistral_batch_tracking.json"


# ---------------------------------------------------------------------------
# Tracking persistence
# ---------------------------------------------------------------------------

def _load_batch_tracking() -> Dict:
    if os.path.exists(BATCH_TRACKING_FILE):
        with open(BATCH_TRACKING_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"batches": []}


def _save_batch_tracking(tracking: Dict):
    os.makedirs(os.path.dirname(BATCH_TRACKING_FILE), exist_ok=True)
    with open(BATCH_TRACKING_FILE, "w", encoding="utf-8") as f:
        json.dump(tracking, f, indent=2)


def _log_batch(
    job_id: str,
    model_name: str,
    save_file: str,
    save_file_batch: str,
    save_file_batch_response: str,
    round_num: int,
):
    tracking = _load_batch_tracking()
    tracking["batches"].append({
        "batch_id": job_id,
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
    print(f"[BATCH LOGGED] ID: {job_id}, Model: {model_name}, Round: {round_num}")


def _update_batch_status(job_id: str, status: str, completed_at: Optional[str] = None):
    tracking = _load_batch_tracking()
    for batch in tracking["batches"]:
        if batch["batch_id"] == job_id:
            batch["status"] = status
            if completed_at:
                batch["completed_at"] = completed_at
            break
    _save_batch_tracking(tracking)


def get_pending_batches(model_name: Optional[str] = None) -> List[Dict]:
    tracking = _load_batch_tracking()
    pending = [b for b in tracking["batches"] if b["status"] == "pending"]
    if model_name:
        pending = [b for b in pending if b["model_name"] == model_name]
    return pending


# ---------------------------------------------------------------------------
# Mistral API helpers
# ---------------------------------------------------------------------------

def _build_single_batch_request(
    model_info: RunnableModel, custom_id: str, conversation: List
) -> Dict:
    """Build one JSONL request line for Mistral batch (no method/url fields)."""
    return {
        "custom_id": custom_id,
        "body": {
            "messages": conversation,
            "temperature": model_info["temperature"],
            "max_tokens": model_info["max_tokens"],
        },
    }


def _upload_and_create_job(
    client: Mistral, model_info: RunnableModel, batch_file_path: str
) -> str:
    """Upload the JSONL file and create a batch job. Returns job ID."""
    batch_data = client.files.upload(
        file={
            "file_name": Path(batch_file_path).name,
            "content": open(batch_file_path, "rb"),
        },
        purpose="batch",
    )
    job = client.batch.jobs.create(
        input_files=[batch_data.id],
        model=model_info["name"],
        endpoint="/v1/chat/completions",
        metadata={"description": "ABAP LLM Benchmark"},
    )
    return job.id


def _poll_job(client: Mistral, job_id: str):
    return client.batch.jobs.get(job_id=job_id)


def _download_results(client: Mistral, output_file_id: str) -> str:
    stream = client.files.download(file_id=output_file_id)
    raw = stream.read() if hasattr(stream, "read") else b"".join(stream.stream)
    return raw.decode("utf-8")


def _convert_to_json(save_file: str, batch_file: str, batch_response_file: str):
    """Merge Mistral batch responses into the main data JSON, same as OpenAI converter."""
    result: Dict[str, List] = {
        prompt: [[] for _ in range(REPETITIONS)] for prompt in PROMPT_FILES
    }
    if os.path.exists(save_file):
        with open(save_file, "r", encoding="utf-8") as f:
            result = json.load(f)

    with open(batch_file, "r", encoding="utf-8") as f:
        batch_entries = [json.loads(line) for line in f if line.strip()]

    with open(batch_response_file, "r", encoding="utf-8") as f:
        batch_responses = [json.loads(line) for line in f if line.strip()]

    for batch_entry in batch_entries:
        for batch_response in batch_responses:
            if batch_entry["custom_id"] == batch_response["custom_id"]:
                cid: str = batch_entry["custom_id"]
                prompt = cid[1 : cid.rindex("-R")]
                repetition = int(cid[cid.rindex("-R") + 2 :])

                messages = batch_entry["body"]["messages"]
                content = batch_response["response"]["body"]["choices"][0]["message"]["content"]
                messages.append({
                    "role": "assistant",
                    "content": remove_code_blocks(content),
                })
                result[prompt][repetition] = messages
                break

    with open(save_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Public entry points (same signatures as the OpenAI batch module)
# ---------------------------------------------------------------------------

def check_and_complete_pending_batches(
    client: Mistral, model_name: Optional[str] = None
) -> List[Dict]:
    pending = get_pending_batches(model_name)
    completed = []

    for batch_info in pending:
        job_id = batch_info["batch_id"]
        print(f"[CHECKING] Batch {job_id} (Model: {batch_info['model_name']}, Round: {batch_info['round_num']})")

        try:
            job = _poll_job(client, job_id)
            total = getattr(job, "total_requests", 0) or 0
            succeeded = getattr(job, "succeeded_requests", 0) or 0
            failed_cnt = getattr(job, "failed_requests", 0) or 0
            print(f"  Status: {job.status} ({succeeded + failed_cnt}/{total})")

            if job.status == "SUCCESS":
                output_file_id = getattr(job, "output_file", None)
                if output_file_id:
                    content = _download_results(client, output_file_id)
                    resp_path = batch_info["save_file_batch_response"]
                    with open(resp_path, "w", encoding="utf-8") as f:
                        f.write(content)
                    _convert_to_json(
                        batch_info["save_file"],
                        batch_info["save_file_batch"],
                        resp_path,
                    )
                    _update_batch_status(job_id, "completed", datetime.now().isoformat())
                    print(f"  [COMPLETED] Results saved to {batch_info['save_file']}")
                    completed.append(batch_info)
            elif job.status in ("FAILED", "TIMEOUT_EXCEEDED", "CANCELLED"):
                _update_batch_status(job_id, job.status)
                print(f"  [FAILED] Batch ended with status: {job.status}")
        except Exception as e:
            print(f"  [ERROR] Could not check batch: {e}")

    return completed


def generate_first_response_batch(
    client: Mistral,
    model_info: RunnableModel,
    save_file_batch: str,
    save_file: str = None,
) -> Optional[str]:
    if save_file is None:
        save_file = save_file_batch.replace("_batch.jsonl", ".json")
    save_file_batch_response = save_file_batch[:-6] + "_response.jsonl"

    existing = {}
    if os.path.exists(save_file):
        with open(save_file, "r", encoding="utf-8") as f:
            existing = json.load(f)

    conversations = []
    skipped = 0
    for prompt_file in PROMPT_FILES:
        with open(f"dataset/prompts/{prompt_file}", "r", encoding="utf-8") as f:
            prompt_content = f.read()
        for i in range(REPETITIONS):
            if prompt_file in existing:
                reps = existing[prompt_file]
                if i < len(reps) and reps[i] and assistant_count(reps[i]) > 0:
                    skipped += 1
                    continue

            conversation = [
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

    job_id = _upload_and_create_job(client, model_info, save_file_batch)
    _log_batch(job_id, model_info["name"], save_file, save_file_batch,
               save_file_batch_response, round_num=1)
    return job_id


def generate_next_response_batch(
    client: Mistral,
    model_info: RunnableModel,
    save_file: str,
    save_file_batch: str,
    round_num: int = 0,
) -> Optional[str]:
    save_file_batch_response = save_file_batch[:-6] + "_response.jsonl"
    conversations = []

    with open(save_file, "r", encoding="utf-8") as f:
        current = json.load(f)

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
        for i, conversation in enumerate(current[prompt_file]):
            if not conversation:
                continue
            if chat_is_success(conversation):
                continue
            if not chat_waiting_for_llm(conversation):
                continue
            if assistant_count(conversation) >= MAX_ASSISTANTS:
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

    job_id = _upload_and_create_job(client, model_info, save_file_batch)
    _log_batch(job_id, model_info["name"], save_file, save_file_batch,
               save_file_batch_response, round_num)
    return job_id


def wait_for_batch_and_save(
    client: Mistral,
    job_id: str,
    save_file: str,
    save_file_batch: str,
    save_file_batch_response: str,
) -> bool:
    print(f"[WAITING] Batch {job_id}...")
    while True:
        job = _poll_job(client, job_id)
        total = getattr(job, "total_requests", 0) or 0
        succeeded = getattr(job, "succeeded_requests", 0) or 0
        failed_cnt = getattr(job, "failed_requests", 0) or 0
        print(f"  Status: {job.status} ({succeeded + failed_cnt}/{total})")

        if job.status == "SUCCESS":
            break
        elif job.status in ("FAILED", "TIMEOUT_EXCEEDED", "CANCELLED"):
            _update_batch_status(job_id, job.status)
            print(f"[FAILED] Batch ended with status: {job.status}")
            return False
        time.sleep(30)

    output_file_id = getattr(job, "output_file", None)
    if output_file_id:
        content = _download_results(client, output_file_id)
        with open(save_file_batch_response, "w", encoding="utf-8") as f:
            f.write(content)
        _convert_to_json(save_file, save_file_batch, save_file_batch_response)
        _update_batch_status(job_id, "completed", datetime.now().isoformat())
        print(f"[COMPLETED] Results saved to {save_file}")
        return True
    return False


def list_all_batches():
    tracking = _load_batch_tracking()
    if not tracking["batches"]:
        print("No batches tracked yet.")
        return

    print("\n=== Mistral Batch Tracking ===")
    for batch in tracking["batches"]:
        print(
            f"  {batch['batch_id'][:20]}... | {batch['model_name']} | "
            f"Round {batch['round_num']} | {batch['status']} | {batch['created_at']}"
        )
