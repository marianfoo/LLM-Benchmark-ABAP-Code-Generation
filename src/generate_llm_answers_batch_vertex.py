"""Google Vertex AI Batch Prediction implementation for Gemini models.

Uses the google-genai SDK to submit batch jobs via Vertex AI, which:
  - Has no predefined rate limits (dynamically allocated)
  - Offers 50% cost reduction vs real-time inference
  - Uses GCS buckets for input/output

Requires:
  - gcloud auth application-default login
  - GOOGLE_CLOUD_PROJECT env var
  - GCS_BUCKET_NAME env var
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional

from google import genai
from google.genai.types import CreateBatchJobConfig, HttpOptions, JobState
from google.cloud import storage as gcs

from llms import RunnableModel
from generate_llm_answers import REPETITIONS, PROMPT_FILES, SYSTEM_PROMPT, remove_code_blocks
from chat_state import (
    chat_is_success,
    chat_needs_test,
    chat_waiting_for_llm,
    assistant_count,
    any_needs_test,
    MAX_ASSISTANTS,
)

BATCH_TRACKING_FILE = "data/vertex_batch_tracking.json"


# =============================================================================
# GCP config helpers
# =============================================================================

def _get_project() -> str:
    project = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
    if not project:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT env var not set")
    return project


def _get_bucket_name() -> str:
    bucket = os.getenv("GCS_BUCKET_NAME", "").strip()
    if not bucket:
        raise RuntimeError("GCS_BUCKET_NAME env var not set")
    return bucket


def _make_genai_client() -> genai.Client:
    """Create a google-genai client configured for Vertex AI.

    Uses 'global' location which supports batch prediction for preview models
    (e.g. gemini-3.1-pro-preview) that are not yet available in regional endpoints.
    """
    return genai.Client(
        vertexai=True,
        project=_get_project(),
        location="global",
        http_options=HttpOptions(api_version="v1"),
    )


def _make_gcs_client() -> gcs.Client:
    return gcs.Client(project=_get_project())


# =============================================================================
# Format conversion: OpenAI-style messages -> Vertex AI batch JSONL
# =============================================================================

def _convert_messages_to_vertex(
    model_info: RunnableModel,
    custom_id: str,
    conversation: List[Dict],
) -> Dict:
    """Convert an OpenAI-style message list to a Vertex AI batch request line.

    Based on Adrian's conversion script. Produces:
    {
        "custom_id": "P...-R0",
        "request": {
            "contents": [...],
            "systemInstruction": {...},
            "generationConfig": {...}
        }
    }
    """
    vertex_contents = []
    system_instruction = None

    for msg in conversation:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role in ("system", "developer"):
            system_instruction = {"parts": [{"text": content}]}
            continue

        vertex_role = "model" if role == "assistant" else role
        vertex_contents.append({"role": vertex_role, "parts": [{"text": content}]})

    request_payload = {"contents": vertex_contents}

    if system_instruction:
        request_payload["systemInstruction"] = system_instruction

    generation_config = {}
    if "temperature" in model_info:
        generation_config["temperature"] = model_info["temperature"]
    if "max_tokens" in model_info:
        generation_config["maxOutputTokens"] = model_info["max_tokens"]
    if generation_config:
        request_payload["generationConfig"] = generation_config

    return {
        "custom_id": custom_id,
        "request": request_payload,
    }


# =============================================================================
# GCS helpers
# =============================================================================

def _upload_to_gcs(local_path: str, gcs_path: str) -> str:
    """Upload a local file to GCS. Returns gs:// URI."""
    bucket_name = _get_bucket_name()
    client = _make_gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    uri = f"gs://{bucket_name}/{gcs_path}"
    print(f"  [GCS] Uploaded {local_path} -> {uri}")
    return uri


def _download_from_gcs_prefix(gcs_prefix: str, local_dir: str) -> List[str]:
    """Download all files under a GCS prefix. Returns list of local paths."""
    bucket_name = _get_bucket_name()
    client = _make_gcs_client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=gcs_prefix))

    os.makedirs(local_dir, exist_ok=True)
    local_paths = []
    for blob in blobs:
        if blob.name.endswith("/"):
            continue
        local_path = os.path.join(local_dir, os.path.basename(blob.name))
        blob.download_to_filename(local_path)
        local_paths.append(local_path)
        print(f"  [GCS] Downloaded gs://{bucket_name}/{blob.name} -> {local_path}")

    return local_paths


# =============================================================================
# Batch tracking
# =============================================================================

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
    job_name: str,
    model_name: str,
    save_file: str,
    save_file_batch: str,
    save_file_batch_response: str,
    round_num: int,
    gcs_input_uri: str,
    gcs_output_prefix: str,
):
    tracking = _load_batch_tracking()
    tracking["batches"].append({
        "job_name": job_name,
        "model_name": model_name,
        "save_file": save_file,
        "save_file_batch": save_file_batch,
        "save_file_batch_response": save_file_batch_response,
        "round_num": round_num,
        "gcs_input_uri": gcs_input_uri,
        "gcs_output_prefix": gcs_output_prefix,
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "completed_at": None,
    })
    _save_batch_tracking(tracking)
    print(f"[BATCH LOGGED] Job: {job_name}, Model: {model_name}, Round: {round_num}")


def _update_batch_status(job_name: str, status: str, completed_at: Optional[str] = None):
    tracking = _load_batch_tracking()
    for batch in tracking["batches"]:
        if batch["job_name"] == job_name:
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


# =============================================================================
# Submit batch job
# =============================================================================

def _submit_vertex_batch(
    model_name: str,
    local_jsonl_path: str,
    round_num: int,
) -> tuple:
    """Upload JSONL to GCS and submit a Vertex AI batch job.

    Returns (job_name, gcs_input_uri, gcs_output_prefix).
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gcs_input_path = f"batch_input/{model_name}/{timestamp}_round{round_num}.jsonl"
    gcs_output_prefix = f"batch_output/{model_name}/{timestamp}_round{round_num}/"

    gcs_input_uri = _upload_to_gcs(local_jsonl_path, gcs_input_path)
    gcs_output_uri = f"gs://{_get_bucket_name()}/{gcs_output_prefix}"

    client = _make_genai_client()
    job = client.batches.create(
        model=model_name,
        src=gcs_input_uri,
        config=CreateBatchJobConfig(dest=gcs_output_uri),
    )

    print(f"[SUBMITTED] Vertex AI batch job: {job.name}")
    print(f"  Input:  {gcs_input_uri}")
    print(f"  Output: {gcs_output_uri}")

    return job.name, gcs_input_uri, gcs_output_prefix


# =============================================================================
# Parse Vertex AI batch output
# =============================================================================

def _parse_vertex_responses(response_files: List[str]) -> Dict[str, str]:
    """Parse Vertex AI output JSONL files into {custom_id: response_text} mapping.

    Vertex output format per line:
    {
        "status": "",
        "processed_time": "...",
        "request": {...},
        "response": {
            "candidates": [{
                "content": {"parts": [{"text": "..."}]},
                ...
            }],
            ...
        },
        "custom_id": "P...-R0"  // passed through from input
    }
    """
    results = {}
    for filepath in response_files:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)

                custom_id = entry.get("custom_id")
                if not custom_id:
                    # Fallback: try to find custom_id in the request
                    continue

                status = entry.get("status", "")
                if status:
                    print(f"  [WARN] Request {custom_id} failed: {status}")
                    continue

                response = entry.get("response", {})
                candidates = response.get("candidates", [])
                if not candidates:
                    print(f"  [WARN] Request {custom_id}: no candidates in response")
                    continue

                parts = candidates[0].get("content", {}).get("parts", [])
                if not parts:
                    print(f"  [WARN] Request {custom_id}: no parts in candidate")
                    continue

                text = parts[0].get("text", "")
                results[custom_id] = text

    return results


def _convert_vertex_to_json(
    save_file: str,
    save_file_batch: str,
    response_files: List[str],
):
    """Merge Vertex AI batch responses back into the benchmark's standard JSON format."""
    result: Dict[str, List] = {
        prompt: [[] for _ in range(REPETITIONS)] for prompt in PROMPT_FILES
    }
    if os.path.exists(save_file):
        with open(save_file, "r", encoding="utf-8") as f:
            existing = json.load(f)
        # Merge existing data into the full prompt dict, padding lists to REPETITIONS
        for key, val in existing.items():
            if isinstance(val, list):
                # Pad to REPETITIONS if shorter
                while len(val) < REPETITIONS:
                    val.append([])
            result[key] = val

    # Load the input batch to get the original conversations
    with open(save_file_batch, "r", encoding="utf-8") as f:
        batch_entries = {
            json.loads(line)["custom_id"]: json.loads(line)
            for line in f if line.strip()
        }

    # Parse responses
    responses = _parse_vertex_responses(response_files)

    matched = 0
    for custom_id, response_text in responses.items():
        entry = batch_entries.get(custom_id)
        if not entry:
            print(f"  [WARN] Response for unknown custom_id: {custom_id}")
            continue

        prompt = custom_id[1:custom_id.rindex("-R")]
        repetition = int(custom_id[custom_id.rindex("-R") + 2:])

        # Reconstruct the OpenAI-style messages from the Vertex request
        request = entry["request"]
        messages = []

        # System instruction -> system message
        sys_instr = request.get("systemInstruction")
        if sys_instr:
            sys_text = sys_instr.get("parts", [{}])[0].get("text", "")
            if sys_text:
                messages.append({"role": "system", "content": sys_text})

        # Contents -> user/assistant messages
        for content_item in request.get("contents", []):
            role = content_item.get("role", "user")
            text = content_item.get("parts", [{}])[0].get("text", "")
            openai_role = "assistant" if role == "model" else role
            messages.append({"role": openai_role, "content": text})

        # Append the new assistant response
        messages.append({
            "role": "assistant",
            "content": remove_code_blocks(response_text),
        })

        result[prompt][repetition] = messages
        matched += 1

    print(f"  [MERGED] {matched} responses into {save_file}")

    with open(save_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)


# =============================================================================
# Public API: check / complete pending batches
# =============================================================================

COMPLETED_STATES = {
    JobState.JOB_STATE_SUCCEEDED,
    JobState.JOB_STATE_FAILED,
    JobState.JOB_STATE_CANCELLED,
    JobState.JOB_STATE_PAUSED,
}


def check_and_complete_pending_batches(model_name: Optional[str] = None) -> List[Dict]:
    """Check all pending Vertex batches and process any that have finished."""
    pending = get_pending_batches(model_name)
    completed = []
    client = _make_genai_client()

    for batch_info in pending:
        job_name = batch_info["job_name"]
        print(f"[CHECKING] Job {job_name} (Model: {batch_info['model_name']}, Round: {batch_info['round_num']})")

        try:
            job = client.batches.get(name=job_name)
            print(f"  State: {job.state}")

            if job.state == JobState.JOB_STATE_SUCCEEDED:
                # Download output from GCS
                gcs_prefix = batch_info["gcs_output_prefix"]
                local_dir = os.path.dirname(batch_info["save_file_batch_response"])
                response_files = _download_from_gcs_prefix(gcs_prefix, local_dir)

                if response_files:
                    _convert_vertex_to_json(
                        batch_info["save_file"],
                        batch_info["save_file_batch"],
                        response_files,
                    )
                    _update_batch_status(job_name, "completed", datetime.now().isoformat())
                    print(f"  [COMPLETED] Results saved to {batch_info['save_file']}")
                    completed.append(batch_info)
                else:
                    print(f"  [WARN] No output files found at gs://{_get_bucket_name()}/{gcs_prefix}")

            elif job.state in (JobState.JOB_STATE_FAILED, JobState.JOB_STATE_CANCELLED):
                _update_batch_status(job_name, str(job.state))
                print(f"  [FAILED] Job ended with state: {job.state}")

        except Exception as e:
            print(f"  [ERROR] Could not check job: {e}")

    return completed


# =============================================================================
# Public API: generate batch requests
# =============================================================================

def generate_first_response_batch(
    model_info: RunnableModel,
    save_file_batch: str,
    save_file: str = None,
) -> Optional[str]:
    """Submit a Vertex AI batch for all Round-0 (first) responses.

    Non-destructive: skips prompt+rep combos that already have an assistant response.
    Returns the job name, or None if there is nothing to submit.
    """
    if save_file is None:
        save_file = save_file_batch.replace("_batch.jsonl", ".json")
    save_file_batch_response = save_file_batch[:-6] + "_response.jsonl"

    existing: Dict = {}
    if os.path.exists(save_file):
        with open(save_file, "r", encoding="utf-8") as f:
            existing = json.load(f)

    requests = []
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
            requests.append(
                _convert_messages_to_vertex(model_info, custom_id, conversation)
            )

    if skipped > 0:
        print(f"[RESUME] Skipped {skipped} already-generated repetitions")

    if not requests:
        print(f"[SKIP] All first-round responses already exist in {save_file}")
        return None

    # Write local JSONL
    with open(save_file_batch, "w", encoding="utf-8") as f:
        for line in requests:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    print(f"[INFO] Submitting {len(requests)} requests for round 0")

    job_name, gcs_input_uri, gcs_output_prefix = _submit_vertex_batch(
        model_info["name"], save_file_batch, round_num=0,
    )
    _log_batch(
        job_name, model_info["name"], save_file, save_file_batch,
        save_file_batch_response, round_num=0,
        gcs_input_uri=gcs_input_uri, gcs_output_prefix=gcs_output_prefix,
    )
    return job_name


def generate_next_response_batch(
    model_info: RunnableModel,
    save_file: str,
    save_file_batch: str,
    round_num: int = 0,
) -> Optional[str]:
    """Submit a Vertex AI batch for one correction round (WaitingForLLM conversations only).

    Returns the job name, or None if there is nothing to submit.
    """
    save_file_batch_response = save_file_batch[:-6] + "_response.jsonl"

    with open(save_file, "r", encoding="utf-8") as f:
        current: Dict = json.load(f)

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

    requests = []
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
            requests.append(
                _convert_messages_to_vertex(model_info, custom_id, conversation)
            )

    if not requests:
        print(f"[SKIP] No conversations need processing for round {round_num}")
        return None

    with open(save_file_batch, "w", encoding="utf-8") as f:
        for line in requests:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    print(f"[INFO] Submitting {len(requests)} requests for round {round_num}")

    job_name, gcs_input_uri, gcs_output_prefix = _submit_vertex_batch(
        model_info["name"], save_file_batch, round_num=round_num,
    )
    _log_batch(
        job_name, model_info["name"], save_file, save_file_batch,
        save_file_batch_response, round_num,
        gcs_input_uri=gcs_input_uri, gcs_output_prefix=gcs_output_prefix,
    )
    return job_name


# =============================================================================
# Public API: wait / poll for completion
# =============================================================================

def wait_for_batch_and_save(
    job_name: str,
    save_file: str,
    save_file_batch: str,
    save_file_batch_response: str,
) -> bool:
    """Poll the Vertex AI batch job until completion, then download and merge results."""
    client = _make_genai_client()
    print(f"[WAITING] Job {job_name}...")

    while True:
        job = client.batches.get(name=job_name)
        print(f"  State: {job.state}")

        if job.state == JobState.JOB_STATE_SUCCEEDED:
            break
        elif job.state in (JobState.JOB_STATE_FAILED, JobState.JOB_STATE_CANCELLED):
            # Find tracking entry for gcs_output_prefix
            tracking = _load_batch_tracking()
            for batch in tracking["batches"]:
                if batch["job_name"] == job_name:
                    _update_batch_status(job_name, str(job.state))
                    break
            print(f"[FAILED] Job ended with state: {job.state}")
            return False

        time.sleep(30)

    # Find the GCS output prefix from tracking
    tracking = _load_batch_tracking()
    gcs_output_prefix = None
    for batch in tracking["batches"]:
        if batch["job_name"] == job_name:
            gcs_output_prefix = batch["gcs_output_prefix"]
            break

    if not gcs_output_prefix:
        print(f"[ERROR] Could not find GCS output prefix for job {job_name}")
        return False

    local_dir = os.path.dirname(save_file_batch_response) or "data"
    response_files = _download_from_gcs_prefix(gcs_output_prefix, local_dir)

    if response_files:
        _convert_vertex_to_json(save_file, save_file_batch, response_files)
        _update_batch_status(job_name, "completed", datetime.now().isoformat())
        print(f"[COMPLETED] Results saved to {save_file}")
        return True

    print(f"[WARN] No output files found")
    return False


def list_all_batches():
    """Print a summary of all tracked Vertex batches."""
    tracking = _load_batch_tracking()
    if not tracking["batches"]:
        print("No Vertex batches tracked yet.")
        return

    print("\n=== Vertex AI Batch Tracking ===")
    for batch in tracking["batches"]:
        print(
            f"  {batch['job_name'][:40]}... | {batch['model_name']} "
            f"| Round {batch['round_num']} | {batch['status']} | {batch['created_at']}"
        )
