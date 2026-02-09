#!/usr/bin/env python3
"""
Parallel prompt runner with dynamic queue (work-stealing) for SAP ADT tests.

Each worker:
  1. Acquires file lock on the queue file
  2. Claims the next 'pending' prompt (sets it to 'in_progress')
  3. Releases lock
  4. Processes all repetitions for that prompt (login once, reuse session)
  5. Acquires lock again, writes results to shared JSON + tiers, marks prompt 'done'
  6. Repeats until no 'pending' prompts remain

Usage:
    python src/parallel_runner.py --model <name> --workers 5 [--use-canonicalization false] [--dry-run]
"""

import argparse
import fcntl
import json
import multiprocessing
import os
import sys
import time
from datetime import datetime, timezone

# Ensure src/ is on sys.path so we can import siblings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from abap_adt_py.adt_client import AdtClient
from abap_interaction import (
    run_single_prompt,
    load_tiers,
    save_tiers,
    _current_client,
    ADT_THROTTLE_DELAY,
)
import abap_interaction  # for setting module-level _current_client


# =============================================================================
# Queue helpers (file-based with fcntl locking)
# =============================================================================

def _queue_path(model_name: str) -> str:
    """Return the path to the queue JSON file for a given model."""
    return f"data/{model_name.replace(':', '_')}_queue.json"


def _lock_path(model_name: str) -> str:
    """Return the path to the lock file next to the queue."""
    return _queue_path(model_name) + ".lock"


def _read_queue(model_name: str) -> dict:
    """Read queue JSON (caller must hold lock)."""
    path = _queue_path(model_name)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_queue(model_name: str, queue: dict):
    """Write queue JSON atomically (caller must hold lock)."""
    path = _queue_path(model_name)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(queue, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)  # atomic on POSIX


def _locked(lock_fd):
    """Context-manager-style acquire/release for an fcntl exclusive lock."""

    class _Lock:
        def __enter__(self):
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            return self

        def __exit__(self, *_):
            fcntl.flock(lock_fd, fcntl.LOCK_UN)

    return _Lock()


# =============================================================================
# Queue management
# =============================================================================

STALE_TIMEOUT_SECONDS = 600  # 10 minutes – reclaim in_progress older than this


def create_queue(model_name: str):
    """
    Create (or reset) a prompt queue from the model's data file.
    Any already-existing 'done' entries are preserved.
    Stale 'in_progress' entries (older than timeout) are reclaimed as 'pending'.
    """
    data_path = f"data/{model_name.replace(':', '_')}.json"
    with open(data_path, "r", encoding="utf-8") as f:
        prompt_files = json.load(f)

    queue_path = _queue_path(model_name)
    existing_queue = {}
    if os.path.exists(queue_path):
        with open(queue_path, "r", encoding="utf-8") as f:
            existing_queue = json.load(f)

    queue = {}
    now = datetime.now(timezone.utc).isoformat()
    for prompt_key in prompt_files:
        existing_entry = existing_queue.get(prompt_key, {})
        status = existing_entry.get("status", "pending")

        if status == "done":
            # Already finished – keep it
            queue[prompt_key] = existing_entry
        elif status == "in_progress":
            # Check for staleness
            started = existing_entry.get("started_at", "")
            if started:
                started_dt = datetime.fromisoformat(started)
                elapsed = (datetime.now(timezone.utc) - started_dt).total_seconds()
                if elapsed > STALE_TIMEOUT_SECONDS:
                    # Reclaim stale entry
                    queue[prompt_key] = {"status": "pending"}
                else:
                    queue[prompt_key] = existing_entry
            else:
                queue[prompt_key] = {"status": "pending"}
        else:
            queue[prompt_key] = {"status": "pending"}

    _write_queue(model_name, queue)

    total = len(queue)
    done = sum(1 for v in queue.values() if v.get("status") == "done")
    pending = sum(1 for v in queue.values() if v.get("status") == "pending")
    in_progress = sum(1 for v in queue.values() if v.get("status") == "in_progress")
    print(f"Queue created: {total} prompts ({done} done, {in_progress} in_progress, {pending} pending)")
    return queue


def claim_next_prompt(model_name: str, lock_fd) -> str | None:
    """
    Claim the next pending prompt. Returns the prompt key, or None if none left.
    Must be called with lock_fd open (lock acquired inside).
    """
    with _locked(lock_fd):
        queue = _read_queue(model_name)

        for prompt_key, entry in queue.items():
            if entry.get("status") == "pending":
                queue[prompt_key] = {
                    "status": "in_progress",
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "worker_pid": os.getpid(),
                }
                _write_queue(model_name, queue)
                return prompt_key

    return None  # No pending prompts left


def mark_done(model_name: str, prompt_key: str, lock_fd):
    """Mark a prompt as done in the queue."""
    with _locked(lock_fd):
        queue = _read_queue(model_name)
        queue[prompt_key] = {
            "status": "done",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "worker_pid": os.getpid(),
        }
        _write_queue(model_name, queue)


def save_prompt_results(
    model_name: str,
    prompt_key: str,
    updated_chats: list,
    prompt_tiers: list,
    lock_fd,
):
    """
    Atomically merge the results for one prompt back into the shared data
    files (model JSON and tiers JSON). Caller does not need to hold lock;
    locking is handled inside.
    """
    data_path = f"data/{model_name.replace(':', '_')}.json"

    with _locked(lock_fd):
        # --- merge chats ---
        with open(data_path, "r", encoding="utf-8") as f:
            all_data = json.load(f)
        all_data[prompt_key] = updated_chats
        tmp = data_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(all_data, f, indent=4, ensure_ascii=False)
        os.replace(tmp, data_path)

        # --- merge tiers ---
        all_tiers = load_tiers(model_name)
        all_tiers[prompt_key] = prompt_tiers
        save_tiers(model_name, all_tiers)


# =============================================================================
# Worker process
# =============================================================================

def worker_main(
    worker_id: int,
    model_name: str,
    use_canonicalization: bool,
    use_abaplint: bool,
    abaplint_hard_gate: bool,
):
    """
    Worker entry point. Logs into SAP once, then loops:
    claim prompt -> process -> save results -> mark done.
    """
    # Create a per-worker ADT client
    client = AdtClient(
        sap_host="http://localhost:50000",
        username="DEVELOPER",
        password="ABAPtr2023#00",
        client="001",
        language="EN",
    )
    client.login()
    # Set module-level _current_client so _retry_adt can re-login
    abap_interaction._current_client = client

    lock_file = _lock_path(model_name)
    lock_fd = open(lock_file, "a+")

    data_path = f"data/{model_name.replace(':', '_')}.json"
    processed = 0

    try:
        while True:
            prompt_key = claim_next_prompt(model_name, lock_fd)
            if prompt_key is None:
                break  # All prompts done

            # Read the chats for this prompt (under lock, so we get a consistent snapshot)
            with _locked(lock_fd):
                with open(data_path, "r", encoding="utf-8") as f:
                    all_data = json.load(f)
                chats = all_data[prompt_key]

            print(f"[Worker {worker_id} / PID {os.getpid()}] Processing: {prompt_key}")
            t0 = time.time()

            updated_chats, prompt_tiers = run_single_prompt(
                client,
                prompt_key,
                chats,
                use_canonicalization=use_canonicalization,
                use_abaplint=use_abaplint,
                abaplint_hard_gate=abaplint_hard_gate,
                model_name=model_name,
            )

            # Save results (atomically, under lock)
            save_prompt_results(model_name, prompt_key, updated_chats, prompt_tiers, lock_fd)
            mark_done(model_name, prompt_key, lock_fd)

            elapsed = time.time() - t0
            processed += 1
            print(f"[Worker {worker_id}] Done: {prompt_key} ({elapsed:.1f}s) | total processed: {processed}")

    finally:
        lock_fd.close()

    print(f"[Worker {worker_id}] Finished. Processed {processed} prompts.")


# =============================================================================
# Monitor: live progress display
# =============================================================================

def print_progress(model_name: str):
    """Print current queue status."""
    path = _queue_path(model_name)
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        queue = json.load(f)

    total = len(queue)
    done = sum(1 for v in queue.values() if v.get("status") == "done")
    in_progress = sum(1 for v in queue.values() if v.get("status") == "in_progress")
    pending = total - done - in_progress
    print(f"  Progress: {done}/{total} done, {in_progress} in_progress, {pending} pending")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run SAP ADT tests in parallel with a shared prompt queue."
    )
    parser.add_argument("--model", required=True, help="Model name (e.g. claude-opus-4-5-20251101)")
    parser.add_argument("--workers", type=int, default=5, help="Number of parallel workers (default: 5)")
    parser.add_argument(
        "--use-canonicalization",
        type=str,
        default="false",
        help="Use canonicalization (true/false, default: false)",
    )
    parser.add_argument(
        "--use-abaplint",
        type=str,
        default="true",
        help="Run abaplint preflight (true/false, default: true)",
    )
    parser.add_argument(
        "--abaplint-hard-gate",
        type=str,
        default="false",
        help="Hard gate on abaplint parse errors (true/false, default: false)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Create queue and exit without running workers")
    args = parser.parse_args()

    use_canonicalization = args.use_canonicalization.lower() in ("true", "1", "yes")
    use_abaplint = args.use_abaplint.lower() in ("true", "1", "yes")
    abaplint_hard_gate = args.abaplint_hard_gate.lower() in ("true", "1", "yes")

    model_name = args.model
    num_workers = args.workers

    # Verify data file exists
    data_path = f"data/{model_name.replace(':', '_')}.json"
    if not os.path.exists(data_path):
        print(f"Error: data file not found: {data_path}")
        sys.exit(1)

    # Create / reset queue (preserves 'done' entries for restartability)
    create_queue(model_name)

    if args.dry_run:
        print("Dry run – queue created. Exiting.")
        return

    # Ensure lock file exists
    lock_file = _lock_path(model_name)
    open(lock_file, "a+").close()

    print(f"\nStarting {num_workers} workers for model '{model_name}'...")
    print(f"  canonicalization={use_canonicalization}, abaplint={use_abaplint}, hard_gate={abaplint_hard_gate}\n")

    # Spawn workers via multiprocessing
    processes = []
    for i in range(num_workers):
        p = multiprocessing.Process(
            target=worker_main,
            args=(i, model_name, use_canonicalization, use_abaplint, abaplint_hard_gate),
            daemon=False,
        )
        processes.append(p)
        p.start()
        # Small stagger to avoid thundering herd on SAP login
        time.sleep(1.0)

    # Monitor loop: print progress every 10s until all workers finish
    try:
        while any(p.is_alive() for p in processes):
            time.sleep(10)
            print_progress(model_name)
    except KeyboardInterrupt:
        print("\nInterrupted! Waiting for workers to finish current prompt...")
        for p in processes:
            p.join(timeout=30)

    # Final join
    for p in processes:
        p.join()

    print_progress(model_name)

    # Check exit codes
    failed = [i for i, p in enumerate(processes) if p.exitcode != 0]
    if failed:
        print(f"\nWarning: workers {failed} exited with non-zero codes.")
    else:
        print("\nAll workers finished successfully.")

    # Cleanup queue file
    queue_path = _queue_path(model_name)
    lock_path = _lock_path(model_name)
    try:
        queue = _read_queue(model_name)
        remaining = sum(1 for v in queue.values() if v.get("status") != "done")
        if remaining == 0:
            os.remove(queue_path)
            os.remove(lock_path)
            print("Queue files cleaned up.")
        else:
            print(f"{remaining} prompts not done – queue file preserved for re-run.")
    except Exception:
        pass


if __name__ == "__main__":
    main()
