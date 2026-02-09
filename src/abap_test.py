#!/usr/bin/env python3
"""
Unified ABAP testing CLI with resume, retry and status modes.

Usage:
    # Check how many conversations still need SAP testing
    python src/abap_test.py --model claude-opus-4-5-20251101 --mode status

    # Run SAP/ADT tests on all untested conversations (safe to restart)
    python src/abap_test.py --model claude-opus-4-5-20251101 --mode resume

    # Retry only conversations that had transient/infra failures or missing
    # feedback (max 3 attempts per conversation, restartable)
    python src/abap_test.py --model claude-opus-4-5-20251101 --mode retry --max-attempts 3
"""

import argparse
import json
import os
import sys
import time

# Ensure src/ is on sys.path so we can import siblings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from abap_adt_py.adt_client import AdtClient
import abap_interaction
from abap_interaction import (
    INFRA_FEEDBACK_PREFIX,
    add_to_chat,
    create_tier_result,
    get_round_number,
    load_tiers,
    log_failure,
    run_single_prompt,
    save_tiers,
)
from chat_state import (
    chat_needs_test,
    chat_has_infra_feedback,
    chat_is_success,
    chat_waiting_for_llm,
    print_status,
    round_num as chat_round_num,
)


# =============================================================================
# Retry-state persistence (data/<model>_retry_state.json)
# =============================================================================

def _retry_state_path(model_name: str) -> str:
    return f"data/{model_name.replace(':', '_')}_retry_state.json"


def load_retry_state(model_name: str) -> dict:
    """Load retry-attempt counts. Keys: "<prompt>|<rep_idx>|<round_num>" -> int."""
    path = _retry_state_path(model_name)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_retry_state(model_name: str, state: dict):
    path = _retry_state_path(model_name)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _state_key(prompt_key: str, rep_idx: int, round_num: int) -> str:
    return f"{prompt_key}|{rep_idx}|{round_num}"


# =============================================================================
# Mode: status
# =============================================================================

def mode_status(model_name: str, prompt_files: dict):
    """Print a concise progress summary for the model."""
    state = load_retry_state(model_name)
    print_status(model_name, prompt_files, retry_state=state if state else None)


# =============================================================================
# Mode: resume  (run SAP tests on all untested conversations)
# =============================================================================

def mode_resume(
    model_name: str,
    prompt_files: dict,
    use_canonicalization: bool,
    use_abaplint: bool,
    abaplint_hard_gate: bool,
):
    """Run SAP/ADT tests on every conversation that needs it (idempotent)."""
    client = _create_client()

    filename = _data_path(model_name)
    all_tiers = load_tiers(model_name)

    tested = 0
    skipped = 0
    for prompt_key, chats in prompt_files.items():
        # Quick check: any reps need testing?
        any_needed = any(chat_needs_test(c) for c in chats)
        if not any_needed:
            skipped += 1
            continue

        updated_chats, prompt_tiers = run_single_prompt(
            client, prompt_key, chats,
            use_canonicalization=use_canonicalization,
            use_abaplint=use_abaplint,
            abaplint_hard_gate=abaplint_hard_gate,
            model_name=model_name,
        )

        # Merge results back
        prompt_files[prompt_key] = updated_chats
        all_tiers[prompt_key] = prompt_tiers

        # Incremental save after each prompt
        _save_data(filename, prompt_files)
        save_tiers(model_name, all_tiers)
        tested += 1

    print(f"\nResume complete: tested {tested} prompts, skipped {skipped}.")


# =============================================================================
# Mode: retry  (re-test transient/infra failures and missing feedback)
# =============================================================================

def mode_retry(
    model_name: str,
    prompt_files: dict,
    max_attempts: int,
    use_canonicalization: bool,
    use_abaplint: bool,
    abaplint_hard_gate: bool,
):
    """
    Re-test only conversations that:
      1. Have no feedback at all (last role is assistant, but we missed testing), OR
      2. Have an [INFRA] transient feedback marker.

    Retries up to max_attempts per conversation (across runs, via retry state).
    """
    client = _create_client()

    filename = _data_path(model_name)
    all_tiers = load_tiers(model_name)
    retry_state = load_retry_state(model_name)

    retried = 0
    skipped_max = 0
    skipped_ok = 0

    for prompt_key, chats in prompt_files.items():
        changed = False
        for rep_idx, chat in enumerate(chats):
            round_num = get_round_number(chat)
            key = _state_key(prompt_key, rep_idx, round_num)

            # Determine if this conversation is retriable
            is_missing = chat_needs_test(chat)
            is_infra = chat_has_infra_feedback(chat)

            if not is_missing and not is_infra:
                skipped_ok += 1
                continue

            # Check attempt budget
            attempts_so_far = retry_state.get(key, 0)
            if attempts_so_far >= max_attempts:
                skipped_max += 1
                continue

            # Roll back the [INFRA] feedback message so the same assistant
            # output is re-tested cleanly.
            if is_infra and chat[-1]["role"] == "user":
                chat.pop()

            # Increment attempt count
            attempts_so_far += 1
            retry_state[key] = attempts_so_far

            # Re-run SAP test for just this one conversation
            # We wrap in a single-element list so run_single_prompt processes it.
            single_chats = [chat]
            updated_chats, prompt_tiers = run_single_prompt(
                client, prompt_key, single_chats,
                use_canonicalization=use_canonicalization,
                use_abaplint=use_abaplint,
                abaplint_hard_gate=abaplint_hard_gate,
                model_name=model_name,
            )

            # Write back
            chats[rep_idx] = updated_chats[0]

            # Merge tier for this rep
            if prompt_key not in all_tiers:
                all_tiers[prompt_key] = [{} for _ in range(len(chats))]
            while len(all_tiers[prompt_key]) <= rep_idx:
                all_tiers[prompt_key].append({})
            all_tiers[prompt_key][rep_idx] = prompt_tiers[0]

            # Log the retry attempt
            last_content = chats[rep_idx][-1].get("content", "")
            was_transient = chats[rep_idx][-1].get("role") == "user" and last_content.startswith(INFRA_FEEDBACK_PREFIX)
            stage = "infra" if was_transient else "retry"
            log_failure(
                model_name=model_name,
                prompt_key=prompt_key,
                rep_idx=rep_idx,
                round_num=round_num,
                attempt=attempts_so_far,
                stage=stage,
                transient=was_transient,
                message=last_content[:200] if last_content else "(empty)",
            )

            changed = True
            retried += 1

        if changed:
            prompt_files[prompt_key] = chats
            _save_data(filename, prompt_files)
            save_tiers(model_name, all_tiers)
            save_retry_state(model_name, retry_state)

    # Final save of retry state
    save_retry_state(model_name, retry_state)

    print(f"\nRetry complete: retried {retried}, skipped (maxed out) {skipped_max}, "
          f"skipped (OK/deterministic) {skipped_ok}.")


# =============================================================================
# Shared helpers
# =============================================================================

def _data_path(model_name: str) -> str:
    return f"data/{model_name.replace(':', '_')}.json"


def _save_data(filename: str, prompt_files: dict):
    tmp = filename + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(prompt_files, f, indent=4, ensure_ascii=False)
    os.replace(tmp, filename)


def _create_client() -> AdtClient:
    """Create and login an ADT client (reuses abap_interaction module state)."""
    client = AdtClient(
        sap_host="http://localhost:50000",
        username="DEVELOPER",
        password="ABAPtr2023#00",
        client="001",
        language="EN",
    )
    client.login()
    # Store reference so _retry_adt can re-login on auth errors
    abap_interaction._current_client = client
    return client


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ABAP testing CLI: resume, retry, and status for SAP/ADT tests.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model", "-m", required=True, help="Model name (e.g. claude-opus-4-5-20251101)")
    parser.add_argument(
        "--mode", required=True,
        choices=["resume", "retry", "status"],
        help="resume = run tests on untested chats; retry = re-test infra/missing failures; status = print progress",
    )
    parser.add_argument("--max-attempts", type=int, default=3, help="Max retry attempts per conversation (retry mode only, default: 3)")
    parser.add_argument("--use-canonicalization", type=str, default="true", help="Use canonicalization (true/false, default: true)")
    parser.add_argument("--use-abaplint", type=str, default="true", help="Run abaplint preflight (true/false, default: true)")
    parser.add_argument("--abaplint-hard-gate", type=str, default="false", help="Hard gate on abaplint parse errors (true/false, default: false)")

    args = parser.parse_args()

    model_name = args.model
    use_canonicalization = args.use_canonicalization.lower() in ("true", "1", "yes")
    use_abaplint = args.use_abaplint.lower() in ("true", "1", "yes")
    abaplint_hard_gate = args.abaplint_hard_gate.lower() in ("true", "1", "yes")

    # Load model data
    data_file = _data_path(model_name)
    if not os.path.exists(data_file):
        print(f"Error: data file not found: {data_file}")
        sys.exit(1)

    with open(data_file, "r", encoding="utf-8") as f:
        prompt_files = json.load(f)

    if args.mode == "status":
        mode_status(model_name, prompt_files)
    elif args.mode == "resume":
        mode_resume(model_name, prompt_files, use_canonicalization, use_abaplint, abaplint_hard_gate)
    elif args.mode == "retry":
        mode_retry(model_name, prompt_files, args.max_attempts, use_canonicalization, use_abaplint, abaplint_hard_gate)


if __name__ == "__main__":
    main()
