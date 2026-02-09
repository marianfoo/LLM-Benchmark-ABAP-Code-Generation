"""
Shared conversation-state helpers for the ABAP LLM Benchmark.

Every prompt+repetition chat in ``data/<model>.json`` is a list of messages
(system, user, assistant, user-feedback, assistant-fix, …).  The functions
below classify a chat into one of the following states used by the
generation and testing pipelines:

  NeedsSAPTest    – last message is ``assistant`` (code ready, feedback missing)
  WaitingForLLM   – last message is user feedback, not yet successful
  InfraRetriable  – last message is ``[INFRA] …`` transient ADT error
  Success         – a ``The unit tests were successful.`` feedback exists
  MaxedOut        – assistant_count >= MAX_ASSISTANTS (Round 0‥5 exhausted)

Import this module instead of duplicating classification logic.
"""

from __future__ import annotations

from typing import Dict, List

# Marker prefix used in user feedback messages for transient/infra errors.
# Must stay in sync with ``abap_interaction.INFRA_FEEDBACK_PREFIX``.
INFRA_FEEDBACK_PREFIX = "[INFRA] Transient ADT error"

# Paper methodology: up to 5 feedback loops → 6 total assistant messages
# (Round 0 = initial, Rounds 1–5 = corrections).
MAX_ASSISTANTS = 6
MAX_FEEDBACK_ROUNDS = 5  # 0-indexed: rounds 0..5


# ── counting helpers ─────────────────────────────────────────────────────

def assistant_count(chat: list) -> int:
    """Return the number of assistant messages in the conversation."""
    return sum(1 for msg in chat if msg.get("role") == "assistant")


def feedback_count(chat: list) -> int:
    """Return the number of user feedback messages (excludes the initial prompt)."""
    count = 0
    saw_assistant = False
    for msg in chat:
        if msg.get("role") == "assistant":
            saw_assistant = True
        elif msg.get("role") == "user" and saw_assistant:
            count += 1
    return count


def round_num(chat: list) -> int:
    """
    Current round number (0-indexed).
    Round 0 = first assistant response, Round N = (N+1)th assistant response.
    Returns -1 if no assistant message exists yet.
    """
    ac = assistant_count(chat)
    return ac - 1 if ac > 0 else -1


# ── state classification ─────────────────────────────────────────────────

def chat_is_success(chat: list) -> bool:
    """True when the conversation reached 'The unit tests were successful.'"""
    for msg in chat:
        if (
            msg.get("role") == "user"
            and msg.get("content") == "The unit tests were successful."
        ):
            return True
    return False


def chat_needs_test(chat: list) -> bool:
    """True when the chat ends with an assistant message (needs SAP testing)."""
    if not chat:
        return False
    last = chat[-1]
    if last.get("role") != "assistant":
        return False
    # Already succeeded – skip
    if chat_is_success(chat):
        return False
    return True


def chat_has_infra_feedback(chat: list) -> bool:
    """True when the last user message is a transient [INFRA] marker."""
    if not chat:
        return False
    last = chat[-1]
    return (
        last.get("role") == "user"
        and isinstance(last.get("content"), str)
        and last["content"].startswith(INFRA_FEEDBACK_PREFIX)
    )


def chat_waiting_for_llm(chat: list) -> bool:
    """
    True when the last message is from the user (waiting for next LLM response)
    and the conversation has not yet succeeded and is not an infra marker.
    """
    if not chat:
        return False
    if chat_is_success(chat):
        return False
    if chat_has_infra_feedback(chat):
        return False
    last = chat[-1]
    return last.get("role") == "user"


def chat_maxed_out(chat: list) -> bool:
    """True when the conversation has reached the maximum number of rounds."""
    return assistant_count(chat) >= MAX_ASSISTANTS


# ── aggregate scanning ───────────────────────────────────────────────────

def count_states(prompt_files: dict) -> dict:
    """
    Scan all conversations and return aggregate counts.

    Returns a dict with keys:
        total, success, needs_test, infra_retriable,
        waiting_llm, deterministic_fail, maxed_out
    """
    counts = {
        "total": 0,
        "success": 0,
        "needs_test": 0,
        "infra_retriable": 0,
        "waiting_llm": 0,
        "deterministic_fail": 0,
        "maxed_out": 0,
    }
    for _prompt_key, chats in prompt_files.items():
        for chat in chats:
            counts["total"] += 1
            if chat_is_success(chat):
                counts["success"] += 1
            elif chat_needs_test(chat):
                counts["needs_test"] += 1
            elif chat_has_infra_feedback(chat):
                counts["infra_retriable"] += 1
            elif chat_waiting_for_llm(chat):
                if chat_maxed_out(chat):
                    counts["maxed_out"] += 1
                else:
                    counts["waiting_llm"] += 1
            else:
                # Deterministic failure: last user feedback is real error,
                # not infra, not success, and chat has maxed out or is stale.
                counts["deterministic_fail"] += 1
    return counts


def any_needs_test(prompt_files: dict) -> bool:
    """Return True if any conversation still needs SAP testing."""
    for _prompt_key, chats in prompt_files.items():
        for chat in chats:
            if chat_needs_test(chat):
                return True
    return False


def print_status(model_name: str, prompt_files: dict, retry_state: dict | None = None):
    """Print a concise progress summary for the model."""
    c = count_states(prompt_files)
    total = c["total"]
    if total == 0:
        print(f"  No conversations found for {model_name}.")
        return

    total_prompts = len(prompt_files)
    print(f"\n{'='*60}")
    print(f"  Model: {model_name}")
    print(f"{'='*60}")
    print(f"  Total prompts          : {total_prompts}")
    print(f"  Total repetitions      : {total}")
    print(f"  ---")
    print(f"  Success (unit tests OK): {c['success']:5d}  ({c['success']/total*100:.1f}%)")
    print(f"  Needs SAP test (resume): {c['needs_test']:5d}  ({c['needs_test']/total*100:.1f}%)")
    print(f"  Retriable infra errors : {c['infra_retriable']:5d}  ({c['infra_retriable']/total*100:.1f}%)")
    print(f"  Waiting for LLM round  : {c['waiting_llm']:5d}  ({c['waiting_llm']/total*100:.1f}%)")
    print(f"  Maxed out (round cap)  : {c['maxed_out']:5d}  ({c['maxed_out']/total*100:.1f}%)")
    print(f"  Deterministic failures : {c['deterministic_fail']:5d}  ({c['deterministic_fail']/total*100:.1f}%)")

    if retry_state:
        maxed_retries = sum(1 for v in retry_state.values() if v >= 3)
        print(f"  ---")
        print(f"  Retry state entries    : {len(retry_state)}")
        print(f"  Maxed out (>= 3 tries) : {maxed_retries}")

    print()
