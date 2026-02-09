#!/usr/bin/env python3
"""
LLM generation CLI – single entry-point for generating / resuming model responses.

Modes:
    status           Print conversation state counts (no API calls).
    complete-pending Complete any pending provider batch jobs (Anthropic / OpenAI).
    first            Generate initial (Round 0) assistant responses.
                     Non-destructive: skips prompt+rep combos that already have
                     an assistant response.
    next             Generate exactly one correction round for WaitingForLLM chats.
                     Refuses to run while any chats still NeedSAPTest (guard).

Usage examples:
    python src/llm_generate.py --model claude-opus-4-5-20251101 --mode status
    python src/llm_generate.py --model claude-opus-4-5-20251101 --mode first
    python src/llm_generate.py --model claude-opus-4-5-20251101 --mode next
    python src/llm_generate.py --model claude-opus-4-5-20251101 --mode complete-pending
"""

import argparse
import asyncio
import json
import os
import sys

# Ensure src/ is on sys.path so we can import siblings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

load_dotenv()

from llms import API_PROVIDERS, MODELS_TO_RUN, RunnableModel, get_provider_api_key
from chat_state import (
    print_status,
    count_states,
    any_needs_test,
    MAX_ASSISTANTS,
    MAX_FEEDBACK_ROUNDS,
)


# =============================================================================
# Helpers
# =============================================================================

def _data_path(model_name: str) -> str:
    return f"data/{model_name.replace(':', '_')}.json"


def _get_model_info(model_name: str) -> RunnableModel | None:
    """Find a model by name (case-insensitive, partial match)."""
    name_lower = model_name.lower()
    for model in MODELS_TO_RUN:
        if model["name"].lower() == name_lower or name_lower in model["name"].lower():
            return model
    return None


def _load_data(model_name: str) -> dict | None:
    """Load model data JSON. Returns None if file doesn't exist."""
    path = _data_path(model_name)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# Mode: status
# =============================================================================

def mode_status(model_name: str):
    """Print conversation state summary."""
    data = _load_data(model_name)
    if data is None:
        print(f"No data file found for model '{model_name}' ({_data_path(model_name)})")
        return

    # Also try loading retry state for richer output
    retry_state = None
    retry_path = f"data/{model_name.replace(':', '_')}_retry_state.json"
    if os.path.exists(retry_path):
        with open(retry_path, "r", encoding="utf-8") as f:
            retry_state = json.load(f)

    print_status(model_name, data, retry_state=retry_state)


# =============================================================================
# Mode: complete-pending
# =============================================================================

def mode_complete_pending(model_name: str, model_info: RunnableModel):
    """Complete any pending batch jobs for this model's provider."""
    provider = model_info["provider"]

    if provider == "ANTHROPIC":
        import anthropic
        import generate_llm_answers_batch_anthropic as gen

        api_key = get_provider_api_key(provider)
        client = anthropic.Anthropic(api_key=api_key)
        completed = gen.check_and_complete_pending_batches(client, model_info["name"])
        if completed:
            print(f"[DONE] Completed {len(completed)} pending Anthropic batch(es)")
        else:
            print(f"[OK] No pending Anthropic batches for {model_name}")

    elif provider == "OPENAI":
        import openai
        import generate_llm_answers_batch_openai as gen

        api_key = get_provider_api_key(provider)
        client = openai.OpenAI(api_key=api_key)
        completed = gen.check_and_complete_pending_batches(client, model_info["name"])
        if completed:
            print(f"[DONE] Completed {len(completed)} pending OpenAI batch(es)")
        else:
            print(f"[OK] No pending OpenAI batches for {model_name}")

    else:
        print(f"[SKIP] Provider '{provider}' does not use batch processing.")


# =============================================================================
# Mode: first  (generate initial Round 0 responses)
# =============================================================================

def mode_first(model_name: str, model_info: RunnableModel):
    """Generate initial (Round 0) assistant responses. Non-destructive."""
    provider = model_info["provider"]
    save_file = _data_path(model_name)
    save_file_batch = f"data/{model_name.replace(':', '_')}_batch.jsonl"
    save_file_batch_response = save_file_batch[:-6] + "_response.jsonl"

    if provider == "ANTHROPIC":
        import anthropic
        import generate_llm_answers_batch_anthropic as gen

        api_key = get_provider_api_key(provider)
        client = anthropic.Anthropic(api_key=api_key)

        batch_id = gen.generate_first_response_batch(
            client, model_info, save_file_batch, save_file
        )
        if batch_id is None:
            # All first responses already exist
            return
        gen.wait_for_batch_and_save(
            client, batch_id, save_file, save_file_batch, save_file_batch_response
        )

    elif provider == "OPENAI":
        import openai
        import generate_llm_answers_batch_openai as gen

        api_key = get_provider_api_key(provider)
        client = openai.OpenAI(api_key=api_key)

        batch_id = gen.generate_first_response_batch(
            client, model_info, save_file_batch, save_file
        )
        if batch_id is None:
            return
        gen.wait_for_batch_and_save(
            client, batch_id, save_file, save_file_batch, save_file_batch_response
        )

    elif provider == "OPENAI_DIRECT":
        # Non-batch OpenAI (e.g. gpt-5.2-codex) – direct async API calls
        import openai
        import generate_llm_answers_openai_direct as gen

        api_key = get_provider_api_key(provider)
        llm_client = openai.AsyncOpenAI(api_key=api_key)
        asyncio.run(gen.generate_first_response(llm_client, model_info))

    elif provider == "SAP_AICORE":
        # SAP ABAP-1 is available through SAP AI Core orchestration, not batch APIs.
        import generate_llm_answers_parallel as gen
        from abap1_orchestration import ABAP1OrchestrationClient

        llm_client = ABAP1OrchestrationClient.from_env(
            model_name=model_info["name"],
            temperature=model_info["temperature"],
            max_tokens=model_info["max_tokens"],
        )
        asyncio.run(gen.generate_first_response(llm_client, model_info))

    else:
        # OpenAI-compatible provider (Groq, Mistral, etc.) – uses async parallel
        import generate_llm_answers_parallel as gen

        base_url = API_PROVIDERS[provider].get("base_url")
        api_key = get_provider_api_key(provider)
        llm_client = __import__("openai").AsyncOpenAI(
            base_url=base_url, api_key=api_key
        )
        asyncio.run(gen.generate_first_response(llm_client, model_info))

    print(f"[DONE] First-round generation complete for {model_name}")


# =============================================================================
# Mode: next  (generate one correction round)
# =============================================================================

def mode_next(model_name: str, model_info: RunnableModel):
    """
    Generate exactly one correction round for WaitingForLLM conversations.
    
    Refuses to run if any chats still NeedSAPTest (the batch generation
    functions also enforce this guard internally).
    """
    data = _load_data(model_name)
    if data is None:
        print(f"Error: data file not found: {_data_path(model_name)}")
        sys.exit(1)

    # Pre-flight check: how many conversations would actually be generated?
    states = count_states(data)
    if states["waiting_llm"] == 0:
        print(
            f"[SKIP] No conversations waiting for LLM response.\n"
            f"  Success: {states['success']}, Needs SAP test: {states['needs_test']}, "
            f"Infra retriable: {states['infra_retriable']}, "
            f"Maxed out: {states['maxed_out']}, "
            f"Deterministic fail: {states['deterministic_fail']}"
        )
        return

    if states["needs_test"] > 0:
        print(
            f"[BLOCKED] {states['needs_test']} conversation(s) still need SAP testing.\n"
            f"  Run SAP tests first:  python src/abap_test.py --model {model_name} --mode resume\n"
            f"  Then retry:           python src/abap_test.py --model {model_name} --mode retry --max-attempts 3"
        )
        return

    print(
        f"[INFO] Generating next round for {states['waiting_llm']} conversation(s) "
        f"(success: {states['success']}, infra: {states['infra_retriable']}, "
        f"maxed: {states['maxed_out']})"
    )

    provider = model_info["provider"]
    save_file = _data_path(model_name)
    save_file_batch = f"data/{model_name.replace(':', '_')}_batch.jsonl"
    save_file_batch_response = save_file_batch[:-6] + "_response.jsonl"

    # Determine round number for logging (approximate – not all chats may be at the same round)
    from chat_state import assistant_count as _ac, chat_waiting_for_llm as _wl
    max_round = 0
    for chats in data.values():
        for c in chats:
            if _wl(c):
                r = _ac(c)  # next assistant would be this number
                if r > max_round:
                    max_round = r
    round_num = max_round  # the round about to be generated

    if provider == "ANTHROPIC":
        import anthropic
        import generate_llm_answers_batch_anthropic as gen

        api_key = get_provider_api_key(provider)
        client = anthropic.Anthropic(api_key=api_key)

        batch_id = gen.generate_next_response_batch(
            client, model_info, save_file, save_file_batch, round_num=round_num
        )
        if batch_id is None:
            return
        gen.wait_for_batch_and_save(
            client, batch_id, save_file, save_file_batch, save_file_batch_response
        )

    elif provider == "OPENAI":
        import openai
        import generate_llm_answers_batch_openai as gen

        api_key = get_provider_api_key(provider)
        client = openai.OpenAI(api_key=api_key)

        batch_id = gen.generate_next_response_batch(
            client, model_info, save_file, save_file_batch, round_num=round_num
        )
        if batch_id is None:
            return
        gen.wait_for_batch_and_save(
            client, batch_id, save_file, save_file_batch, save_file_batch_response
        )

    elif provider == "OPENAI_DIRECT":
        # Non-batch OpenAI (e.g. gpt-5.2-codex) – direct async API calls
        import openai
        import generate_llm_answers_openai_direct as gen

        api_key = get_provider_api_key(provider)
        llm_client = openai.AsyncOpenAI(api_key=api_key)
        asyncio.run(
            gen.generate_next_response(llm_client, model_info, round_num)
        )

    elif provider == "SAP_AICORE":
        import generate_llm_answers_parallel as gen
        from abap1_orchestration import ABAP1OrchestrationClient

        llm_client = ABAP1OrchestrationClient.from_env(
            model_name=model_info["name"],
            temperature=model_info["temperature"],
            max_tokens=model_info["max_tokens"],
        )
        asyncio.run(
            gen.generate_next_response(llm_client, model_info, round_num)
        )

    else:
        # OpenAI-compatible provider – uses async parallel
        import generate_llm_answers_parallel as gen

        base_url = API_PROVIDERS[provider].get("base_url")
        api_key = get_provider_api_key(provider)
        llm_client = __import__("openai").AsyncOpenAI(
            base_url=base_url, api_key=api_key
        )
        asyncio.run(
            gen.generate_next_response(llm_client, model_info, round_num)
        )

    print(f"[DONE] Next-round generation complete for {model_name} (round ~{round_num})")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="LLM generation CLI: generate/resume model responses for ABAP benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model", "-m", required=True,
        help="Model name (e.g. claude-opus-4-5-20251101). Partial match supported.",
    )
    parser.add_argument(
        "--mode", required=True,
        choices=["status", "complete-pending", "first", "next"],
        help=(
            "status = print state counts; "
            "complete-pending = complete pending batch jobs; "
            "first = generate initial responses (Round 0); "
            "next = generate one correction round"
        ),
    )
    parser.add_argument(
        "--list", "-l", action="store_true",
        help="List all available models and exit",
    )

    args = parser.parse_args()

    if args.list:
        print("\nAvailable models:")
        for m in MODELS_TO_RUN:
            print(f"  - {m['name']} ({m['provider']})")
        return

    # Resolve model
    model_info = _get_model_info(args.model)
    if model_info is None:
        print(f"Error: Model '{args.model}' not found.")
        print("\nAvailable models:")
        for m in MODELS_TO_RUN:
            print(f"  - {m['name']}")
        sys.exit(1)

    model_name = model_info["name"]
    print(f"Model: {model_name} ({model_info['provider']})")

    if args.mode == "status":
        mode_status(model_name)
    elif args.mode == "complete-pending":
        mode_complete_pending(model_name, model_info)
    elif args.mode == "first":
        mode_first(model_name, model_info)
    elif args.mode == "next":
        mode_next(model_name, model_info)


if __name__ == "__main__":
    main()
