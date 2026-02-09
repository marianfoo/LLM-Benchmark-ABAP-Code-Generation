"""
Non-batch OpenAI generator for models that don't support the Batch API
(e.g. gpt-5.2-codex).

Uses async parallel requests (like generate_llm_answers_parallel.py) but
with GPT-5-family parameter handling:
  - ``max_completion_tokens`` instead of ``max_tokens``
  - temperature is NOT passed (GPT-5 only supports temperature=1)

Writes the same ``data/<model>.json`` conversation format so that
``abap_test.py``, ``parallel_runner.py``, and ``consolidate_results.py``
work unchanged.
"""

import asyncio
import json
import os
from typing import Any, Dict, Iterable, List

from more_itertools import divide
from openai.types.chat import ChatCompletionMessageParam
import openai
import tqdm

from llms import RunnableModel
from generate_llm_answers import (
    remove_code_blocks,
    REPETITIONS,
    SYSTEM_PROMPT,
    PROMPT_FILES,
)
from chat_state import (
    chat_is_success,
    chat_needs_test,
    chat_waiting_for_llm,
    assistant_count as chat_assistant_count,
    any_needs_test,
    MAX_ASSISTANTS,
)

# How many concurrent async tasks to run (keeps API rate manageable)
NUM_PARALLEL_LLM_REQUESTS = 3


# =============================================================================
# LLM request helper
# =============================================================================

async def ask_openai_direct(
    client: openai.AsyncOpenAI,
    model_info: RunnableModel,
    chat_history: Iterable[ChatCompletionMessageParam],
) -> str:
    """
    Send a chat completion request with GPT-5-aware parameter handling.

    - GPT-5 family: uses ``max_completion_tokens``, omits ``temperature``
    - Other models: uses ``max_tokens`` and ``temperature``
    """
    max_retries = 5
    retry_delay = 60
    messages = list(chat_history)

    for attempt in range(max_retries):
        try:
            # GPT-5 family requires max_completion_tokens and does not accept
            # a custom temperature (always 1).
            if "gpt-5" in model_info["name"]:
                response = await client.chat.completions.create(
                    model=model_info["name"],
                    messages=messages,
                    max_completion_tokens=model_info["max_tokens"],
                )
            else:
                response = await client.chat.completions.create(
                    model=model_info["name"],
                    messages=messages,
                    temperature=model_info["temperature"],
                    max_tokens=model_info["max_tokens"],
                )

            text_response = response.choices[0].message.content or ""
            text_response = remove_code_blocks(text_response)
            return text_response

        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            else:
                tqdm.tqdm.write(f"[retry {attempt + 1}/{max_retries}] {e}")
                await asyncio.sleep(retry_delay)


# =============================================================================
# Helpers
# =============================================================================

def _data_path(model_info: RunnableModel) -> str:
    return f"data/{model_info['name'].replace(':', '_')}.json"


def _read_or_create(path: str) -> Dict[str, List]:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_json(path: str, data: dict):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    os.replace(tmp, path)  # atomic on POSIX


# =============================================================================
# Async worker: first-round (Round 0) generation
# =============================================================================

async def _first_response_worker(
    client: openai.AsyncOpenAI,
    model_info: RunnableModel,
    prompt_list: list[str],
    conversations: Dict[str, List],
    main_pbar: tqdm.tqdm,
    thread_id: int,
    save_file_path: str,
    save_lock: asyncio.Lock,
):
    """Generate initial assistant responses for a subset of prompts."""
    total_requests = len(prompt_list) * REPETITIONS

    with tqdm.tqdm(
        total=total_requests,
        desc=f"Thread {thread_id}",
        position=thread_id,
        leave=False,
    ) as thread_pbar:
        processed_count = 0
        for prompt_file in prompt_list:
            with open(f"dataset/prompts/{prompt_file}", "r", encoding="utf-8") as f:
                prompt_content = f.read()

            for i in range(REPETITIONS):
                if prompt_file not in conversations:
                    conversations[prompt_file] = []

                # Non-destructive: skip if this repetition already has data
                if len(conversations[prompt_file]) > i:
                    main_pbar.update(1)
                    thread_pbar.update(1)
                    main_pbar.refresh()
                    thread_pbar.refresh()
                    continue

                conversation: List[ChatCompletionMessageParam] = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_content},
                ]
                response = await ask_openai_direct(client, model_info, conversation)
                conversation.append({"role": "assistant", "content": response})
                conversations[prompt_file].append(conversation)

                main_pbar.update(1)
                thread_pbar.update(1)
                main_pbar.refresh()
                thread_pbar.refresh()
                processed_count += 1

                # Periodically save progress
                if processed_count % 10 == 0:
                    await _save_progress(conversations, save_file_path, save_lock, thread_id)

    # Final save for this thread
    await _save_progress(conversations, save_file_path, save_lock, thread_id)
    return conversations


# =============================================================================
# Async worker: next-round (correction) generation
# =============================================================================

async def _next_response_worker(
    client: openai.AsyncOpenAI,
    model_info: RunnableModel,
    conversations: Dict[str, List],
    response_number: int,
    main_pbar: tqdm.tqdm,
    thread_id: int,
    save_file_path: str,
    save_lock: asyncio.Lock,
):
    """Generate one correction round for a subset of conversations."""
    total_in_subset = sum(
        len(reps) for reps in conversations.values()
    )

    with tqdm.tqdm(
        total=total_in_subset,
        desc=f"Thread {thread_id}",
        position=thread_id,
        leave=False,
    ) as thread_pbar:
        processed_count = 0
        for prompt_file, repetitions in conversations.items():
            for conversation in repetitions:
                # Determine whether this conversation should be skipped
                skip = False
                if chat_assistant_count(conversation) >= response_number:
                    skip = True
                elif chat_is_success(conversation):
                    skip = True
                elif not chat_waiting_for_llm(conversation):
                    # Covers: NeedsSAPTest, InfraRetriable, empty, etc.
                    skip = True
                elif chat_assistant_count(conversation) >= MAX_ASSISTANTS:
                    # Already at Round 5 (6 assistant messages) – no more rounds
                    skip = True

                if skip:
                    main_pbar.update(1)
                    thread_pbar.update(1)
                    main_pbar.refresh()
                    thread_pbar.refresh()
                    continue

                response = await ask_openai_direct(client, model_info, conversation)
                conversation.append({"role": "assistant", "content": response})

                main_pbar.update(1)
                thread_pbar.update(1)
                main_pbar.refresh()
                thread_pbar.refresh()
                processed_count += 1

                # Periodically save progress
                if processed_count % 10 == 0:
                    await _save_progress(conversations, save_file_path, save_lock, thread_id)

    # Final save for this thread
    await _save_progress(conversations, save_file_path, save_lock, thread_id)
    return conversations


# =============================================================================
# Progress saving (shared lock)
# =============================================================================

async def _save_progress(
    conversations: Dict[str, List],
    save_file_path: str,
    save_lock: asyncio.Lock,
    thread_id: int,
):
    """Atomically merge thread-local conversations into the shared JSON file."""
    async with save_lock:
        try:
            current_data = _read_or_create(save_file_path)
            current_data.update(conversations)
            _save_json(save_file_path, current_data)
            tqdm.tqdm.write(f"Thread {thread_id}: Progress saved")
        except Exception as e:
            tqdm.tqdm.write(f"Thread {thread_id}: Error saving progress: {e}")


# =============================================================================
# Public entry-points (called from llm_generate.py / main.py)
# =============================================================================

async def generate_first_response(
    client: openai.AsyncOpenAI,
    model_info: RunnableModel,
):
    """
    Generate initial (Round 0) assistant responses for all prompts.

    Non-destructive: skips prompt+repetition pairs that already have an
    assistant response in the data file.
    """
    save_file_path = _data_path(model_info)
    conversations: Dict[str, List] = _read_or_create(save_file_path)

    total_requests = len(PROMPT_FILES) * REPETITIONS

    split_prompts = [
        list(part) for part in divide(NUM_PARALLEL_LLM_REQUESTS, PROMPT_FILES)
    ]

    save_lock = asyncio.Lock()
    with tqdm.tqdm(
        total=total_requests, desc="Total Progress", position=0
    ) as main_pbar:
        tasks = [
            asyncio.create_task(
                _first_response_worker(
                    client,
                    model_info,
                    sublist,
                    conversations,
                    main_pbar,
                    i + 1,
                    save_file_path,
                    save_lock,
                )
            )
            for i, sublist in enumerate(split_prompts)
        ]
        results: List[Dict[str, List]] = await asyncio.gather(*tasks)

    # Final merge across all threads
    merged: Dict[str, List] = {}
    for res in results:
        merged.update(res)
    _save_json(save_file_path, merged)


async def generate_next_response(
    client: openai.AsyncOpenAI,
    model_info: RunnableModel,
    response_number: int,
):
    """
    Generate exactly one correction round for all WaitingForLLM conversations.

    Guards:
    - Blocks if any conversations still NeedsSAPTest.
    - Skips conversations that are Success, InfraRetriable, or MaxedOut.
    """
    save_file_path = _data_path(model_info)
    with open(save_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ── GUARD: refuse to generate next round if any chats still need SAP testing ──
    if any_needs_test(data):
        needs = sum(
            1 for chats in data.values()
            for c in chats if chat_needs_test(c)
        )
        print(
            f"[BLOCKED] {needs} conversation(s) still need SAP testing "
            f"(last message is assistant without feedback).\n"
            f"  Run SAP tests first:  python src/abap_test.py --model {model_info['name']} --mode resume\n"
            f"  Then retry:           python src/abap_test.py --model {model_info['name']} --mode retry --max-attempts 3"
        )
        return

    total_conversations = sum(
        len(reps) for reps in data.values()
    )

    prompt_files = list(data.keys())
    split_prompts = [
        list(part) for part in divide(NUM_PARALLEL_LLM_REQUESTS, prompt_files)
    ]
    subsets = [{key: data[key] for key in sublist} for sublist in split_prompts]

    save_lock = asyncio.Lock()
    with tqdm.tqdm(
        total=total_conversations, desc="Total Progress", position=0
    ) as main_pbar:
        tasks = [
            _next_response_worker(
                client,
                model_info,
                subset,
                response_number,
                main_pbar,
                i + 1,
                save_file_path,
                save_lock,
            )
            for i, subset in enumerate(subsets)
        ]
        results: List[Dict[str, List]] = await asyncio.gather(*tasks)

    # Final merge across all threads
    merged: Dict[str, List] = {}
    for res in results:
        merged.update(res)
    _save_json(save_file_path, merged)
