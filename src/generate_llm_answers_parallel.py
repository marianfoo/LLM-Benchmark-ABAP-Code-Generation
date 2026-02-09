import asyncio
import json
import os
from typing import Any, Dict, Iterable, List
from openai.types.chat import ChatCompletionMessageParam
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

# Maximum number of concurrent API requests.  Requests are throttled via an
# asyncio.Semaphore so that up to this many are in-flight at any time.
# Increase to saturate the API; decrease if you hit rate-limit errors.
MAX_CONCURRENT_REQUESTS = 9

# Save progress every N completed API requests
SAVE_INTERVAL = 20


async def ask_provider(
    client: Any,
    model_info: RunnableModel,
    chat_history: Iterable[ChatCompletionMessageParam],
):

    max_retries = 5
    messages = list(chat_history)

    for attempt in range(max_retries):
        try:
            if model_info["provider"] == "SAP_AICORE":
                text_response = await client.complete(messages)
                text_response = remove_code_blocks(text_response)
                return text_response

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
                # Exponential backoff: 5s, 10s, 20s, 40s, 80s (capped at 120s)
                retry_delay = min(5 * (2 ** attempt), 120)
                tqdm.tqdm.write(f"retrying in {retry_delay}s... ({e!r})")
                await asyncio.sleep(retry_delay)


def read_file_or_create(save_file_path: str):
    if os.path.exists(save_file_path):
        with open(save_file_path, "r", encoding="utf-8") as file:
            conversations = json.load(file)
    else:
        conversations = {}
    return conversations


# =============================================================================
# First-response generation (semaphore-based concurrency)
# =============================================================================

async def _first_response_single(
    llm_client: Any,
    model_info: RunnableModel,
    prompt_file: str,
    rep_index: int,
    conversations: Dict[str, List],
    sem: asyncio.Semaphore,
    pbar: tqdm.tqdm,
    counter: dict,
    save_file_path: str,
    save_lock: asyncio.Lock,
):
    """Process a single (prompt_file, repetition) pair, throttled by *sem*."""
    async with sem:
        if prompt_file not in conversations:
            conversations[prompt_file] = []

        # Skip if already done
        if len(conversations[prompt_file]) > rep_index:
            pbar.update(1)
            pbar.refresh()
            return

        with open(f"dataset/prompts/{prompt_file}", "r", encoding="utf-8") as file:
            prompt_content = file.read()

        conversation: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_content},
        ]
        response = await ask_provider(llm_client, model_info, conversation)
        conversation.append({"role": "assistant", "content": response})

        # Append in order – we hold the event loop so this is safe for
        # the same prompt_file as long as we don't yield between check & append.
        while len(conversations[prompt_file]) < rep_index:
            # Another task for an earlier rep hasn't finished yet; yield briefly.
            await asyncio.sleep(0.05)
        conversations[prompt_file].append(conversation)

        pbar.update(1)
        pbar.refresh()

        counter["done"] += 1
        if counter["done"] % SAVE_INTERVAL == 0:
            await _save_conversations(conversations, save_file_path, save_lock)


async def generate_first_response(
    llm_client: Any, model_info: RunnableModel
):
    save_file_path = f"data/{model_info['name'].replace(':', '_')}.json"
    conversations: Dict[str, List] = read_file_or_create(save_file_path)

    total_requests = len(PROMPT_FILES) * REPETITIONS

    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    save_lock = asyncio.Lock()
    counter: dict = {"done": 0}

    with tqdm.tqdm(total=total_requests, desc="Total Progress") as pbar:
        tasks = []
        for prompt_file in PROMPT_FILES:
            for i in range(REPETITIONS):
                tasks.append(
                    asyncio.create_task(
                        _first_response_single(
                            llm_client,
                            model_info,
                            prompt_file,
                            i,
                            conversations,
                            sem,
                            pbar,
                            counter,
                            save_file_path,
                            save_lock,
                        )
                    )
                )
        await asyncio.gather(*tasks)

    # Final save
    await _save_conversations(conversations, save_file_path, save_lock)

    with open(save_file_path, "w", encoding="utf-8") as f:
        json.dump(conversations, f, ensure_ascii=False, indent=4)


# =============================================================================
# Next-round generation (semaphore-based concurrency)
# =============================================================================

async def _next_response_single(
    llm_client: Any,
    model_info: RunnableModel,
    prompt_file: str,
    rep_index: int,
    conversation: List,
    response_number: int,
    sem: asyncio.Semaphore,
    pbar: tqdm.tqdm,
    counter: dict,
    conversations: Dict[str, List],
    save_file_path: str,
    save_lock: asyncio.Lock,
):
    """Process a single conversation for the next correction round."""
    async with sem:
        # Skip if already has enough responses, is successful,
        # or is not in WaitingForLLM state (e.g. NeedsSAPTest, InfraRetriable).
        # Note: response_number is the round being generated (1-indexed from
        # llm_generate.py), which equals the current assistant_count for chats
        # that are waiting.  Use `>` so we don't skip conversations that are
        # exactly at that count and need the next response.
        skip = False
        if chat_assistant_count(conversation) > response_number:
            skip = True
        elif chat_is_success(conversation):
            skip = True
        elif not chat_waiting_for_llm(conversation):
            # Not waiting for LLM – e.g. NeedsSAPTest or InfraRetriable
            skip = True
        elif chat_assistant_count(conversation) >= MAX_ASSISTANTS:
            # Already at Round 5 (6 assistant messages) – no more rounds
            skip = True

        if skip:
            pbar.update(1)
            pbar.refresh()
            return

        response = await ask_provider(llm_client, model_info, conversation)
        conversation.append({"role": "assistant", "content": response})

        pbar.update(1)
        pbar.refresh()

        counter["done"] += 1
        if counter["done"] % SAVE_INTERVAL == 0:
            await _save_conversations(conversations, save_file_path, save_lock)


async def generate_next_response(
    llm_client: Any,
    model_info: RunnableModel,
    response_number: int,
):
    save_file_path = f"data/{model_info['name'].replace(':', '_')}.json"
    with open(save_file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

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

    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    save_lock = asyncio.Lock()
    counter: dict = {"done": 0}

    with tqdm.tqdm(total=total_conversations, desc="Total Progress") as pbar:
        tasks = []
        for prompt_file, repetitions in data.items():
            for rep_idx, conversation in enumerate(repetitions):
                tasks.append(
                    asyncio.create_task(
                        _next_response_single(
                            llm_client,
                            model_info,
                            prompt_file,
                            rep_idx,
                            conversation,
                            response_number,
                            sem,
                            pbar,
                            counter,
                            data,
                            save_file_path,
                            save_lock,
                        )
                    )
                )
        await asyncio.gather(*tasks)

    # Final save
    with open(save_file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# =============================================================================
# Shared helpers
# =============================================================================

async def _save_conversations(
    conversations: Dict[str, List],
    save_file_path: str,
    save_lock: asyncio.Lock,
):
    """Atomically save all conversations to disk."""
    async with save_lock:
        try:
            with open(save_file_path, "w", encoding="utf-8") as f:
                json.dump(conversations, f, ensure_ascii=False, indent=4)
            tqdm.tqdm.write("Progress saved")
        except Exception as e:
            tqdm.tqdm.write(f"Error saving progress: {e}")
