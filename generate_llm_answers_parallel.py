import asyncio
import json
import os
from typing import Dict, Iterable, List
from more_itertools import divide
from openai.types.chat import ChatCompletionMessageParam
import openai
import tqdm

from llms import RunnableModel
from generate_llm_answers import (
    remove_code_blocks,
    amount_of_llm_responses,
    REPETITIONS,
    SYSTEM_PROMPT,
    PROMPT_FILES,
)

NUM_PARALLEL_LLM_REQUESTS = 3


async def ask_provider(
    client: openai.AsyncOpenAI,
    model_info: RunnableModel,
    chat_history: Iterable[ChatCompletionMessageParam],
):

    max_retries = 5
    retry_delay = 60

    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=model_info["name"],
                messages=chat_history,
                temperature=model_info["temperature"],
                max_tokens=model_info["max_tokens"],
            )
            text_response = response.choices[0].message.content
            text_response = remove_code_blocks(text_response)
            return text_response

        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            else:
                tqdm.tqdm.write("retrying...")
                await asyncio.sleep(retry_delay)


def amount_of_llm_responses(conversation: dict):
    return sum(1 for message in conversation if message["role"] == "assistant")


def read_file_or_create(save_file_path: str):
    if os.path.exists(save_file_path):
        with open(save_file_path, "r", encoding="utf-8") as file:
            conversations = json.load(file)
    else:
        conversations = {}
    return conversations


async def _first_response_requests(
    llm_client: openai.AsyncOpenAI,
    model_info: RunnableModel,
    prompt_list: list[str],
    conversations: Dict[str, List],
    main_pbar: tqdm.tqdm,
    thread_id: int,
    save_file_path: str,
    save_lock: asyncio.Lock,
):
    total_requests = len(prompt_list) * REPETITIONS

    with tqdm.tqdm(
        total=total_requests,
        desc=f"Thread {thread_id}",
        position=thread_id,
        leave=False,
    ) as thread_pbar:
        processed_count = 0
        for prompt_file in prompt_list:
            with open(f"prompts/{prompt_file}", "r", encoding="utf-8") as file:
                prompt_content = file.read()
            for i in range(REPETITIONS):
                if prompt_file not in conversations:
                    conversations[prompt_file] = []
                if len(conversations[prompt_file]) > i:
                    # print(f"skipping {prompt_file} repetiontion {i} because it was already done")
                    main_pbar.update(1)
                    thread_pbar.update(1)
                    main_pbar.refresh()
                    thread_pbar.refresh()
                elif len(conversations[prompt_file]) <= i:
                    conversation: List[ChatCompletionMessageParam] = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt_content},
                    ]
                    response = await ask_provider(llm_client, model_info, conversation)
                    conversation.append({"role": "assistant", "content": response})
                    conversations[prompt_file].append(conversation)
                    main_pbar.update(1)
                    thread_pbar.update(1)
                    main_pbar.refresh()
                    thread_pbar.refresh()
                    processed_count += 1

                    # Save progress
                    if processed_count % 10 == 0:
                        await save_progress_first(
                            conversations, save_file_path, save_lock, thread_id
                        )

    # Final save for this thread
    await save_progress_first(conversations, save_file_path, save_lock, thread_id)
    return conversations


async def generate_first_response(
    llm_client: openai.AsyncOpenAI, model_info: RunnableModel
):
    save_file_path = f"{model_info["name"].replace(':', '_')}.json"
    conversations: Dict[str, List] = read_file_or_create(save_file_path)

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
                _first_response_requests(
                    llm_client,
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

    merged = {}
    for res in results:
        merged.update(res)
    with open(save_file_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=4)


async def save_progress_first(
    conversations: Dict[str, List],
    save_file_path: str,
    save_lock: asyncio.Lock,
    thread_id: int,
):
    async with save_lock:
        try:
            current_data = read_file_or_create(save_file_path)
            current_data.update(conversations)

            with open(save_file_path, "w", encoding="utf-8") as f:
                json.dump(current_data, f, ensure_ascii=False, indent=4)

            tqdm.tqdm.write(f"Thread {thread_id}: Progress saved")
        except Exception as e:
            tqdm.tqdm.write(f"Thread {thread_id}: Error saving progress: {e}")


async def next_response_request(
    llm_client: openai.AsyncOpenAI,
    model_info: RunnableModel,
    conversations: Dict[str, List],
    response_number: int,
    main_pbar: tqdm.tqdm,
    thread_id: int,
    save_file_path: str,
    save_lock: asyncio.Lock,
):
    total_in_subset = 0
    for prompt_file in conversations:
        repetitions = conversations[prompt_file]
        for conversation in repetitions:
            total_in_subset += 1

    with tqdm.tqdm(
        total=total_in_subset,
        desc=f"Thread {thread_id}",
        position=thread_id,
        leave=False,
    ) as thread_pbar:
        processed_count = 0
        for prompt_file in conversations:
            repetitions = conversations[prompt_file]
            for conversation in repetitions:
                if (
                    amount_of_llm_responses(conversation) >= response_number
                    or conversation[-1]["content"] == "The unit tests were successful."
                ):
                    main_pbar.update(1)
                    thread_pbar.update(1)
                    main_pbar.refresh()
                    thread_pbar.refresh()
                    continue
                response = await ask_provider(llm_client, model_info, conversation)
                conversation.append({"role": "assistant", "content": response})
                main_pbar.update(1)
                thread_pbar.update(1)
                main_pbar.refresh()
                thread_pbar.refresh()
                processed_count += 1

                # Save progress every 10 processed conversations
                if processed_count % 10 == 0:
                    await save_progress(
                        conversations, save_file_path, save_lock, thread_id
                    )

    # Final save for this thread
    await save_progress(conversations, save_file_path, save_lock, thread_id)
    return conversations


async def save_progress(
    conversations: Dict[str, List],
    save_file_path: str,
    save_lock: asyncio.Lock,
    thread_id: int,
):
    async with save_lock:
        try:
            with open(save_file_path, "r", encoding="utf-8") as file:
                current_data = json.load(file)
            current_data.update(conversations)
            with open(save_file_path, "w", encoding="utf-8") as f:
                json.dump(current_data, f, ensure_ascii=False, indent=4)

            tqdm.tqdm.write(f"Thread {thread_id}: Progress saved")
        except Exception as e:
            tqdm.tqdm.write(f"Thread {thread_id}: Error saving progress: {e}")


async def generate_next_response(
    llm_client: openai.AsyncOpenAI,
    model_info: RunnableModel,
    response_number: int,
):
    save_file_path = f"{model_info["name"].replace(':', '_')}.json"
    with open(save_file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    total_conversations = 0
    for prompt_file in data:
        repetitions = data[prompt_file]
        for conversation in repetitions:
            total_conversations += 1

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
            next_response_request(
                llm_client,
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

    merged = {}
    for res in results:
        merged.update(res)
    with open(save_file_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=4)
