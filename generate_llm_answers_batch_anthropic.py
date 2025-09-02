import json
import os
import time
from typing import Dict, List
import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from anthropic.types import TextBlock

from llms import RunnableModel
from generate_llm_answers import (
    SYSTEM_PROMPT,
    REPETITIONS,
    PROMPT_FILES,
    remove_code_blocks,
)


def _build_single_batch_request(
    model_info: RunnableModel, custom_id: str, conversation: List
) -> Request:

    conversation_request = Request(
        custom_id=custom_id,
        params=MessageCreateParamsNonStreaming(
            model=model_info["name"],
            max_tokens=model_info["max_tokens"],
            temperature=model_info["temperature"],
            system=conversation[0]["content"],
            messages=conversation[1:],
        ),
    )
    return conversation_request


def _create_batch(client: anthropic.Anthropic, requests: List[Request]):
    batch_info = client.messages.batches.create(requests=requests)
    return batch_info


def _status_batch(client: anthropic.Anthropic, batch_id: str):
    batch = client.messages.batches.retrieve(batch_id)
    return batch


def _retrieve_batch(client: anthropic.Anthropic, batch_id: str):
    responses = [m for m in client.messages.batches.results(batch_id)]
    return responses


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
                prompt = id[1 : id.rindex("-R")] + ".txt"
                repetition = int(id[id.rindex("-R") + 2 :])

                system_prompt = {"role": "system", "content": batch_entry["params"]["system"]}
                
                messages = [system_prompt] + batch_entry["params"]["messages"]
                new_message = batch_response["response"]
                new_message = {
                    "role": "assistant",
                    "content": remove_code_blocks(new_message),
                }
                messages.append(new_message)
                result[prompt][repetition] = messages
                break

    with open(save_file, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=4, ensure_ascii=False)


def generate_next_response_batch(
    client: anthropic.Anthropic,
    model_info: RunnableModel,
    save_file: str,
    save_file_batch: str,
):
    conversations_requests: List[Request] = []

    current = {}
    with open(save_file, "r", encoding="utf-8") as file:
        current = json.loads(file.read())

    for prompt_file in current:
        for i, repetition in enumerate(current[prompt_file]):
            custom_id = f"P{prompt_file[:-4]}-R{i}"
            conversation = current[prompt_file][i]
            if conversation[-1]["content"] == "The unit tests were successful.":
                continue
            conversation_request = _build_single_batch_request(
                model_info, custom_id, conversation
            )
            conversations_requests.append(conversation_request)

    with open(save_file_batch, "w", encoding="utf-8") as file:
        for line in conversations_requests:
            file.write(json.dumps(line, ensure_ascii=False) + "\n")

    batch = _create_batch(client, conversations_requests)
    return batch.id


def generate_first_response_batch(
    client, model_info: RunnableModel, save_file_batch: str
) -> str:
    conversations: List[Request] = []
    for prompt_file in PROMPT_FILES:
        with open(f"prompts/{prompt_file}", "r", encoding="utf-8") as file:
            prompt_content = file.read()
        for i in range(REPETITIONS):
            conversation: List = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_content},
            ]
            custom_id = f"P{prompt_file[:-4]}-R{i}"
            conversation_request = _build_single_batch_request(
                model_info, custom_id, conversation
            )
            conversations.append(conversation_request)

    with open(save_file_batch, "w", encoding="utf-8") as file:
        for line in conversations:
            file.write(json.dumps(line, ensure_ascii=False) + "\n")

    batch = _create_batch(client, conversations)
    return batch.id


def wait_for_batch_and_save(
    client: anthropic.Anthropic,
    batch_id: str,
    save_file: str,
    save_file_batch: str,
    save_file_batch_response: str,
):
    while True:
        status = _status_batch(client, batch_id)
        # print(status)
        if status.processing_status == "ended":
            break
        time.sleep(30)
    results_url = status.results_url
    if results_url:
        content = _retrieve_batch(client, batch_id)
        with open(save_file_batch_response, "w", encoding="utf-8") as file:
            
            failed = 0
            successes = []
            for entry in content:
                if entry.result.type == "succeeded":
                    successful_entry = {
                        "custom_id": entry.custom_id,
                        "response": " ".join(
                            remove_code_blocks(c.text)
                            for c in entry.result.message.content
                            if isinstance(c, TextBlock)
                        ),
                    }
                    successes.append(successful_entry)
                    file.write(json.dumps(successful_entry, ensure_ascii=False) + "\n")
                if entry.result.type != "succeeded":
                    failed += 1
                    print(f"request {entry.custom_id} failed")
        if failed > 0: print(f"{failed} requests failed")
        _convert_to_json(save_file, save_file_batch, save_file_batch_response)
