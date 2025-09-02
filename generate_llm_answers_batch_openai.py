import json
import os
import time
from typing import Dict, List
from openai.types.chat import ChatCompletionMessageParam
import openai

from llms import RunnableModel
from generate_llm_answers import (
    SYSTEM_PROMPT,
    REPETITIONS,
    PROMPT_FILES,
    remove_code_blocks,
)


def _build_single_batch_request(
    model_info: RunnableModel, custom_id: str, conversation: List
):
    conversation_request = {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model_info["name"],
            "messages": conversation,
            "temperature": model_info["temperature"],
            "max_completion_tokens": model_info["max_tokens"],
        },
    }
    return conversation_request


def _upload_batch_input_file(client: openai.OpenAI, file_name: str):
    batch_input_file = client.files.create(file=open(file_name, "rb"), purpose="batch")
    return batch_input_file


def _create_batch(client: openai.OpenAI, batch_input_file_id: str):
    batch_info = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "first gpt 5 test run"},
    )
    return batch_info


def _status_batch(client: openai.OpenAI, batch_id: str):
    batch = client.batches.retrieve(batch_id)
    return batch


def _retrieve_batch(client: openai.OpenAI, batch_id: str):
    file_response = client.files.content(batch_id)
    return file_response.text


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
                prompt = id[1 : id.rindex("-R")]
                repetition = int(id[id.rindex("-R") + 2 :])

                messages = batch_entry["body"]["messages"]
                new_message = batch_response["response"]["body"]["choices"][0][
                    "message"
                ]
                new_message = {
                    "role": new_message["role"],
                    "content": remove_code_blocks(new_message["content"]),
                }
                messages.append(new_message)
                result[prompt][repetition] = messages

                break

    with open(save_file, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=4, ensure_ascii=False)


def generate_next_response_batch(
    client: openai.OpenAI,
    model_info: RunnableModel,
    save_file: str,
    save_file_batch: str,
):
    conversations = []

    current = {}
    with open(save_file, "r", encoding="utf-8") as file:
        current = json.loads(file.read())

    for prompt_file in current:
        for i, repetition in enumerate(current[prompt_file]):
            custom_id = f"P{prompt_file}-R{i}"
            conversation = current[prompt_file][i]
            if conversation[-1]["content"] == "The unit tests were successful.":
                continue
            conversation_request = _build_single_batch_request(
                model_info, custom_id, conversation
            )
            conversations.append(conversation_request)

    with open(save_file_batch, "w", encoding="utf-8") as file:
        for line in conversations:
            file.write(json.dumps(line, ensure_ascii=False) + "\n")

    batch_input_file = _upload_batch_input_file(client, save_file_batch)
    batch = _create_batch(client, batch_input_file.id)
    return batch.id


def generate_first_response_batch(
    client, model_info: RunnableModel, save_file_batch: str
) -> str:
    conversations = []
    for prompt_file in PROMPT_FILES:
        with open(f"prompts/{prompt_file}", "r", encoding="utf-8") as file:
            prompt_content = file.read()
        for i in range(REPETITIONS):
            conversation: List[ChatCompletionMessageParam] = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_content},
            ]
            custom_id = f"P{prompt_file}-R{i}"
            conversation_request = _build_single_batch_request(
                model_info, custom_id, conversation
            )
            conversations.append(conversation_request)

    with open(save_file_batch, "w", encoding="utf-8") as file:
        for line in conversations:
            file.write(json.dumps(line, ensure_ascii=False) + "\n")

    batch_input_file = _upload_batch_input_file(client, save_file_batch)
    batch = _create_batch(client, batch_input_file.id)
    return batch.id


def wait_for_batch_and_save(
    client: openai.OpenAI,
    batch_id: str,
    save_file: str,
    save_file_batch: str,
    save_file_batch_response: str,
):
    while True:
        status = _status_batch(client, batch_id)
        # print(status)
        if status.status == "completed" or status.status == "cancelled":
            break
        time.sleep(30)
    output_file_id = status.output_file_id
    if output_file_id:
        content = _retrieve_batch(client, output_file_id)
        with open(save_file_batch_response, "w", encoding="utf-8") as file:
            file.write(content)
        _convert_to_json(save_file, save_file_batch, save_file_batch_response)
