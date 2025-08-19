import datetime
import json
import os
import time
from typing import Iterable, List
from openai.types.chat import ChatCompletionMessageParam
import openai
from dotenv import load_dotenv
import tqdm
import re

load_dotenv()

REPETITIONS = 10
TEMPERATURE = 0.2
MODELS_TO_RUN = [
    "llama-3.3-70b-instruct",
    "qwen2.5-coder-32b-instruct",
    "codestral-22b",
]
SYSTEM_PROMPT = """\
You are an expert ABAP developer specializing in generating high-quality ABAP classes for SAP systems.
You write global ABAP classes with static methods.

System Environment
SAP Version: NetWeaver 7.57 (S/4HANA 2022)
Decimal Separator: Always use . (period) as decimal separator in numeric literals.

Development Rules
- Always use RETURNING parameters instead of EXPORTING.
- There should only be one public method per class.
- Class names must always start with Z.

Always respond with both the class definition and implementation code.
Only respond with the code. Do not include any explanations or comments.\
"""

KISKI_BASE_URL = "https://chat-ai.academiccloud.de/v1"
KISKI_API_KEY = os.getenv("KISKI_API_KEY")


client = openai.OpenAI(api_key=KISKI_API_KEY, base_url=KISKI_BASE_URL)
time_last_request = 0


def remove_code_blocks(llm_response):
    cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", llm_response, flags=re.MULTILINE)
    cleaned = re.sub(r"\n?```$", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^\s*```\s*$", "", cleaned, flags=re.MULTILINE)
    cleaned = cleaned.strip()
    return cleaned


def ask_kiski(model_name: str, chat_history: Iterable[ChatCompletionMessageParam]):
    global time_last_request

    current_hour = datetime.datetime.now().hour
    is_day_hours = 7 <= current_hour < 19
    if is_day_hours:
        min_interval = 6
    else:
        min_interval = 1

    time_since_last = time.time() - time_last_request

    if time_since_last < min_interval:
        sleep_time = min_interval - time_since_last
        time.sleep(sleep_time)

    time_last_request = time.time()

    max_retries = 5
    retry_delay = 60

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name, messages=chat_history, temperature=TEMPERATURE, max_tokens=5000
            )
            text_response = response.choices[0].message.content
            text_response = remove_code_blocks(text_response)
            return text_response

        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            else:
                print("retrying...")
                time.sleep(retry_delay)


def amount_of_llm_responses(conversation: dict):
    return sum(1 for message in conversation if message["role"] == "assistant")


def read_file_or_create(save_file_path: str):
    if os.path.exists(save_file_path):
        with open(save_file_path, "r", encoding="utf-8") as file:
            conversations = json.load(file)
    else:
        conversations = {}
    return conversations


def generate_first_response(save_file_path: str, model_name: str):
    conversations = read_file_or_create(save_file_path)

    prompt_files = os.listdir("prompts")
    for prompt_file in tqdm.tqdm(prompt_files, desc="Processing prompts", leave=False):

        with open(f"prompts/{prompt_file}", "r", encoding="utf-8") as file:
            prompt_content = file.read()

        for i in tqdm.tqdm(
            range(REPETITIONS), desc="Generating responses", leave=False
        ):
            if not prompt_file in conversations:
                conversations[prompt_file] = []
            if len(conversations[prompt_file]) <= i:
                conversation: List[ChatCompletionMessageParam] = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_content},
                ]
                response = ask_kiski(model_name, conversation)
                conversation.append({"role": "assistant", "content": response})
                conversations[prompt_file].append(conversation)

            with open(save_file_path, "w", encoding="utf-8") as f:
                json.dump(conversations, f, ensure_ascii=False, indent=4)

    with open(save_file_path, "w", encoding="utf-8") as f:
        json.dump(conversations, f, ensure_ascii=False, indent=4)


def generate_next_response(response_number: int, save_file_path: str, model_name: str):
    with open(save_file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    for conversations in tqdm.tqdm(
        data.values(), desc="Processing prompts", leave=False
    ):
        for i, conversation in tqdm.tqdm(
            enumerate(conversations),
            desc="Processing conversations",
            leave=False,
            total=len(conversations),
        ):
            if amount_of_llm_responses(conversation) >= response_number:
                continue
            if conversation[-1]["content"] == "The unit tests were successful.":
                continue

            response = ask_kiski(model_name, conversation)
            conversation.append({"role": "assistant", "content": response})

        with open(save_file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    with open(save_file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    for model in tqdm.tqdm(MODELS_TO_RUN, desc="Models"):
        # generate_first_response(f"{model}.json", model)
        generate_next_response(6, f"{model}.json", model)
