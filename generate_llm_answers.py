import json
import os
from dotenv import load_dotenv
import re


load_dotenv()

PROMPT_FILES = os.listdir("prompts")
REPETITIONS = 10


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


time_last_request = 0


def remove_code_blocks(llm_response):
    cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", llm_response, flags=re.MULTILINE)
    cleaned = re.sub(r"\n?```$", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^\s*```\s*$", "", cleaned, flags=re.MULTILINE)
    cleaned = cleaned.strip()
    return cleaned


def amount_of_llm_responses(conversation: dict):
    return sum(1 for message in conversation if message["role"] == "assistant")


def read_file_or_create(save_file_path: str):
    if os.path.exists(save_file_path):
        with open(save_file_path, "r", encoding="utf-8") as file:
            conversations = json.load(file)
    else:
        conversations = {}
    return conversations
