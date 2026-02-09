"""
Smoke test script for OpenAI batch API with 10 random prompts (single run).
This is a one-time test to verify the batch pipeline works correctly.
"""

import json
import os
import random
import time
from typing import List

import openai
from dotenv import load_dotenv
from openai.types.chat import ChatCompletionMessageParam

from generate_llm_answers import SYSTEM_PROMPT, remove_code_blocks
from llms import MODELS_TO_RUN

load_dotenv()

# Configuration for smoke test
NUM_PROMPTS = 10
REPETITIONS = 1  # Single run only

# Get all available prompts and select 10 random ones
ALL_PROMPTS = os.listdir("dataset/prompts")
random.seed(42)  # Fixed seed for reproducibility
SELECTED_PROMPTS = random.sample(ALL_PROMPTS, NUM_PROMPTS)

print(f"Selected {NUM_PROMPTS} random prompts for smoke test:")
for p in sorted(SELECTED_PROMPTS):
    print(f"  - {p}")


def _build_single_batch_request(model_name: str, custom_id: str, conversation: List, temperature: float, max_tokens: int):
    """Build a single batch request for the OpenAI batch API."""
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model_name,
            "messages": conversation,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
        },
    }


def _upload_batch_input_file(client: openai.OpenAI, file_name: str):
    """Upload the batch input file to OpenAI."""
    batch_input_file = client.files.create(file=open(file_name, "rb"), purpose="batch")
    return batch_input_file


def _create_batch(client: openai.OpenAI, batch_input_file_id: str):
    """Create a batch job on OpenAI."""
    batch_info = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "ABAP LLM Benchmark - Smoke Test"},
    )
    return batch_info


def _status_batch(client: openai.OpenAI, batch_id: str):
    """Check the status of a batch job."""
    batch = client.batches.retrieve(batch_id)
    return batch


def _retrieve_batch(client: openai.OpenAI, file_id: str):
    """Retrieve the results of a completed batch job."""
    file_response = client.files.content(file_id)
    return file_response.text


def generate_smoke_test_batch(client: openai.OpenAI, model_info: dict, batch_file: str) -> str:
    """Generate the initial batch request for smoke test prompts."""
    conversations = []
    
    for prompt_file in SELECTED_PROMPTS:
        with open(f"dataset/prompts/{prompt_file}", "r", encoding="utf-8") as file:
            prompt_content = file.read()
        
        for i in range(REPETITIONS):
            conversation: List[ChatCompletionMessageParam] = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_content},
            ]
            custom_id = f"P{prompt_file}-R{i}"
            conversation_request = _build_single_batch_request(
                model_info["name"],
                custom_id,
                conversation,
                model_info["temperature"],
                model_info["max_tokens"],
            )
            conversations.append(conversation_request)
    
    # Write batch file
    with open(batch_file, "w", encoding="utf-8") as file:
        for line in conversations:
            file.write(json.dumps(line, ensure_ascii=False) + "\n")
    
    print(f"Created batch file with {len(conversations)} requests: {batch_file}")
    
    # Upload and create batch
    batch_input_file = _upload_batch_input_file(client, batch_file)
    batch = _create_batch(client, batch_input_file.id)
    
    print(f"Batch created with ID: {batch.id}")
    return batch.id


def wait_for_batch_completion(client: openai.OpenAI, batch_id: str, response_file: str):
    """Wait for batch to complete and save results."""
    print(f"Waiting for batch {batch_id} to complete...")
    
    while True:
        status = _status_batch(client, batch_id)
        print(f"  Status: {status.status} (completed: {status.request_counts.completed}/{status.request_counts.total})")
        
        if status.status == "completed":
            print("Batch completed successfully!")
            break
        elif status.status in ("failed", "cancelled", "expired"):
            print(f"Batch ended with status: {status.status}")
            if status.errors:
                print(f"Errors: {status.errors}")
            return None
        
        time.sleep(30)
    
    # Retrieve and save results
    output_file_id = status.output_file_id
    if output_file_id:
        content = _retrieve_batch(client, output_file_id)
        with open(response_file, "w", encoding="utf-8") as file:
            file.write(content)
        print(f"Results saved to: {response_file}")
        return response_file
    
    return None


def parse_results(batch_file: str, response_file: str, output_file: str):
    """Parse batch results into a structured JSON format."""
    result = {prompt: [[] for _ in range(REPETITIONS)] for prompt in SELECTED_PROMPTS}
    
    with open(batch_file, "r", encoding="utf-8") as file:
        batch_entries = [json.loads(line) for line in file.readlines()]
    
    with open(response_file, "r", encoding="utf-8") as file:
        batch_responses = [json.loads(line) for line in file.readlines()]
    
    for batch_entry in batch_entries:
        for batch_response in batch_responses:
            if batch_entry["custom_id"] == batch_response["custom_id"]:
                custom_id = batch_entry["custom_id"]
                prompt = custom_id[1:custom_id.rindex("-R")]
                repetition = int(custom_id[custom_id.rindex("-R") + 2:])
                
                messages = batch_entry["body"]["messages"]
                new_message = batch_response["response"]["body"]["choices"][0]["message"]
                new_message = {
                    "role": new_message["role"],
                    "content": remove_code_blocks(new_message["content"]),
                }
                messages.append(new_message)
                result[prompt][repetition] = messages
                break
    
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=4, ensure_ascii=False)
    
    print(f"Parsed results saved to: {output_file}")


def main():
    # Find an OpenAI model from the MODELS_TO_RUN list
    openai_model = None
    for model in MODELS_TO_RUN:
        if model.get("provider") == "OPENAI":
            openai_model = model
            break
    
    if not openai_model:
        print("No OpenAI model found in MODELS_TO_RUN list")
        return
    
    print(f"\nUsing model: {openai_model['name']}")
    
    # Initialize OpenAI client
    client = openai.OpenAI()
    
    # File paths for this smoke test
    batch_file = "data/smoke_test_batch.jsonl"
    response_file = "data/smoke_test_batch_response.jsonl"
    output_file = "data/smoke_test_results.json"
    
    # Generate and submit batch
    batch_id = generate_smoke_test_batch(client, openai_model, batch_file)
    
    # Wait for completion and save results
    result_file = wait_for_batch_completion(client, batch_id, response_file)
    
    if result_file:
        # Parse results into structured format
        parse_results(batch_file, response_file, output_file)
        print("\nSmoke test completed successfully!")
    else:
        print("\nSmoke test failed - batch did not complete successfully")


if __name__ == "__main__":
    main()
