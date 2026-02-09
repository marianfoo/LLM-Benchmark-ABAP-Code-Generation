"""
Original one-shot benchmark runner.

WARNING: This script runs the full generate→test→feedback loop (Rounds 0–5)
in one go.  It is now safe to re-run: first-round generation is non-destructive,
and next-round generation is blocked if any chats still need SAP testing.

For finer-grained control (resume, status, single-round generation), use:
    python src/llm_generate.py   (LLM generation)
    python src/abap_test.py      (SAP testing / retry)
"""

import argparse
import asyncio
import os
from time import sleep
import anthropic
import openai
import tqdm

from abap_interaction import run_abap_interaction

import generate_llm_answers_batch_openai
import generate_llm_answers_batch_anthropic
import generate_llm_answers_parallel
import generate_llm_answers_openai_direct
from llms import API_PROVIDERS, MODELS_TO_RUN, RunnableModel, get_provider_api_key


def get_available_models():
    """Return list of available model names."""
    return [m["name"] for m in MODELS_TO_RUN]


def get_model_by_name(name: str) -> RunnableModel | None:
    """Find a model by name (case-insensitive, partial match)."""
    name_lower = name.lower()
    for model in MODELS_TO_RUN:
        if model["name"].lower() == name_lower or name_lower in model["name"].lower():
            return model
    return None


def batch_openai(model_info: RunnableModel):
    client = openai.OpenAI(api_key=get_provider_api_key(model_info["provider"]))
    save_file = f"data/{model_info['name']}.json"
    save_file_batch = f"{save_file[:-5]}_batch.jsonl"
    save_file_batch_response = save_file_batch[:-6] + "_response.jsonl"

    # Check for any pending batches first
    print(f"\n[OpenAI] Checking for pending batches for {model_info['name']}...")
    completed = generate_llm_answers_batch_openai.check_and_complete_pending_batches(
        client, model_info["name"]
    )
    if completed:
        print(f"[OpenAI] Completed {len(completed)} pending batch(es)")
        for batch in completed:
            run_abap_interaction(model_info["name"])

    batch_id = generate_llm_answers_batch_openai.generate_first_response_batch(
        client, model_info, save_file_batch, save_file
    )
    if batch_id is not None:
        generate_llm_answers_batch_openai.wait_for_batch_and_save(
            client, batch_id, save_file, save_file_batch, save_file_batch_response
        )
    run_abap_interaction(model_info["name"])

    for i in range(5):
        batch_id = generate_llm_answers_batch_openai.generate_next_response_batch(
            client, model_info, save_file, save_file_batch, round_num=i + 2
        )
        if batch_id is None:
            print(f"[OpenAI] No more conversations to process at round {i + 2}")
            break
        generate_llm_answers_batch_openai.wait_for_batch_and_save(
            client, batch_id, save_file, save_file_batch, save_file_batch_response
        )
        run_abap_interaction(model_info["name"])


def batch_anthropic(model_info: RunnableModel):
    client = anthropic.Anthropic(api_key=get_provider_api_key(model_info["provider"]))
    save_file = f"data/{model_info['name']}.json"
    save_file_batch = f"{save_file[:-5]}_batch.jsonl"
    save_file_batch_response = save_file_batch[:-6] + "_response.jsonl"

    # Check for any pending batches first
    print(f"\n[Anthropic] Checking for pending batches for {model_info['name']}...")
    completed = generate_llm_answers_batch_anthropic.check_and_complete_pending_batches(
        client, model_info["name"]
    )
    if completed:
        print(f"[Anthropic] Completed {len(completed)} pending batch(es)")
        for batch in completed:
            run_abap_interaction(model_info["name"])

    batch_id = generate_llm_answers_batch_anthropic.generate_first_response_batch(
        client, model_info, save_file_batch, save_file
    )
    if batch_id is not None:
        generate_llm_answers_batch_anthropic.wait_for_batch_and_save(
            client, batch_id, save_file, save_file_batch, save_file_batch_response
        )
    run_abap_interaction(model_info["name"])
    
    for i in range(5):
        batch_id = generate_llm_answers_batch_anthropic.generate_next_response_batch(
            client, model_info, save_file, save_file_batch, round_num=i + 2
        )
        if batch_id is None:
            print(f"[Anthropic] No more conversations to process at round {i + 2}")
            break
        generate_llm_answers_batch_anthropic.wait_for_batch_and_save(
            client, batch_id, save_file, save_file_batch, save_file_batch_response
        )
        run_abap_interaction(model_info["name"])


def run_parrallel_model(model_info: RunnableModel):
    if model_info["provider"] == "SAP_AICORE":
        from abap1_orchestration import ABAP1OrchestrationClient

        client = ABAP1OrchestrationClient.from_env(
            model_name=model_info["name"],
            temperature=model_info["temperature"],
            max_tokens=model_info["max_tokens"],
        )
    else:
        provider = API_PROVIDERS[model_info["provider"]]
        client = openai.AsyncOpenAI(
            base_url=provider["base_url"],
            api_key=get_provider_api_key(model_info["provider"]),
        )
    asyncio.run(
        generate_llm_answers_parallel.generate_first_response(client, model_info)
    )
    run_abap_interaction(model_info["name"])

    for num in range(2, 7):
        asyncio.run(
            generate_llm_answers_parallel.generate_next_response(
                client, model_info, num
            )
        )
        run_abap_interaction(model_info["name"])


def run_openai_direct_model(model_info: RunnableModel):
    """Non-batch OpenAI runner (e.g. gpt-5.2-codex) using direct async API calls."""
    client = openai.AsyncOpenAI(
        api_key=get_provider_api_key(model_info["provider"]),
    )
    asyncio.run(
        generate_llm_answers_openai_direct.generate_first_response(client, model_info)
    )
    run_abap_interaction(model_info["name"])

    for num in range(2, 7):
        asyncio.run(
            generate_llm_answers_openai_direct.generate_next_response(
                client, model_info, num
            )
        )
        run_abap_interaction(model_info["name"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM benchmark for ABAP code generation")
    parser.add_argument(
        "--model", "-m",
        type=str,
        help="Run only a specific model (partial name match supported). Use --list to see available models."
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all available models and exit"
    )
    parser.add_argument(
        "--provider", "-p",
        type=str,
        choices=["openai", "openai_direct", "anthropic", "groq", "mistral", "sap_aicore"],
        help="Run only models from a specific provider"
    )
    
    args = parser.parse_args()
    
    # List models and exit
    if args.list:
        print("\nAvailable models:")
        for model in MODELS_TO_RUN:
            print(f"  - {model['name']} ({model['provider']})")
        exit(0)
    
    # Filter models based on arguments
    models_to_run = MODELS_TO_RUN
    
    if args.model:
        model = get_model_by_name(args.model)
        if model is None:
            print(f"Error: Model '{args.model}' not found.")
            print("\nAvailable models:")
            for m in MODELS_TO_RUN:
                print(f"  - {m['name']}")
            exit(1)
        models_to_run = [model]
        print(f"Running single model: {model['name']}")
    
    if args.provider:
        provider_upper = args.provider.upper()
        models_to_run = [m for m in models_to_run if m["provider"] == provider_upper]
        if not models_to_run:
            print(f"Error: No models found for provider '{args.provider}'")
            exit(1)
        print(f"Running {len(models_to_run)} model(s) for provider: {args.provider}")
    
    # Run the models
    for model_info in tqdm.tqdm(models_to_run, desc="Models"):
        print(f"\n{'='*60}")
        print(f"Starting: {model_info['name']} ({model_info['provider']})")
        print(f"{'='*60}")
        
        if model_info["provider"] == "ANTHROPIC":
            batch_anthropic(model_info)
        elif model_info["provider"] == "OPENAI":
            batch_openai(model_info)
        elif model_info["provider"] == "OPENAI_DIRECT":
            run_openai_direct_model(model_info)
        else:
            run_parrallel_model(model_info)
