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
from llms import API_PROVIDERS, MODELS_TO_RUN, RunnableModel


def batch_openai(model_info: RunnableModel):
    api_provider = API_PROVIDERS[model_info["provider"]]
    client = openai.OpenAI(api_key=api_provider["api_key"])
    save_file = f"{model_info["name"]}.json"
    save_file_batch = f"{save_file[:-5]}_batch.jsonl"
    save_file_batch_response = save_file_batch[:-6] + "_response.jsonl"

    batch_id = generate_llm_answers_batch_openai.generate_first_response_batch(
        client, model_info, save_file_batch
    )
    generate_llm_answers_batch_openai.wait_for_batch_and_save(
        client, batch_id, save_file, save_file_batch, save_file_batch_response
    )
    run_abap_interaction(model_info["name"])

    for i in range(5):
        batch_id = generate_llm_answers_batch_openai.generate_next_response_batch(
            client, model_info, save_file, save_file_batch
        )
        generate_llm_answers_batch_openai.wait_for_batch_and_save(
            client, batch_id, save_file, save_file_batch, save_file_batch_response
        )
        run_abap_interaction(model_info["name"])


def batch_anthropic(model_info: RunnableModel):
    api_provider = API_PROVIDERS[model_info["provider"]]
    client = anthropic.Anthropic(api_key=api_provider["api_key"])
    save_file = f"{model_info["name"]}.json"
    save_file_batch = f"{save_file[:-5]}_batch.jsonl"
    save_file_batch_response = save_file_batch[:-6] + "_response.jsonl"

    batch_id = generate_llm_answers_batch_anthropic.generate_first_response_batch(
        client, model_info, save_file_batch
    )

    generate_llm_answers_batch_anthropic.wait_for_batch_and_save(
        client, batch_id, save_file, save_file_batch, save_file_batch_response
    )
    run_abap_interaction(model_info["name"])
    
    for i in range(5):
        batch_id = generate_llm_answers_batch_anthropic.generate_next_response_batch(
            client, model_info, save_file, save_file_batch
        )
        generate_llm_answers_batch_anthropic.wait_for_batch_and_save(
            client, batch_id, save_file, save_file_batch, save_file_batch_response
        )
        run_abap_interaction(model_info["name"])


def run_parrallel_model(model_info: RunnableModel):
    client = openai.AsyncOpenAI(
        base_url=provider["base_url"], api_key=provider["api_key"]
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


if __name__ == "__main__":
    for model_info in tqdm.tqdm(MODELS_TO_RUN, desc="Models"):
        provider = API_PROVIDERS[model_info["provider"]]
        if model_info["provider"] == "ANTHROPIC":
            batch_anthropic(model_info)
        elif model_info["provider"] == "OPENAI":
            batch_openai(model_info)
        else:
            run_parrallel_model(model_info)
