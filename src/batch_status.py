"""
Utility script to check batch status and retrieve results for pending batches.

Usage:
    python src/batch_status.py              # Check all pending batches
    python src/batch_status.py --list       # List all tracked batches
    python src/batch_status.py --complete   # Complete any finished batches and save results
"""

import argparse
from dotenv import load_dotenv
import openai
import anthropic

from llms import get_provider_api_key
import generate_llm_answers_batch_openai as openai_batch
import generate_llm_answers_batch_anthropic as anthropic_batch

load_dotenv()


def check_openai_batches(client: openai.OpenAI, complete: bool = False):
    """Check status of all pending OpenAI batches."""
    print("\n" + "=" * 60)
    print("OPENAI BATCHES")
    print("=" * 60)
    
    pending = openai_batch.get_pending_batches()
    
    if not pending:
        print("No pending OpenAI batches.")
        return
    
    print(f"Found {len(pending)} pending batch(es):\n")
    
    for batch_info in pending:
        batch_id = batch_info["batch_id"]
        print(f"Batch: {batch_id}")
        print(f"  Model: {batch_info['model_name']}")
        print(f"  Round: {batch_info['round_num']}")
        print(f"  Created: {batch_info['created_at']}")
        
        try:
            status = client.batches.retrieve(batch_id)
            print(f"  Status: {status.status}")
            print(f"  Progress: {status.request_counts.completed}/{status.request_counts.total}")
            
            if status.status == "completed" and complete:
                print("  -> Retrieving results...")
                output_file_id = status.output_file_id
                if output_file_id:
                    content = client.files.content(output_file_id).text
                    with open(batch_info["save_file_batch_response"], "w", encoding="utf-8") as f:
                        f.write(content)
                    openai_batch._convert_to_json(
                        batch_info["save_file"],
                        batch_info["save_file_batch"],
                        batch_info["save_file_batch_response"]
                    )
                    openai_batch._update_batch_status(batch_id, "completed")
                    print(f"  -> Results saved to {batch_info['save_file']}")
            elif status.status in ("failed", "cancelled", "expired"):
                openai_batch._update_batch_status(batch_id, status.status)
                if status.errors:
                    print(f"  Errors: {status.errors}")
        except Exception as e:
            print(f"  ERROR: {e}")
        
        print()


def check_anthropic_batches(client: anthropic.Anthropic, complete: bool = False):
    """Check status of all pending Anthropic batches."""
    print("\n" + "=" * 60)
    print("ANTHROPIC BATCHES")
    print("=" * 60)
    
    pending = anthropic_batch.get_pending_batches()
    
    if not pending:
        print("No pending Anthropic batches.")
        return
    
    print(f"Found {len(pending)} pending batch(es):\n")
    
    for batch_info in pending:
        batch_id = batch_info["batch_id"]
        print(f"Batch: {batch_id}")
        print(f"  Model: {batch_info['model_name']}")
        print(f"  Round: {batch_info['round_num']}")
        print(f"  Created: {batch_info['created_at']}")
        
        try:
            status = client.messages.batches.retrieve(batch_id)
            print(f"  Status: {status.processing_status}")
            
            if status.processing_status == "ended" and complete:
                print("  -> Retrieving results...")
                if status.results_url:
                    content = list(client.messages.batches.results(batch_id))
                    from anthropic.types import TextBlock
                    from generate_llm_answers import remove_code_blocks
                    
                    with open(batch_info["save_file_batch_response"], "w", encoding="utf-8") as f:
                        failed = 0
                        for entry in content:
                            if entry.result.type == "succeeded":
                                import json
                                successful_entry = {
                                    "custom_id": entry.custom_id,
                                    "response": " ".join(
                                        remove_code_blocks(c.text)
                                        for c in entry.result.message.content
                                        if isinstance(c, TextBlock)
                                    ),
                                }
                                f.write(json.dumps(successful_entry, ensure_ascii=False) + "\n")
                            else:
                                failed += 1
                        if failed > 0:
                            print(f"  {failed} requests failed")
                    
                    anthropic_batch._convert_to_json(
                        batch_info["save_file"],
                        batch_info["save_file_batch"],
                        batch_info["save_file_batch_response"]
                    )
                    anthropic_batch._update_batch_status(batch_id, "completed")
                    print(f"  -> Results saved to {batch_info['save_file']}")
        except Exception as e:
            print(f"  ERROR: {e}")
        
        print()


def list_all_batches():
    """List all tracked batches (both providers)."""
    print("\n" + "=" * 60)
    print("ALL TRACKED BATCHES")
    print("=" * 60)
    
    openai_batch.list_all_batches()
    anthropic_batch.list_all_batches()


def main():
    parser = argparse.ArgumentParser(description="Check batch status and retrieve results")
    parser.add_argument("--list", "-l", action="store_true", help="List all tracked batches")
    parser.add_argument("--complete", "-c", action="store_true", help="Complete finished batches and save results")
    parser.add_argument("--openai-only", action="store_true", help="Only check OpenAI batches")
    parser.add_argument("--anthropic-only", action="store_true", help="Only check Anthropic batches")
    
    args = parser.parse_args()
    
    if args.list:
        list_all_batches()
        return
    
    if not args.anthropic_only:
        openai_client = openai.OpenAI(api_key=get_provider_api_key("OPENAI"))
        check_openai_batches(openai_client, complete=args.complete)
    
    if not args.openai_only:
        anthropic_client = anthropic.Anthropic(api_key=get_provider_api_key("ANTHROPIC"))
        check_anthropic_batches(anthropic_client, complete=args.complete)
    
    print("\n" + "-" * 60)
    if args.complete:
        print("Done! Finished batches have been processed.")
    else:
        print("To retrieve results for completed batches, run with --complete flag")


if __name__ == "__main__":
    main()
