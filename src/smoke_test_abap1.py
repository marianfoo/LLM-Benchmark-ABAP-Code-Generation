#!/usr/bin/env python3
"""
Smoke test for SAP ABAP-1 via SAP AI Core orchestration.

Usage:
    python src/smoke_test_abap1.py
    python src/smoke_test_abap1.py --model sap--abap-1 --message "Create a tiny ABAP class."
"""

import argparse
import sys

from abap1_orchestration import ABAP1OrchestrationClient, missing_aicore_env_vars


def main():
    parser = argparse.ArgumentParser(
        description="Validate SAP ABAP-1 connection through SAP AI Core orchestration."
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name in orchestration (default: AICORE_MODEL_NAME or sap-abap-1)",
    )
    parser.add_argument(
        "--message",
        default="Reply with OK only.",
        help="Prompt message to send to the model.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for the orchestration LLM module (default: 0.0)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum output tokens (default: 128)",
    )
    args = parser.parse_args()

    missing = missing_aicore_env_vars()
    if missing:
        print("Missing SAP AI Core environment variables:")
        for name in missing:
            print(f"  - {name}")
        print("\nSet them in .env and retry.")
        sys.exit(1)

    client = ABAP1OrchestrationClient.from_env(
        model_name=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    response = client.complete_sync(
        [
            {"role": "system", "content": "You are a helpful ABAP coding assistant."},
            {"role": "user", "content": args.message},
        ]
    )

    print("ABAP-1 response:")
    print(response)


if __name__ == "__main__":
    main()
