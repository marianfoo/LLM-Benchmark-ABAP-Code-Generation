#!/usr/bin/env python3
"""
Smoke test to verify API keys and model names are correct for all configured models.
Run this before starting the main benchmark to catch configuration issues early.
"""

import sys
from openai import OpenAI
import anthropic

from llms import API_PROVIDERS, MODELS_TO_RUN, get_provider_api_key


def test_openai_compatible(model_name: str, provider_name: str, base_url: str, api_key: str) -> tuple[bool, str]:
    """Test OpenAI-compatible APIs (Groq, Mistral, OpenAI)."""
    try:
        client = OpenAI(base_url=base_url, api_key=api_key) if base_url else OpenAI(api_key=api_key)
        
        # GPT-5 family requires max_completion_tokens and does not accept
        # a custom temperature (always 1). Applies to both OPENAI and OPENAI_DIRECT.
        if "gpt-5" in model_name:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "Say 'OK' and nothing else."}],
                max_completion_tokens=10,
                # GPT-5 only supports temperature=1 (default)
            )
        else:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "Say 'OK' and nothing else."}],
                max_tokens=10,
                temperature=0,
            )
        content = response.choices[0].message.content or ""
        content = content.strip()
        return True, f"Response: '{content[:50]}...'" if len(content) > 50 else f"Response: '{content}'"
    except Exception as e:
        return False, str(e)


def test_anthropic(model_name: str, api_key: str) -> tuple[bool, str]:
    """Test Anthropic API."""
    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model_name,
            max_tokens=10,
            messages=[{"role": "user", "content": "Say 'OK' and nothing else."}],
        )
        content = response.content[0].text.strip()
        return True, f"Response: '{content[:50]}...'" if len(content) > 50 else f"Response: '{content}'"
    except Exception as e:
        return False, str(e)


def test_sap_aicore(model_name: str) -> tuple[bool, str]:
    """Test SAP ABAP-1 through SAP AI Core orchestration."""
    try:
        from abap1_orchestration import ABAP1OrchestrationClient, missing_aicore_env_vars

        missing = missing_aicore_env_vars()
        if missing:
            return False, f"Missing env vars: {', '.join(missing)}"

        client = ABAP1OrchestrationClient.from_env(
            model_name=model_name,
            temperature=0.0,
            max_tokens=64,
        )
        content = client.complete_sync(
            [
                {"role": "system", "content": "Reply with OK only."},
                {"role": "user", "content": "Return OK."},
            ]
        ).strip()
        return True, f"Response: '{content[:50]}...'" if len(content) > 50 else f"Response: '{content}'"
    except Exception as e:
        return False, str(e)


def main():
    print("=" * 70)
    print("SMOKE TEST - Verifying API Keys and Model Names")
    print("=" * 70)
    print()

    # Track results
    results = []
    seen_models = set()  # Skip duplicates

    for model in MODELS_TO_RUN:
        model_name = model["name"]
        provider_name = model["provider"]

        # Skip duplicates
        model_key = (model_name, provider_name)
        if model_key in seen_models:
            continue
        seen_models.add(model_key)

        provider = API_PROVIDERS.get(provider_name)
        if not provider:
            results.append((model_name, provider_name, False, f"Provider '{provider_name}' not found"))
            continue

        base_url = provider.get("base_url")

        print(f"Testing: {model_name} ({provider_name})...", end=" ", flush=True)

        if provider_name == "SAP_AICORE":
            success, message = test_sap_aicore(model_name)
        elif provider_name == "ANTHROPIC":
            try:
                api_key = get_provider_api_key(provider_name)
            except RuntimeError as e:
                success, message = False, str(e)
            else:
                success, message = test_anthropic(model_name, api_key)
        else:
            # OpenAI-compatible (GROQ, MISTRAL, OPENAI, OPENAI_DIRECT)
            try:
                api_key = get_provider_api_key(provider_name)
            except RuntimeError as e:
                success, message = False, str(e)
            else:
                success, message = test_openai_compatible(model_name, provider_name, base_url, api_key)

        status = "PASS" if success else "FAIL"
        print(status)
        results.append((model_name, provider_name, success, message))

    # Summary
    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    passed = sum(1 for r in results if r[2])
    failed = len(results) - passed

    for model_name, provider_name, success, message in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}  {model_name} ({provider_name})")
        if not success:
            # Truncate long error messages
            msg = message[:100] + "..." if len(message) > 100 else message
            print(f"         Error: {msg}")

    print()
    print(f"Total: {passed}/{len(results)} passed")

    if failed > 0:
        print(f"\n⚠️  {failed} model(s) failed. Fix configuration before running benchmark.")
        sys.exit(1)
    else:
        print("\n✅ All models verified successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
