# ABAP-1 Setup in SAP BTP (SAP AI Core + Orchestration)

This guide sets up ABAP-1 in SAP BTP and wires it into this benchmark repository.

## 1. Prerequisites in BTP

1. In your BTP subaccount, subscribe to:
   - SAP AI Core
   - SAP AI Launchpad
2. Enable Cloud Foundry and create an org/space for your runtime.

Reference:
- [SAP AI Core Service Guide - Initial Setup](https://help.sap.com/docs/sap-ai-core/sap-ai-core-service-guide/initial-setup)
- [SAP AI Core Service Guide - PDF export](https://help.sap.com/doc/65de2977205c403bbc107264b8eccf4b/CLOUD/en-US/SAP_AI_Core_Service_Guide_en-US.pdf)

## 2. Create SAP AI Core service instance + service key

1. In BTP Cockpit, create a service instance for **SAP AI Core** using plan **`extended`**.
2. Create a service key for that instance.
3. Keep the service key JSON. You need these fields:
   - `clientid`
   - `clientsecret`
   - `url`
   - `serviceurls.AI_API_URL`
   - `AI-Resource-Group` (your chosen resource group in SAP AI Core)

Reference:
- [SAP AI Core Service Guide - Use a Service Key with SAP AI Core](https://help.sap.com/docs/sap-ai-core/sap-ai-core-service-guide/use-service-key-with-sap-ai-core)
- [SAP AI Core Service Guide - PDF export](https://help.sap.com/doc/65de2977205c403bbc107264b8eccf4b/CLOUD/en-US/SAP_AI_Core_Service_Guide_en-US.pdf)

## 3. Configure AI Launchpad connection and ABAP-1 deployment

1. Open SAP AI Launchpad and connect it to your SAP AI Core runtime.
2. In **Model Library**, create a deployment from **SAP ABAP-1**.
3. Ensure you have an **orchestration deployment** available (the Launchpad flow creates it automatically in the tutorial flow).

References:
- [Set Up Generative AI Hub in SAP AI Core - tutorial](https://developers.sap.com/tutorials/ai-core-genai-hub-set-up.html)
- [SAP AI Launchpad User Guide - Create Deployment for Orchestration](https://help.sap.com/docs/sap-ai-launchpad/sap-ai-launchpad-user-guide/create-deployment-for-orchestration)
- [SAP Cloud SDK for AI (Python) - Chat Completion API (prerequisite: orchestration deployment)](https://help.sap.com/doc/generative-ai-hub-sdk/CLOUD/en-US/_attachments/python-sdk/docs/notebooks/04_chat_completion.html)

## 4. Add credentials to `.env`

Make sure dependencies are installed first (includes `sap-ai-sdk-gen`):

```bash
uv sync
```

Copy and fill the template:

```bash
cp .env.example .env
```

Required variables for ABAP-1:

```bash
AICORE_AUTH_URL=<service-key url>/oauth/token
AICORE_CLIENT_ID=<service-key clientid>
AICORE_CLIENT_SECRET=<service-key clientsecret>
AICORE_BASE_URL=<service-key serviceurls.AI_API_URL>
AICORE_RESOURCE_GROUP=<your-resource-group>

# optional model routing overrides
AICORE_MODEL_NAME=sap--abap-1
AICORE_MODEL_VERSION=latest
```

Notes:
- `AICORE_AUTH_URL` is the OAuth token endpoint (`/oauth/token`).
- `AICORE_MODEL_NAME` defaults to `sap--abap-1` in this repository.

References:
- [Python SDK env variable configuration](https://help.sap.com/doc/generative-ai-hub-sdk/CLOUD/en-US/_attachments/python-sdk/docs/index.html)
- [SAP AI SDK model support list (`sap--abap-1`) and orchestration-only note](https://sap.github.io/ai-sdk/docs/js/orchestration/models)

## 5. Run the ABAP-1 connection smoke test

```bash
python src/smoke_test_abap1.py
```

Expected result: a non-empty model response. If environment values are missing, the script prints the exact variable names.

## 6. Run ABAP-1 in the benchmark flow

```bash
# round 0 generation
python src/llm_generate.py --model sap--abap-1 --mode first

# SAP testing
python src/parallel_runner.py --model sap--abap-1 --workers 4
python src/abap_test.py --model sap--abap-1 --mode retry --max-attempts 3

# next correction round
python src/llm_generate.py --model sap--abap-1 --mode next
```

Repeat test + next-round generation until all rounds are done.

## 7. Source mapping (what each reference explains)

- **Service instance/service key and credential fields**:
  [SAP AI Core Service Guide](https://help.sap.com/doc/generative-ai-hub-sdk/CLOUD/en-US/_attachments/python-sdk/docs/index.html)
- **Launchpad connection and orchestration deployment flow**:
  [Set Up Generative AI Hub tutorial](https://developers.sap.com/tutorials/ai-core-genai-hub-set-up.html)
- **ABAP-1 availability and model identifier**:
  [SAP ABAP-1 in SAP Business AI catalog](https://www.sap.com/products/artificial-intelligence/business-ai/abap-ai-models.html),
  [SAP AI SDK model support](https://sap.github.io/ai-sdk/docs/js/orchestration/models)
- **Python orchestration client setup and required env vars**:
  [SAP Cloud SDK for AI (Python)](https://help.sap.com/doc/generative-ai-hub-sdk/CLOUD/en-US/_attachments/python-sdk/docs/index.html),
  [Chat Completion notebook](https://help.sap.com/doc/generative-ai-hub-sdk/CLOUD/en-US/_attachments/python-sdk/docs/notebooks/04_chat_completion.html)
