#!/usr/bin/env bash
# Dedicated Hindsight server for the experiment (SPEC §10). Never the shared household instance.
#
# Pinned: hindsight-api-slim (uv.lock), embedder + reranker revisions (models.lock.json), the
# memory LLM alias. Binds to loopback only. The gateway key is read from the private env file at
# start-up and passed through the environment; it is never written anywhere.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"

set -a
# shellcheck disable=SC1090
. "${ELH_MODEL_ENV_FILE:-$HOME/.config/wintermute/model-api.env}"
set +a

export HINDSIGHT_API_HOST=127.0.0.1
export HINDSIGHT_API_PORT="${ELH_HINDSIGHT_PORT:-8901}"
export HINDSIGHT_API_DATABASE_URL="pg0://elh-hindsight:${ELH_HINDSIGHT_PG_PORT:-8902}"
export HINDSIGHT_API_WORKERS=1
export HINDSIGHT_API_MCP_ENABLED=false
export HINDSIGHT_API_LOG_LEVEL="${ELH_HINDSIGHT_LOG_LEVEL:-info}"

# Memory LLM: the study alias, one request at a time (the alias serves one at a time).
export HINDSIGHT_API_LLM_PROVIDER=openai
export HINDSIGHT_API_LLM_BASE_URL="$WINTERMUTE_MODEL_API_BASE"
export HINDSIGHT_API_LLM_API_KEY="$WINTERMUTE_MODEL_API_KEY"
export HINDSIGHT_API_LLM_MODEL=study-gemma4-26b-a4b
export HINDSIGHT_API_LLM_MAX_CONCURRENT=1
export HINDSIGHT_API_LLM_TIMEOUT=180
# Gemma reasons by default; with thinking on, structured extraction spends its whole token budget
# reasoning and retain stalls. Off is preregistered for every memory-LLM call.
export HINDSIGHT_API_LLM_EXTRA_BODY='{"chat_template_kwargs":{"enable_thinking":false}}'
# Recorded so the ledger can charge memory-LLM tokens as `learning`.
export HINDSIGHT_API_LLM_TRACE_ENABLED=true

# Embedder and reranker: in-process ONNX on CPU, from the pinned local copies.
m="$here/models"
export HINDSIGHT_API_EMBEDDINGS_PROVIDER=onnx
export HINDSIGHT_API_EMBEDDINGS_ONNX_MODEL_PATH="$m/embedder/onnx/model.onnx"
export HINDSIGHT_API_EMBEDDINGS_ONNX_TOKENIZER_NAME_OR_PATH="$m/embedder"
export HINDSIGHT_API_RERANKER_PROVIDER=onnx
export HINDSIGHT_API_RERANKER_ONNX_MODEL_PATH="$m/reranker/onnx/model.onnx"
export HINDSIGHT_API_RERANKER_ONNX_TOKENIZER_NAME_OR_PATH="$m/reranker"
export HF_HUB_OFFLINE=1   # never fetch a model at run time; fetch_models.py pins them

unset WINTERMUTE_MODEL_API_KEY
cd "$here"
uv run --frozen python fetch_models.py >/dev/null
exec uv run --frozen hindsight-api
