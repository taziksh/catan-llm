#!/bin/sh
# Starts the Tinker shim for one Tinker-served eval row of scripts/run_eval.sh.
#
# Usage: scripts/serve_tinker_row.sh ROW
#   e.g. scripts/serve_tinker_row.sh gptoss-medium   (then: scripts/run_eval.sh gptoss-medium 25 n10_v2)
#
# The renderer pins each row's reasoning mode; ports match the row's url in run_eval.sh.
set -eu

row=$1
limits="--context-limit 65536 --request-timeout 3600"

case "$row" in
  inkling-low)
    args="--base-model thinkingmachines/Inkling-2-Mini:peft:65536:free --tokenizer thinkingmachines/Inkling-2-Mini
      --renderer tml_v0 --effort 0.0 --served-name inkling2-mini --port 8011" ;;
  inklingsmall-low)
    args="--base-model thinkingmachines/Inkling-2-Small --renderer tml_v0 --effort 0.0
      --served-name inkling2-small --port 8025" ;;
  gptoss-medium)
    args="--base-model openai/gpt-oss-120b --renderer gpt_oss_medium_reasoning
      --served-name gpt-oss-120b --port 8026" ;;
  qwen397b-think)
    args="--base-model Qwen/Qwen3.5-397B-A17B --renderer qwen3_5
      --served-name qwen3.5-397b --port 8022" ;;
  qwen27b-medium)
    args="--base-model Qwen/Qwen3.8-27B --renderer qwen3_8_medium_reasoning
      --served-name qwen3.8-27b --port 8023" ;;
  kimi26-think)
    args="--base-model moonshotai/Kimi-K2.6 --renderer kimi_k26
      --served-name kimi-k2.6 --port 8024" ;;
  *) echo "unknown row: $row" >&2; exit 1 ;;
esac

# shellcheck disable=SC2086  # args is a deliberately word-split flag list
PYTHONHASHSEED=0 exec uv run --env-file .env python scripts/tinker_serve.py --parse-reasoning $args $limits
