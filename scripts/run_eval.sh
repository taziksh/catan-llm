#!/bin/sh
# Launches one benchmark eval row with the exact endpoint and sampling config
# used for the published charts.
#
# Usage: scripts/run_eval.sh ROW N ROUND [START]
#   e.g. scripts/run_eval.sh deepseek-think 25 n10_v2
#
# N is num_tasks: seeds START..START+N-1 (shuffle off, START defaults to 0). Reruns of a seed are deduped by
# scripts/plots.py, which keeps the newest first-party run per (model, seed).
#
# Sampling goes through a TOML config; inline --sampling.* stringifies nested keys.
set -eu

row=$1
n=$2
round=$3
start=${4:-0}
limits=""

case "$row" in
  deepseek-*) model="deepseek-v4-flash"; url="https://api.deepseek.com"; key="DEEPSEEK_API_KEY" ;;
  kimi-*)     model="kimi-k3"; url="https://api.moonshot.ai/v1"; key="MOONSHOT_API_KEY" ;;
  mimo-*)     model="mimo-v2.5-pro"; url="https://api.xiaomimimo.com/v1"; key="XIAOMI_API_KEY" ;;
  sonnet-*)   model="claude-sonnet-5"; url="https://api.anthropic.com/v1"; key="ANTHROPIC_API_KEY" ;;
  luna-*)     model="gpt-5.6-luna"; url="https://api.openai.com/v1"; key="OPENAI_API_KEY" ;;
  # Free Tinker variant behind the local shim; start it first (effort is pinned there):
  #   uv run python scripts/tinker_serve.py --base-model thinkingmachines/Inkling-2-Mini:peft:65536:free \
  #     --tokenizer thinkingmachines/Inkling-2-Mini --renderer tml_v0 --effort 0.0 \
  #     --context-limit 65536 --request-timeout 3600 \
  #     --served-name inkling2-mini --port 8011
  inkling-*)  model="inkling2-mini"; url="http://localhost:8011/v1"; key="VLLM_API_KEY"; export VLLM_API_KEY=local ;;
  *) echo "unknown row: $row" >&2; exit 1 ;;
esac

case "$row" in
  deepseek-think|mimo-think) sampling="" ;;
  kimi-low) sampling='[sampling]
reasoning_effort = "low"
max_tokens = 4096'
    limits='[env.player0.timeout]
rollout = 3600' ;;
  deepseek-nt|mimo-nt|sonnet-nt) sampling='[sampling.thinking]
type = "disabled"' ;;
  sonnet-think|luna-think) sampling='[sampling]
reasoning_effort = "medium"' ;;
  # Greedy decoding loops on opening placements until the cap; 1.0 matches hosted thinking defaults.
  inkling-low) sampling='[sampling]
temperature = 1.0' ;;
  luna-nt) sampling='[sampling]
reasoning_effort = "none"' ;;
  *) echo "unknown row: $row" >&2; exit 1 ;;
esac

cfg=$(mktemp -t "catan_eval_${row}").toml
cat > "$cfg" <<EOF
model = "$model"
num_tasks = $n
num_rollouts = 1
rich = false
push = false

[env]
trajectory_dir = "data/eval_traces/$round"

[env.taskset]
id = "catan_v1"
seed_start = $start

$limits

[client]
base_url = "$url"
api_key_var = "$key"

$sampling
EOF

echo "config: $cfg"
PYTHONHASHSEED=0 exec uv run --package catan-v1 --env-file .env eval @ "$cfg"
