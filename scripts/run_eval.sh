#!/bin/sh
# Launches one benchmark eval row with the exact endpoint and sampling config
# used for the published charts.
#
# Usage: scripts/run_eval.sh ROW N ROUND
#   e.g. scripts/run_eval.sh deepseek-think 25 n10_v2
#
# N is num_tasks: seeds 0..N-1 (shuffle off). Reruns of a seed are deduped by
# scripts/plots.py, which keeps the newest first-party run per (model, seed).
#
# Sampling goes through a TOML config; inline --sampling.* stringifies nested keys.
set -eu

row=$1
n=$2
round=$3
limits=""

case "$row" in
  deepseek-*) model="deepseek-v4-flash"; url="https://api.deepseek.com"; key="DEEPSEEK_API_KEY" ;;
  kimi-*)     model="kimi-k3"; url="https://api.moonshot.ai/v1"; key="MOONSHOT_API_KEY" ;;
  mimo-*)     model="mimo-v2.5-pro"; url="https://api.xiaomimimo.com/v1"; key="XIAOMI_API_KEY" ;;
  sonnet-*)   model="claude-sonnet-5"; url="https://api.anthropic.com/v1"; key="ANTHROPIC_API_KEY" ;;
  luna-*)     model="gpt-5.6-luna"; url="https://api.openai.com/v1"; key="OPENAI_API_KEY" ;;
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

$limits

[client]
base_url = "$url"
api_key_var = "$key"

$sampling
EOF

echo "config: $cfg"
PYTHONHASHSEED=0 exec uv run --package catan-v1 --env-file .env eval @ "$cfg"
