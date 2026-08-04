# catan-v1

A Settlers of Catan environment for training and evaluating language models.

Each rollout is a full four-player game. The model sees a text description of the board and chooses from a numbered list of legal moves. Players only trade with the bank.

On Prime, one LLM plays against three bots. Run `CatanEnv` directly for games with multiple LLMs.

## reward

The main reward is winning the game. A small normalized victory-point reward provides denser feedback:

- `reward_win`: `1` for a win, otherwise `0`
- `reward_vp`: `min(victory_points, 10) / 10`, weighted by `vp_coef`

Truncated games keep their victory-point.

## seeds

Seeds `0–9999` are reserved for evals. Training must start at `10000` or above. The loader rejects ranges that cross the boundary.

## hosted setup

Set `PYTHONHASHSEED=0` once so action ordering stays deterministic:

```bash
prime env var create taziksh/catan-v1 \
  --name PYTHONHASHSEED \
  --value 0
```

```toml
[[env]]
id = "taziksh/catan-v1"
args = { seed_start = 10000, seats = "agent,value_function,value_function,value_function", invalid_retries = 1, vp_coef = 0.1, max_turns = 500 }
```

Other args: `num_seeds`, `trajectory_dir`, `system_prompt`, and `timeout_seconds`.
