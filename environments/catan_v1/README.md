# catan-v1

A seeded, multi-turn Settlers of Catan environment for Verifiers and Prime
Hosted Training.

Each rollout is one complete four-player game. Prime's `load_environment`
entrypoint supports exactly one model-controlled seat against three scripted
Catanatron policies. The direct Verifiers v1 `CatanEnv` API also supports
multiple model-controlled seats. At every decision, each agent receives a
self-contained textual board state and selects one legal move ID.

The primary reward is game victory. A small normalized victory-point reward
provides denser feedback:

- `reward_win`: `1` for a win, otherwise `0`
- `reward_vp`: `min(victory_points, 10) / 10`, weighted by `vp_coef`

Seeds `0` through `9999` are reserved for evaluation. Training datasets must
start at `10000` or above; the loader rejects any range that crosses this
boundary.

The engine also requires `PYTHONHASHSEED=0` at interpreter startup for
reproducible action ordering. Link it to the hosted environment once:

```bash
prime env var create taziksh/catan-v1 \
  --name PYTHONHASHSEED \
  --value 0 \
  --description "Required for deterministic Catan rollouts"
```

## Hosted configuration

```toml
[[env]]
id = "taziksh/catan-v1"
args = { seed_start = 10000, seats = "agent,value_function,value_function,value_function", invalid_retries = 1, vp_coef = 0.1, max_turns = 500 }
```

The hosted loader also accepts `num_seeds`, `trajectory_dir`, `system_prompt`,
and `timeout_seconds`.
