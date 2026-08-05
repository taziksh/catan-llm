# Catan post-training handoff

Keep this file live-only; use git history and linked audits for completed work.
Before changing or publishing `environments/catan_v1`, read
`ENV_DESIGN_REVIEW.md`; it is an unresolved checklist, not approval.

## Invariants

- Expose only real-player information; opponent resources and unplayed
  development cards stay hidden.
- Launch with `PYTHONHASHSEED=0`; train on seeds `>=10_000`, evaluate below
  `10_000`, and split only at game boundaries.
- Preserve the replay and determinization invariants in
  `tests/test_replay.py` and `tests/test_determinize.py`.
- Ask before changing settled experimental parameters.

## Pinned model

- Base: `Qwen/Qwen3.5-9B` at revision
  `c202236235762e1c871ad0ccb60c8ee5ba337b9a`; renderer
  `qwen3_5_disable_thinking`; prompt `v3`.
- Initialize GRPO from `data/checkpoints/dpo_fair/dpo_full` on merged dagger2.
  Never use public Qwen, the older 10k checkpoint, or a thinking renderer.
- The merge must report `applied=249 missing=0`; implementation:
  `data/pod_scripts/merge3.py`. Exact hashes and DPO parameters are in the
  adapter's `run_manifest.json` and `scripts/run_dpo.py`.
- Output is one stable move ID, for example `answer: settlement:15`.

## Evaluation references

- Local: `data/audits/dpo_fair_results.md`,
  `data/audits/dpo_fair_eval_runbook.md`, and
  `data/audits/label_stability_200.json`.
- External: Anthropic's
  [Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents).

## GPU recovery

- Use givemeanode snapshot `snap-gu2np`, not RunPod. It contains the validated
  H100 environment, merged model, repository data, DPO adapter, and reward cache;
  use `~/venv312/bin/python`.
- Launch with `CUDA_HOME=/usr/local/cuda`, `/usr/local/cuda/bin` prepended to
  `PATH`, `PYTHONHASHSEED=0`, and `PYTHONPATH=/home/dev/catan-llm`.
- Run long jobs detached. Model load takes roughly 5-6 minutes and may be quiet;
  check GPU/process aliveness. Stop when done and snapshot after stopping if new
  state must persist.

## Operational cautions

- `deepseek-think` repeatedly produced `stop=user_closed` with empty traces.
  Diagnose API interception before spending; bad runs are in
  `outputs/_archived/netfail_aug2/`.
- Protect `outputs/` and `data/eval_traces/`: they contain irreplaceable,
  gitignored games with no off-machine backup.

## Working agreements

- No paid work, top-ups, or external contact without authorization. State the
  measured basis for cost estimates; nothing over $5, credits included, without
  a fresh yes.
- Do not edit `README.md`; put narrative drafts in chat. Do not commit, push,
  publish, or deploy without explicit instruction.
- Preserve the dirty worktree: check `git status --short` immediately before
  editing and avoid unrelated changes.
- Run relevant focused tests, then
  `PYTHONHASHSEED=0 .venv/bin/pytest -q tests/`.
