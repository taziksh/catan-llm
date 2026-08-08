# Hosted replay browser

Read-only Cloudflare Pages + R2 site. Production: <https://catan.tazik.sh>;
catalog: <https://catan.tazik.sh/replays>. Pages serves the UI and two Functions
read `catalog.json` and compressed game bundles from R2.

## Run

Requires `../catanatron/ui` with dependencies installed, `yarn`, authenticated
Wrangler, and the existing `catan-llm-replays` Pages project and R2 bucket.

```sh
PYTHONHASHSEED=0 .venv/bin/python -m scripts.export_replay_site
PYTHONHASHSEED=0 .venv/bin/python -m scripts.preview_replay_site
PYTHONHASHSEED=0 .venv/bin/python -m scripts.deploy_replay_site --dry-run
PYTHONHASHSEED=0 .venv/bin/python -m scripts.deploy_replay_site
```

Preview at <http://127.0.0.1:5002>. Use `--skip-r2` or `--skip-pages` for a
partial deploy; use each command's `--help` for filtering and overrides.

## Guards

- Export excludes malformed, unfinished, diagnostic, and unlabeled games by
  default. It never changes the source database.
- Generated files live in ignored `data/replay_cloudflare/`.
- Deploy verifies hashes, uploads games before the catalog, and refuses to
  publish Pages without the compiled viewer. It does not manage DNS.

## Test

```sh
PYTHONHASHSEED=0 .venv/bin/pytest -q tests/test_{export,preview}_replay_site.py
node --test cloudflare/replay_site/functions/functions.test.mjs
```
