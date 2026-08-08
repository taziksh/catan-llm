"""Upload a generated replay export to Cloudflare R2 and Pages."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from scripts.export_replay_site import write_site_shell


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPORT = ROOT / "data" / "replay_cloudflare"
DEFAULT_BUCKET = "catan-llm-replays"
DEFAULT_PROJECT = "catan-llm-replays"
CLOUDFLARE_ROOT = ROOT / "cloudflare" / "replay_site"
WRANGLER = ("npx", "--yes", "wrangler@4.120.0")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--skip-r2", action="store_true")
    parser.add_argument("--skip-pages", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_verified_objects(export_root: Path) -> list[tuple[dict, Path]]:
    report_path = export_root / "export-report.json"
    if not (export_root / ".catan-replay-export").is_file():
        raise ValueError(f"not a generated replay export: {export_root}")
    report = json.loads(report_path.read_text())
    r2_root = (export_root / "r2").resolve()
    verified = []
    seen_keys = set()
    for item in report["objects"]:
        key = item["key"]
        if not isinstance(key, str) or not key or key in seen_keys:
            raise ValueError(f"invalid or duplicate object key: {key!r}")
        path = (r2_root / key).resolve()
        if not path.is_relative_to(r2_root):
            raise ValueError(f"unsafe object key: {key!r}")
        payload = path.read_bytes()
        if len(payload) != item["bytes"]:
            raise ValueError(f"size mismatch: {path}")
        if hashlib.sha256(payload).hexdigest() != item["sha256"]:
            raise ValueError(f"hash mismatch: {path}")
        verified.append((item, path))
        seen_keys.add(key)
    return verified


def upload_command(bucket: str, item: dict, path: Path) -> list[str]:
    command = [
        *WRANGLER,
        "r2",
        "object",
        "put",
        f"{bucket}/{item['key']}",
        f"--file={path}",
        "--remote",
        "--content-type=application/json",
    ]
    if item["key"].endswith(".gz"):
        command.extend(
            [
                "--content-encoding=gzip",
                "--cache-control=public, max-age=31536000, immutable",
            ]
        )
    else:
        command.append("--cache-control=public, max-age=300")
    return command


def prepare_wrangler(dry_run: bool) -> None:
    """Populate npx's package cache before parallel upload workers start."""
    if dry_run:
        return
    subprocess.run(
        [*WRANGLER, "--version"],
        cwd=CLOUDFLARE_ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )


def upload_one(bucket: str, item: dict, path: Path, dry_run: bool) -> str:
    command = upload_command(bucket, item, path)
    if not dry_run:
        for attempt in range(3):
            try:
                subprocess.run(
                    command,
                    cwd=CLOUDFLARE_ROOT,
                    check=True,
                    stdout=subprocess.DEVNULL,
                )
                break
            except subprocess.CalledProcessError:
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
    return item["key"]


def upload_r2(
    bucket: str,
    objects: list[tuple[dict, Path]],
    jobs: int,
    dry_run: bool,
) -> None:
    if jobs < 1:
        raise ValueError("--jobs must be positive")
    catalogs = [obj for obj in objects if obj[0]["key"] == "catalog.json"]
    if len(catalogs) != 1:
        raise ValueError("export must contain exactly one catalog.json object")
    content_objects = [obj for obj in objects if obj[0]["key"] != "catalog.json"]
    completed = 0
    verb = "planned" if dry_run else "uploaded"
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = [
            executor.submit(upload_one, bucket, item, path, dry_run)
            for item, path in content_objects
        ]
        for future in as_completed(futures):
            future.result()
            completed += 1
            if completed % 25 == 0:
                print(f"{verb} {completed}/{len(objects)} R2 objects", flush=True)
    catalog_item, catalog_path = catalogs[0]
    upload_one(bucket, catalog_item, catalog_path, dry_run)
    completed += 1
    print(f"{verb} {completed}/{len(objects)} R2 objects", flush=True)


def deploy_pages(export_root: Path, project: str, dry_run: bool) -> None:
    site_dir = export_root / "site"
    if not (site_dir / "viewer" / "index.html").is_file():
        raise ValueError(f"compiled replay viewer not found in {site_dir}")
    command = [
        *WRANGLER,
        "pages",
        "deploy",
        str(site_dir),
        f"--project-name={project}",
        "--branch=main",
    ]
    if dry_run:
        print("would deploy Pages assets and Functions")
        return
    write_site_shell(site_dir)
    subprocess.run(command, cwd=CLOUDFLARE_ROOT, check=True)


def main(args) -> None:
    export_root = args.export.resolve()
    objects = load_verified_objects(export_root)
    print(f"verified {len(objects)} local objects")
    prepare_wrangler(args.dry_run)
    if not args.skip_r2:
        upload_r2(args.bucket, objects, args.jobs, args.dry_run)
    if not args.skip_pages:
        deploy_pages(export_root, args.project, args.dry_run)


if __name__ == "__main__":
    main(parse_args())
