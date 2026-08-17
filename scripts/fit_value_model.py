"""Fits a LightGBM win-probability model on the value corpus."""

import argparse
import json
import re
import subprocess
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_data_in_leaf": 200,
    "seed": 0,
    "deterministic": True,
    "force_row_wise": True,
    "verbosity": -1,
}
MAX_ROUNDS = 4000
EARLY_STOPPING = 100

TURN_BUCKETS = [(0, 10), (10, 20), (20, 30), (30, 1000)]
VP_BUCKETS = [(0, 5), (5, 7), (7, 9), (9, 12)]


def bucket_metrics(y, pred, key, buckets, label):
    parts = []
    for lo, hi in buckets:
        mask = (key >= lo) & (key < hi)
        if mask.sum() < 100 or len(set(y[mask].tolist())) < 2:
            continue
        parts.append(f"{lo}-{hi}: AUC {roc_auc_score(y[mask], pred[mask]):.3f}")
    print(f"by {label}:  " + " | ".join(parts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="experiments/tier1_value_exit/corpus")
    parser.add_argument("--out", default="experiments/tier1_value_exit/value_model")
    args = parser.parse_args()

    corpus = Path(args.corpus)
    manifest = json.loads((corpus / "manifest.json").read_text())
    feature_names = manifest["feature_names"]
    lgb_names = [re.sub(r"[^0-9A-Za-z_]", "_", name) for name in feature_names]
    train = np.load(corpus / "train.npz")
    val = np.load(corpus / "val.npz")
    print(f"train {len(train['won'])} rows, val {len(val['won'])} rows, {len(feature_names)} features")

    model = lgb.train(
        PARAMS,
        lgb.Dataset(train["X"], label=train["won"], feature_name=lgb_names),
        num_boost_round=MAX_ROUNDS,
        valid_sets=[lgb.Dataset(val["X"], label=val["won"], feature_name=lgb_names)],
        callbacks=[lgb.early_stopping(EARLY_STOPPING), lgb.log_evaluation(200)],
    )

    pred = model.predict(val["X"], num_iteration=model.best_iteration)
    y = val["won"]
    metrics = {
        "logloss": round(log_loss(y, pred), 4),
        "auc": round(roc_auc_score(y, pred), 4),
        "brier": round(brier_score_loss(y, pred), 4),
        "best_iteration": model.best_iteration,
    }
    print(
        f"best iteration {metrics['best_iteration']} | val logloss {metrics['logloss']} "
        f"| AUC {metrics['auc']} | Brier {metrics['brier']}"
    )
    bucket_metrics(y, pred, val["turn"], TURN_BUCKETS, "turn")
    hero_vp = val["X"][:, feature_names.index("P0_ACTUAL_VPS")]
    bucket_metrics(y, pred, hero_vp, VP_BUCKETS, "hero VP")

    gain = model.feature_importance("gain")
    top = sorted(zip(gain, feature_names), reverse=True)[:20]
    print("top gain:", ", ".join(name for _, name in top))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(out_dir / "model.txt", num_iteration=model.best_iteration)
    git = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    fit_manifest = {
        "corpus": str(corpus),
        "corpus_git": manifest["git_commit"],
        "git_commit": git,
        "params": PARAMS,
        "max_rounds": MAX_ROUNDS,
        "early_stopping": EARLY_STOPPING,
        "metrics": metrics,
        "top_gain": [name for _, name in top],
    }
    (out_dir / "fit_manifest.json").write_text(json.dumps(fit_manifest, indent=2))
    print(f"wrote {out_dir}/model.txt + fit_manifest.json")


if __name__ == "__main__":
    main()
