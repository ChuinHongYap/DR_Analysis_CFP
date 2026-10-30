import os
import json
import copy
import shutil
import itertools
import argparse
from typing import Dict, Any, List, Tuple
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

from train_model import run_training

# =====================================================================
# CONFIGURATION & HYPERPARAMETER SEARCH VARIABLES
# =====================================================================
SEED = 77
ROOT_DIR = "./MMRDR/MMRDR-CFP"
SPLITS_DIR = "./data_splits" #"./data_splits_DCP"
OUTPUT_DIR = "./outputs/hyperTuning"

SEARCH_STRATEGY = "greedy"            # Options: "greedy" or "grid" (can also override via --strategy CLI flag)
OBJECTIVE_METRIC = "neg_val_loss"              # Metric to maximise on val.csv: "qwk", "macro_f1", or "neg_val_loss"

IMG_SIZE = 512
CROP_BLACK_BORDER = True
BATCH_SIZE = 64
NUM_WORKERS = 4
TUNING_EPOCHS = 50                    # Epochs per candidate trial
WEIGHT_DECAY = 0.0001
FOCAL_GAMMA = 2.0

# Early Stopping settings (monitors validation loss)
EARLY_STOPPING_ENABLED = True
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_DELTA = 0.0005

# Search space candidates
SEARCH_SPACE = {
    "lr": [0.0001, 0.00001, 0.000001],
    "aux_lesion_weight": [0.0],
    "dropout": [0.25, 0.50],
    "loss_type": ["ce", "weighted_ce", "focal"],
}
# =====================================================================


def build_base_config() -> Dict[str, Any]:
    return {
        "seed": SEED,
        "root_dir": ROOT_DIR,
        "splits_dir": SPLITS_DIR,
        "img_size": IMG_SIZE,
        "crop_black_border": CROP_BLACK_BORDER,
        "num_workers": NUM_WORKERS,
        "pretrained": True,
        "num_classes": 5,
        "num_lesions": 7,
        "dropout": SEARCH_SPACE["dropout"][0],
        "batch_size": BATCH_SIZE,
        "epochs": TUNING_EPOCHS,
        "lr": SEARCH_SPACE["lr"][0],
        "weight_decay": WEIGHT_DECAY,
        "loss_type": SEARCH_SPACE["loss_type"][0],
        "focal_gamma": FOCAL_GAMMA,
        "aux_lesion_weight": SEARCH_SPACE["aux_lesion_weight"][0],
        "selection_metric": "qwk" if OBJECTIVE_METRIC != "neg_val_loss" else "val_loss",
        "es_enabled": EARLY_STOPPING_ENABLED,
        "es_patience": EARLY_STOPPING_PATIENCE,
        "es_min_delta": EARLY_STOPPING_MIN_DELTA,
    }


def extract_score(summary: Dict[str, Any], metric: str) -> float:
    if metric == "neg_val_loss":
        return -float(summary.get("val_loss", float("inf")))
    return float(summary.get(metric, -float("inf")))


def run_grid_search(base_cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], str, List[Dict[str, Any]]]:
    keys = list(SEARCH_SPACE.keys())
    combinations = list(itertools.product(*(SEARCH_SPACE[k] for k in keys)))
    print(f"=== Starting GRID SEARCH across {len(combinations)} trials ===")

    best_score = -float("inf")
    best_params: Dict[str, Any] = {}
    best_trial_dir = ""
    records: List[Dict[str, Any]] = []

    for idx, values in enumerate(combinations, start=1):
        params = dict(zip(keys, values))
        trial_name = f"grid_t{idx:02d}_" + "_".join(f"{k}-{v}" for k, v in params.items())
        trial_dir = os.path.join(OUTPUT_DIR, trial_name)

        print(f"\n[Grid Trial {idx}/{len(combinations)}] Testing: {params}")
        trial_cfg = copy.deepcopy(base_cfg)
        trial_cfg.update(params)

        summary = run_training(trial_cfg, exp_dir=trial_dir)
        score = extract_score(summary, OBJECTIVE_METRIC)

        record = {
            "trial_id": idx,
            "strategy": "grid",
            **params,
            "val_loss": round(summary.get("val_loss", 0.0), 4),
            "val_accuracy": round(summary.get("accuracy", 0.0), 4),
            "val_macro_f1": round(summary.get("macro_f1", 0.0), 4),
            "val_qwk": round(summary.get("qwk", 0.0), 4),
            "best_epoch": summary.get("best_epoch", 0),
            "objective_score": round(score, 4),
            "trial_dir": trial_dir,
        }
        records.append(record)

        if score > best_score:
            best_score = score
            best_params = copy.deepcopy(params)
            best_trial_dir = trial_dir
            print(f"  *** New Best Grid Trial ({OBJECTIVE_METRIC} = {best_score:.4f}): {best_params} ***")

    return best_params, best_trial_dir, records


def run_greedy_search(base_cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], str, List[Dict[str, Any]]]:
    keys = list(SEARCH_SPACE.keys())
    current_best_params = {k: base_cfg[k] for k in keys}
    print(f"=== Starting GREEDY SEARCH across {len(keys)} hyperparameters ===")
    print(f"Initial parameter state: {current_best_params}")

    evaluated_cache: Dict[Tuple[Tuple[str, Any], ...], Dict[str, Any]] = {}
    records: List[Dict[str, Any]] = []
    best_score = -float("inf")
    best_trial_dir = ""
    trial_counter = 0

    for param_name in keys:
        candidates = SEARCH_SPACE[param_name]
        print(f"\n--- Optimising '{param_name}' over candidates: {candidates} ---")
        best_val_for_param = current_best_params[param_name]

        for val in candidates:
            candidate_params = copy.deepcopy(current_best_params)
            candidate_params[param_name] = val
            cache_key = tuple(sorted(candidate_params.items()))

            if cache_key in evaluated_cache:
                print(f"  Skipping already-evaluated combination: {candidate_params}")
                continue

            trial_counter += 1
            trial_name = f"greedy_t{trial_counter:02d}_{param_name}-{val}"
            trial_dir = os.path.join(OUTPUT_DIR, trial_name)

            print(f"\n[Greedy Trial {trial_counter}] Testing {param_name}={val} | State: {candidate_params}")
            trial_cfg = copy.deepcopy(base_cfg)
            trial_cfg.update(candidate_params)

            summary = run_training(trial_cfg, exp_dir=trial_dir)
            score = extract_score(summary, OBJECTIVE_METRIC)

            record = {
                "trial_id": trial_counter,
                "strategy": "greedy",
                "tuned_param": param_name,
                **candidate_params,
                "val_loss": round(summary.get("val_loss", 0.0), 4),
                "val_accuracy": round(summary.get("accuracy", 0.0), 4),
                "val_macro_f1": round(summary.get("macro_f1", 0.0), 4),
                "val_qwk": round(summary.get("qwk", 0.0), 4),
                "best_epoch": summary.get("best_epoch", 0),
                "objective_score": round(score, 4),
                "trial_dir": trial_dir,
            }
            evaluated_cache[cache_key] = record
            records.append(record)

            if score > best_score:
                best_score = score
                best_val_for_param = val
                best_trial_dir = trial_dir
                print(f"  *** Improved '{param_name}' -> {val} ({OBJECTIVE_METRIC} = {best_score:.4f}) ***")

        current_best_params[param_name] = best_val_for_param
        print(f"Locked '{param_name}' = {best_val_for_param}")

    return current_best_params, best_trial_dir, records


def main(strategy: str) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base_cfg = build_base_config()

    if strategy == "grid":
        best_params, best_trial_dir, records = run_grid_search(base_cfg)
    elif strategy == "greedy":
        best_params, best_trial_dir, records = run_greedy_search(base_cfg)
    else:
        raise ValueError(f"Unsupported strategy '{strategy}'. Choose 'grid' or 'greedy'.")

    # 1. Save leaderboard CSV
    results_df = pd.DataFrame(records).sort_values("objective_score", ascending=False)
    leaderboard_csv = os.path.join(OUTPUT_DIR, f"tuning_leaderboard_{strategy}.csv")
    results_df.to_csv(leaderboard_csv, index=False)

    # 2. Save winning best_model.pt and training_history.json directly in OUTPUT_DIR
    if best_trial_dir and os.path.exists(best_trial_dir):
        for fname in ["best_model.pt", "training_history.json"]:
            src_file = os.path.join(best_trial_dir, fname)
            if os.path.exists(src_file):
                shutil.copy2(src_file, os.path.join(OUTPUT_DIR, fname))

    # 3. Save best_hyperparameters.json so train_model.py can load it automatically
    best_params_path = os.path.join(OUTPUT_DIR, "best_hyperparameters.json")
    with open(best_params_path, "w") as f:
        json.dump(best_params, f, indent=2)

    # 4. Log summary to TensorBoard HParams
    tb_hparams_dir = os.path.join(OUTPUT_DIR, "tensorboard")
    with SummaryWriter(log_dir=tb_hparams_dir) as writer:
        for rec in records:
            hparam_dict = {k: rec[k] for k in SEARCH_SPACE.keys()}
            metric_dict = {
                "hparam/val_qwk": rec["val_qwk"],
                "hparam/val_macro_f1": rec["val_macro_f1"],
                "hparam/val_loss": rec["val_loss"],
            }
            writer.add_hparams(hparam_dict, metric_dict)

    print("\n" + "=" * 70)
    print(f"HYPERPARAMETER TUNING COMPLETE ({strategy.upper()} SEARCH)")
    print("=" * 70)
    print(f"Best Hyperparameters:    {best_params}")
    print(f"Saved Best Checkpoint:   {os.path.join(OUTPUT_DIR, 'best_model.pt')}")
    print(f"Saved Training History:  {os.path.join(OUTPUT_DIR, 'training_history.json')}")
    print(f"Saved TensorBoard Logs:  {tb_hparams_dir}")
    print(f"Saved Best Params JSON:  {best_params_path}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter tuning with Grid or Greedy search.")
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["grid", "greedy"],
        default=SEARCH_STRATEGY,
        help="Search strategy: 'greedy' or 'grid'",
    )
    args = parser.parse_args()
    main(args.strategy)