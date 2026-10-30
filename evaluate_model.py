import os
import json
from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    cohen_kappa_score,
    confusion_matrix,
)

from train_model import MMRDRCFPDataset, DRGradingResNet50, get_transforms, LESION_NAMES

# =====================================================================
# CONFIGURATION VARIABLES
# =====================================================================
ROOT_DIR = "./MMRDR/MMRDR-CFP"
SPLITS_DIR = "./data_splits" #"./data_splits_dcp"
CHECKPOINT_PATH = "./outputs/resnet50_weightedCE/best_model.pt"
EVAL_OUTPUT_DIR = "./outputs/resnet50_weightedCE/evaluation"

IMG_SIZE = 512
CROP_BLACK_BORDER = True
BATCH_SIZE = 64
NUM_WORKERS = 4
NUM_CLASSES = 5
NUM_LESIONS = 7
DROPOUT = 0.25
# =====================================================================

GRADE_NAMES = [
    "Grade 0: No DR",
    "Grade 1: Mild NPDR",
    "Grade 2: Moderate NPDR",
    "Grade 3: Severe NPDR",
    "Grade 4: PDR",
]


def safe_load_checkpoint(ckpt_path: str, device: torch.device) -> Dict[str, Any]:
    try:
        return torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(ckpt_path, map_location=device)


@torch.no_grad()
def collect_outputs(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, Any]:
    model.eval()
    grade_logits_list, lesion_logits_list = [], []
    grades_list, lesions_list, lr_list = [], [], []
    image_ids, image_paths = [], []

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        outputs = model(images)

        grade_logits_list.append(outputs["logits_grade"].cpu())
        lesion_logits_list.append(outputs["logits_lesion"].cpu())
        grades_list.append(batch["grade"])
        lesions_list.append(batch["lesions"])
        lr_list.append(batch["lr"])
        image_ids.extend(batch["image_id"])
        image_paths.extend(batch["image_path"])

    return {
        "grade_logits": torch.cat(grade_logits_list, dim=0),
        "lesion_logits": torch.cat(lesion_logits_list, dim=0),
        "grades": torch.cat(grades_list, dim=0).numpy(),
        "lesions": torch.cat(lesions_list, dim=0).numpy(),
        "lr": torch.cat(lr_list, dim=0).numpy(),
        "image_ids": image_ids,
        "image_paths": image_paths,
    }


def fit_temperature(val_logits: torch.Tensor, val_targets: np.ndarray) -> float:
    targets_tensor = torch.tensor(val_targets, dtype=torch.long)
    log_temp = nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.LBFGS([log_temp], lr=0.05, max_iter=100)

    def eval_loss() -> torch.Tensor:
        optimizer.zero_grad()
        temp = torch.exp(log_temp).clamp(min=1e-3, max=100.0)
        loss = F.cross_entropy(val_logits / temp, targets_tensor)
        loss.backward()
        return loss

    optimizer.step(eval_loss)
    return float(torch.exp(log_temp).item())


def compute_ece(
    probs: np.ndarray, targets: np.ndarray, n_bins: int = 10
) -> Tuple[float, Dict[str, List[float]]]:
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == targets).astype(np.float32)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_accs, bin_confs, bin_counts = [], [], []

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            acc_in_bin = float(np.mean(accuracies[in_bin]))
            conf_in_bin = float(np.mean(confidences[in_bin]))
            ece += np.abs(acc_in_bin - conf_in_bin) * prop_in_bin
            bin_accs.append(acc_in_bin)
            bin_confs.append(conf_in_bin)
            bin_counts.append(int(np.sum(in_bin)))
        else:
            bin_accs.append(0.0)
            bin_confs.append(float((bin_boundaries[i] + bin_boundaries[i + 1]) / 2.0))
            bin_counts.append(0)

    return float(ece), {"bin_accs": bin_accs, "bin_confs": bin_confs, "bin_counts": bin_counts}


def evaluate_quality_sensitivity(
    model: nn.Module, test_csv: str, temperature: float, device: torch.device
) -> Dict[str, Any]:
    import cv2
    degraded_transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE, interpolation=cv2.INTER_AREA),
        A.GaussianBlur(blur_limit=(7, 7), sigma_limit=(2.0, 2.0), p=1.0),
        A.RandomBrightnessContrast(brightness_limit=(-0.25, -0.25), contrast_limit=(-0.20, -0.20), p=1.0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    deg_ds = MMRDRCFPDataset(
        test_csv, ROOT_DIR, transforms=degraded_transform, crop_border=CROP_BLACK_BORDER
    )
    deg_loader = DataLoader(deg_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    deg_out = collect_outputs(model, deg_loader, device)
    probs = F.softmax(deg_out["grade_logits"] / temperature, dim=1).numpy()
    preds = np.argmax(probs, axis=1)
    targets = deg_out["grades"]
    entropies = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=1)

    return {
        "degraded_accuracy": round(float(accuracy_score(targets, preds)), 4),
        "degraded_macro_f1": round(float(f1_score(targets, preds, average="macro", zero_division=0)), 4),
        "degraded_qwk": round(float(cohen_kappa_score(targets, preds, weights="quadratic")), 4),
        "degraded_mean_entropy": round(float(np.mean(entropies)), 4),
    }


def plot_confusion_matrix(cm: np.ndarray, save_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    short_labels = ["G0: No DR", "G1: Mild", "G2: Mod", "G3: Severe", "G4: PDR"]

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=short_labels,
        yticklabels=short_labels, ax=axes[0], cbar=False
    )
    axes[0].set_title("Confusion Matrix (Test Set Counts)")
    axes[0].set_xlabel("Predicted DR Grade")
    axes[0].set_ylabel("True DR Grade")

    cm_norm = cm.astype(np.float32) / np.clip(cm.sum(axis=1, keepdims=True), a_min=1, a_max=None)
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=short_labels,
        yticklabels=short_labels, ax=axes[1], cbar=True, vmin=0.0, vmax=1.0
    )
    axes[1].set_title("Confusion Matrix (Row-Normalised Sensitivity)")
    axes[1].set_xlabel("Predicted DR Grade")
    axes[1].set_ylabel("True DR Grade")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_reliability_and_referral(
    uncal_stats: Dict[str, List[float]],
    cal_stats: Dict[str, List[float]],
    ece_before: float,
    ece_after: float,
    entropies: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str,
) -> Dict[str, List[float]]:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    axes[0].plot([0, 1], [0, 1], "--", color="gray", label="Ideal Calibration")
    uncal_mask = np.array(uncal_stats["bin_counts"]) > 0
    cal_mask = np.array(cal_stats["bin_counts"]) > 0

    axes[0].plot(
        np.array(uncal_stats["bin_confs"])[uncal_mask],
        np.array(uncal_stats["bin_accs"])[uncal_mask],
        "o-", color="#d95f02", label=f"Uncalibrated (ECE = {ece_before:.4f})"
    )
    axes[0].plot(
        np.array(cal_stats["bin_confs"])[cal_mask],
        np.array(cal_stats["bin_accs"])[cal_mask],
        "s-", color="#1b9e77", label=f"Temp-Scaled (ECE = {ece_after:.4f})"
    )
    axes[0].set_title("Confidence Calibration (Reliability Diagram)")
    axes[0].set_xlabel("Mean Predicted Confidence")
    axes[0].set_ylabel("Empirical Test Accuracy")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="lower right")

    referral_fractions = np.linspace(0.0, 0.40, 9)
    retained_qwk, retained_acc, critical_fn_rate = [], [], []
    order = np.argsort(entropies)

    for frac in referral_fractions:
        keep_n = max(1, int(len(y_true) * (1.0 - frac)))
        keep_idx = order[:keep_n]
        yt, yp = y_true[keep_idx], y_pred[keep_idx]

        retained_acc.append(float(accuracy_score(yt, yp)))
        retained_qwk.append(float(cohen_kappa_score(yt, yp, weights="quadratic")))
        severe_mask = yt >= 3
        fn_rate = float(np.mean(yp[severe_mask] <= 1)) if np.any(severe_mask) else 0.0
        critical_fn_rate.append(fn_rate)

    ax2 = axes[1]
    pct_referred = (referral_fractions * 100).tolist()
    ax2.plot(pct_referred, retained_qwk, "o-", color="#2b5c8f", label="Retained QWK")
    ax2.plot(pct_referred, retained_acc, "s--", color="#388e3c", label="Retained Accuracy")
    ax2.set_xlabel("Uncertain Cases Referred to Clinician (%)")
    ax2.set_ylabel("Performance on Automated Cases")
    ax2.set_title("Selective Triage via Predictive Entropy")
    ax2.grid(alpha=0.3)

    ax2_twin = ax2.twinx()
    ax2_twin.plot(
        pct_referred, [r * 100 for r in critical_fn_rate],
        "^:", color="#c62828", label="Sight-Threatening False Negative (%)"
    )
    ax2_twin.set_ylabel("Severe DR Under-Graded as G0/G1 (%)", color="#c62828")
    ax2_twin.tick_params(axis="y", labelcolor="#c62828")

    lines_1, labels_1 = ax2.get_legend_handles_labels()
    lines_2, labels_2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc="center left")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "referral_pct": [round(x, 1) for x in pct_referred],
        "retained_qwk": [round(x, 4) for x in retained_qwk],
        "retained_accuracy": [round(x, 4) for x in retained_acc],
        "sight_threatening_fn_rate": [round(x, 4) for x in critical_fn_rate],
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = CHECKPOINT_PATH
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Could not find checkpoint at {CHECKPOINT_PATH}. "
            "Run train_model.py or train_hyperTuning.py first."
        )

    os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)
    val_csv = os.path.join(SPLITS_DIR, "val.csv")
    test_csv = os.path.join(SPLITS_DIR, "test.csv")

    val_ds = MMRDRCFPDataset(
        val_csv, ROOT_DIR,
        transforms=get_transforms(IMG_SIZE, is_train=False),
        crop_border=CROP_BLACK_BORDER,
    )
    test_ds = MMRDRCFPDataset(
        test_csv, ROOT_DIR,
        transforms=get_transforms(IMG_SIZE, is_train=False),
        crop_border=CROP_BLACK_BORDER,
    )

    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = DRGradingResNet50(
        num_classes=NUM_CLASSES, num_lesions=NUM_LESIONS, pretrained=False, dropout=DROPOUT
    ).to(device)

    ckpt = safe_load_checkpoint(ckpt_path, device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint (epoch {ckpt['epoch']}): {ckpt_path}")

    # 1. Fit Temperature Scaling on Validation Split
    val_out = collect_outputs(model, val_loader, device)
    optimal_temp = fit_temperature(val_out["grade_logits"], val_out["grades"])
    val_probs_cal = F.softmax(val_out["grade_logits"] / optimal_temp, dim=1).numpy()
    val_entropies = -np.sum(val_probs_cal * np.log(np.clip(val_probs_cal, 1e-12, 1.0)), axis=1)
    entropy_threshold = float(np.percentile(val_entropies, 80))

    # 2. Evaluate on Locked Test Split
    test_out = collect_outputs(model, test_loader, device)
    y_true = test_out["grades"]
    logits = test_out["grade_logits"]

    probs_uncal = F.softmax(logits, dim=1).numpy()
    probs_cal = F.softmax(logits / optimal_temp, dim=1).numpy()
    y_pred = np.argmax(probs_cal, axis=1)
    confidences_cal = np.max(probs_cal, axis=1)
    entropies = -np.sum(probs_cal * np.log(np.clip(probs_cal, 1e-12, 1.0)), axis=1)

    acc_g = float(accuracy_score(y_true, y_pred))
    f1_g = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    qwk = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    prec, rec, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(5)), zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(5)))

    lesion_probs = torch.sigmoid(test_out["lesion_logits"]).numpy()
    lesion_preds = (lesion_probs >= 0.5).astype(np.int32)
    lesion_true = test_out["lesions"].astype(np.int32)
    acc_l = float(np.mean(lesion_preds == lesion_true))
    f1_l = float(f1_score(lesion_true, lesion_preds, average="macro", zero_division=0))

    ece_before, uncal_stats = compute_ece(probs_uncal, y_true)
    ece_after, cal_stats = compute_ece(probs_cal, y_true)

    plot_confusion_matrix(cm, os.path.join(EVAL_OUTPUT_DIR, "confusion_matrix.png"))
    referral_metrics = plot_reliability_and_referral(
        uncal_stats, cal_stats, ece_before, ece_after,
        entropies, y_true, y_pred, os.path.join(EVAL_OUTPUT_DIR, "reliability_and_referral.png")
    )
    quality_sensitivity = evaluate_quality_sensitivity(model, test_csv, optimal_temp, device)

    pred_records = []
    for i in range(len(y_true)):
        true_g, pred_g = int(y_true[i]), int(y_pred[i])
        err_diff = pred_g - true_g
        if true_g == pred_g:
            error_type = "correct"
        elif true_g >= 3 and pred_g <= 1:
            error_type = "critical_underestimation"
        elif true_g >= 2 and pred_g <= 1:
            error_type = "missed_referable_dr"
        elif err_diff < 0:
            error_type = "mild_underestimation"
        else:
            error_type = "overestimation"

        present_lesions = [LESION_NAMES[j] for j in range(7) if lesion_true[i, j] == 1]
        pred_lesions = [LESION_NAMES[j] for j in range(7) if lesion_preds[i, j] == 1]

        pred_records.append({
            "image_id": test_out["image_ids"][i],
            "image_path": test_out["image_paths"][i],
            "lr": int(test_out["lr"][i]),
            "true_grade": true_g,
            "pred_grade": pred_g,
            "grade_diff": err_diff,
            "error_type": error_type,
            "uncalibrated_conf": round(float(np.max(probs_uncal[i])), 4),
            "calibrated_conf": round(float(confidences_cal[i]), 4),
            "predictive_entropy": round(float(entropies[i]), 4),
            "needs_clinical_review": bool(entropies[i] >= entropy_threshold),
            "true_lesions": "; ".join(present_lesions) if present_lesions else "None",
            "pred_lesions": "; ".join(pred_lesions) if pred_lesions else "None",
        })

    pred_df = pd.DataFrame(pred_records)
    pred_df.to_csv(os.path.join(EVAL_OUTPUT_DIR, "test_predictions.csv"), index=False)

    summary = {
        "checkpoint_evaluated": ckpt_path,
        "overall_metrics": {
            "Acc_G": round(acc_g, 4),
            "F1_G_macro": round(f1_g, 4),
            "QWK": round(qwk, 4),
            "Acc_L_lesion": round(acc_l, 4),
            "F1_L_lesion_macro": round(f1_l, 4),
        },
        "per_class_metrics": {
            GRADE_NAMES[c]: {
                "precision": round(float(prec[c]), 4),
                "recall": round(float(rec[c]), 4),
                "f1": round(float(f1_per_class[c]), 4),
                "support": int(support[c]),
            }
            for c in range(5)
        },
        "calibration": {
            "optimal_temperature": round(optimal_temp, 4),
            "ece_uncalibrated": round(ece_before, 4),
            "ece_temperature_scaled": round(ece_after, 4),
            "clean_mean_entropy": round(float(np.mean(entropies)), 4),
            "val_80pct_entropy_threshold": round(entropy_threshold, 4),
        },
        "quality_degradation_sensitivity": quality_sensitivity,
        "error_breakdown_counts": pred_df["error_type"].value_counts().to_dict(),
        "referral_simulation": referral_metrics,
        "confusion_matrix": cm.tolist(),
    }

    with open(os.path.join(EVAL_OUTPUT_DIR, "test_metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 68)
    print(f"TEST SET EVALUATION COMPLETE ({len(y_true)} CFP images)")
    print("=" * 68)
    print(f"Accuracy (Acc_G):        {acc_g:.4f}  (MMRDR Paper ResNet-50: 0.823)")
    print(f"Macro-F1 (F1_G):         {f1_g:.4f}  (MMRDR Paper ResNet-50: 0.702)")
    print(f"Quadratic Kappa (QWK):   {qwk:.4f}")
    print(f"Lesion Acc / Macro-F1:   {acc_l:.4f} / {f1_l:.4f}")
    print(f"Calibration ECE:         {ece_before:.4f} (Uncal) --> {ece_after:.4f} (Temp T={optimal_temp:.2f})")
    print(
        f"Quality Degradation:     QWK {qwk:.4f} --> {quality_sensitivity['degraded_qwk']:.4f} | "
        f"Mean Entropy {np.mean(entropies):.3f} --> {quality_sensitivity['degraded_mean_entropy']:.3f}"
    )
    print("=" * 68)
    print(f"Saved evaluation artefacts to: {EVAL_OUTPUT_DIR}/")


if __name__ == "__main__":
    main()