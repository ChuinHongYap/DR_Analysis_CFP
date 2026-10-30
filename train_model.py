import os
import re
import json
import random
from typing import Dict, Any, Optional
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet50, ResNet50_Weights
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix

# =====================================================================
# CONFIGURATION & BEST MODEL HYPERPARAMETERS
# =====================================================================
SEED = 77
ROOT_DIR = "./MMRDR/MMRDR-CFP"
SPLITS_DIR = "./data_splits"
OUTPUT_DIR = "./outputs/resnet50_weightedCE"

IMG_SIZE = 512                        # Matches 512x512 benchmark in Tang et al. (2026)
CROP_BLACK_BORDER = True
NUM_WORKERS = 4

PRETRAINED = True
NUM_CLASSES = 5                       # DR Grades 0 to 4
NUM_LESIONS = 7                       # MA, HE, IH, VB/IRMA, NV, VH, RD
DROPOUT = 0.25

BATCH_SIZE = 64
EPOCHS = 150
LR = 0.0001
WEIGHT_DECAY = 0.0001
LOSS_TYPE = "weighted_ce"             # Options: "ce" (baseline), "weighted_ce", "focal"
FOCAL_GAMMA = 2.0
AUX_LESION_WEIGHT = 0.0               # Set to 0.0 for DR grading only, or >0.0 to include auxiliary lesion loss in total loss
SELECTION_METRIC = "val_loss"         # Options: "qwk", "macro_f1", or "val_loss"

# Early Stopping settings (monitors validation loss)
EARLY_STOPPING_ENABLED = True
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_DELTA = 0.00001
# =====================================================================

LESION_NAMES = [
    "Microaneurysm",
    "Hard exudate",
    "Intraretinal Hemorrhage",
    "VB/IRMA",
    "Neovascularization",
    "Vitreous hemorrhage",
    "Retinal detachment",
]


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def crop_fundus_roi(image: np.ndarray, tol: int = 7) -> np.ndarray:
    """Crops the uninformative dark border around circular fundus photographs."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    if not np.any(mask):
        return image
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return image[y0:y1, x0:x1]


def parse_lesion_vector(lesion_str: Any, num_lesions: int = 7) -> np.ndarray:
    """Extracts the 7-element binary lesion array from CSV string format."""
    digits = [int(x) for x in re.findall(r"[01]", str(lesion_str))]
    if len(digits) != num_lesions:
        raise ValueError(f"Expected {num_lesions} binary lesion flags, got: {lesion_str}")
    return np.array(digits, dtype=np.float32)


def get_transforms(img_size: int = 512, is_train: bool = True) -> A.Compose:
    if is_train:
        return A.Compose([
            A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.12, contrast_limit=0.12, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


class MMRDRCFPDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        root_dir: str,
        transforms: Optional[A.Compose] = None,
        crop_border: bool = True,
    ) -> None:
        self.df = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transforms = transforms
        self.crop_border = crop_border

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_image_path(self, rel_path: str) -> str:
        candidate_1 = os.path.join(self.root_dir, rel_path)
        if os.path.exists(candidate_1):
            return candidate_1
        candidate_2 = os.path.join(self.root_dir, "img", os.path.basename(rel_path))
        if os.path.exists(candidate_2):
            return candidate_2
        raise FileNotFoundError(f"Image not found at {candidate_1} or {candidate_2}")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        img_path = self._resolve_image_path(str(row["image"]))

        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            raise IOError(f"Failed to read image: {img_path}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        if self.crop_border:
            image = crop_fundus_roi(image)

        if self.transforms is not None:
            image_tensor = self.transforms(image=image)["image"]
        else:
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        grade = int(row["grade"])
        lesions = parse_lesion_vector(row["lesion"])
        laterality = int(row["lr"]) if "lr" in row and not pd.isna(row["lr"]) else -1

        return {
            "image": image_tensor,
            "grade": torch.tensor(grade, dtype=torch.long),
            "lesions": torch.from_numpy(lesions),
            "lr": torch.tensor(laterality, dtype=torch.long),
            "image_id": os.path.basename(img_path),
            "image_path": img_path,
        }


class DRGradingResNet50(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        num_lesions: int = 7,
        pretrained: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = resnet50(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.dropout = nn.Dropout(p=dropout)
        self.grade_head = nn.Linear(in_features, num_classes)
        self.lesion_head = nn.Linear(in_features, num_lesions)

    def get_target_layer(self) -> nn.Module:
        return self.backbone.layer4[-1]

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.dropout(self.backbone(x))
        return {
            "logits_grade": self.grade_head(features),
            "logits_lesion": self.lesion_head(features),
        }


class MultiClassFocalLoss(nn.Module):
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce_loss)
        return (((1.0 - pt) ** self.gamma) * ce_loss).mean()


class EarlyStopping:
    """Monitors validation loss and triggers early stopping when improvement stalls."""
    def __init__(self, patience: int = 6, min_delta: float = 0.0005) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < (self.best_loss - self.min_delta):
            self.best_loss = val_loss
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True
        return False


def compute_class_weights(csv_path: str, num_classes: int = 5) -> torch.Tensor:
    df = pd.read_csv(csv_path)
    counts = df["grade"].value_counts().reindex(range(num_classes), fill_value=1).values
    total = counts.sum()
    weights = np.sqrt(total / (num_classes * counts.astype(np.float32)))
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def build_criterion(
    loss_type: str, class_weights: torch.Tensor, focal_gamma: float, device: torch.device
) -> nn.Module:
    if loss_type == "ce":
        return nn.CrossEntropyLoss()
    if loss_type == "weighted_ce":
        return nn.CrossEntropyLoss(weight=class_weights.to(device))
    if loss_type == "focal":
        return MultiClassFocalLoss(alpha=class_weights.to(device), gamma=focal_gamma)
    raise ValueError(f"Unsupported loss_type: {loss_type}")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    grade_criterion: nn.Module,
    lesion_criterion: nn.Module,
    aux_weight: float,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
) -> Dict[str, float]:
    model.train()
    total_loss_sum, grade_loss_sum = 0.0, 0.0

    for batch in tqdm(loader, desc="Train", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        grades = batch["grade"].to(device, non_blocking=True)
        lesions = batch["lesions"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            outputs = model(images)
            loss_grade = grade_criterion(outputs["logits_grade"], grades)
            if aux_weight > 0.0:
                loss_lesion = lesion_criterion(outputs["logits_lesion"], lesions)
                loss = loss_grade + aux_weight * loss_lesion
            else:
                loss = loss_grade

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss_sum += loss.item() * images.size(0)
        grade_loss_sum += loss_grade.item() * images.size(0)

    n = len(loader.dataset)
    return {"train_loss": total_loss_sum / n, "train_grade_loss": grade_loss_sum / n}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    grade_criterion: nn.Module,
    lesion_criterion: nn.Module,
    aux_weight: float,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    total_loss_sum, grade_loss_sum = 0.0, 0.0
    all_preds, all_targets = [], []

    for batch in tqdm(loader, desc="Val", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        grades = batch["grade"].to(device, non_blocking=True)
        lesions = batch["lesions"].to(device, non_blocking=True)

        outputs = model(images)
        logits_grade = outputs["logits_grade"]
        loss_grade = grade_criterion(logits_grade, grades)
        if aux_weight > 0.0:
            loss_lesion = lesion_criterion(outputs["logits_lesion"], lesions)
            total_loss = loss_grade + aux_weight * loss_lesion
        else:
            total_loss = loss_grade

        total_loss_sum += total_loss.item() * images.size(0)
        grade_loss_sum += loss_grade.item() * images.size(0)

        preds = torch.argmax(logits_grade, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(grades.cpu().numpy())

    y_true, y_pred = np.array(all_targets), np.array(all_preds)
    n = len(loader.dataset)

    return {
        "val_loss": total_loss_sum / n,
        "val_grade_loss": grade_loss_sum / n,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "per_class_f1": [
            round(float(x), 4) for x in f1_score(y_true, y_pred, average=None, zero_division=0)
        ],
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(5))).tolist(),
    }


def run_training(cfg: Dict[str, Any], exp_dir: str) -> Dict[str, Any]:
    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tb_dir = os.path.join(exp_dir, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)

    train_csv = os.path.join(cfg["splits_dir"], "train.csv")
    val_csv = os.path.join(cfg["splits_dir"], "val.csv")

    train_ds = MMRDRCFPDataset(
        train_csv, cfg["root_dir"],
        transforms=get_transforms(cfg["img_size"], is_train=True),
        crop_border=cfg["crop_black_border"],
    )
    val_ds = MMRDRCFPDataset(
        val_csv, cfg["root_dir"],
        transforms=get_transforms(cfg["img_size"], is_train=False),
        crop_border=cfg["crop_black_border"],
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=True,
    )

    model = DRGradingResNet50(
        num_classes=cfg["num_classes"],
        num_lesions=cfg["num_lesions"],
        pretrained=cfg["pretrained"],
        dropout=cfg["dropout"],
    ).to(device)

    class_weights = compute_class_weights(train_csv, cfg["num_classes"])
    grade_criterion = build_criterion(cfg["loss_type"], class_weights, cfg["focal_gamma"], device)
    lesion_criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"], eta_min=1e-6
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    early_stopper = EarlyStopping(patience=cfg["es_patience"], min_delta=cfg["es_min_delta"])
    selection_key = cfg["selection_metric"]
    best_selection_score = -float("inf") if selection_key != "val_loss" else float("inf")
    best_summary: Dict[str, Any] = {}
    history = []

    print(f"Training outputs & checkpoints: {exp_dir}")
    print(f"TensorBoard log directory:      {tb_dir}")

    for epoch in range(1, cfg["epochs"] + 1):
        train_stats = train_one_epoch(
            model, train_loader, optimizer, grade_criterion,
            lesion_criterion, cfg["aux_lesion_weight"], device, scaler
        )
        val_metrics = validate(
            model, val_loader, grade_criterion,
            lesion_criterion, cfg["aux_lesion_weight"], device
        )
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # Log to TensorBoard
        writer.add_scalar("Loss/Train_Total", train_stats["train_loss"], epoch)
        writer.add_scalar("Loss/Val_Total", val_metrics["val_loss"], epoch)
        writer.add_scalar("Metrics/Val_Accuracy", val_metrics["accuracy"], epoch)
        writer.add_scalar("Metrics/Val_Macro_F1", val_metrics["macro_f1"], epoch)
        writer.add_scalar("Metrics/Val_QWK", val_metrics["qwk"], epoch)
        writer.add_scalar("Hyperparameters/LR", current_lr, epoch)
        for c_idx, c_f1 in enumerate(val_metrics["per_class_f1"]):
            writer.add_scalar(f"Per_Class_F1/Grade_{c_idx}", c_f1, epoch)

        early_stopper.step(val_metrics["val_loss"])

        current_score = val_metrics[selection_key]
        is_best = (
            (current_score < best_selection_score)
            if selection_key == "val_loss"
            else (current_score > best_selection_score)
        )

        if is_best:
            best_selection_score = current_score
            best_summary = {
                "best_epoch": epoch,
                "best_val_loss": round(early_stopper.best_loss, 4),
                **val_metrics,
            }
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_metrics": val_metrics,
                    "config": cfg,
                },
                os.path.join(exp_dir, "best_model.pt"),
            )

        epoch_log = {
            "epoch": epoch,
            "lr": round(current_lr, 7),
            "train_loss": round(train_stats["train_loss"], 4),
            "es_counter": early_stopper.counter,
            **val_metrics,
        }
        history.append(epoch_log)

        print(
            f"Epoch {epoch:02d}/{cfg['epochs']:02d} | "
            f"Train Loss: {train_stats['train_loss']:.4f} | "
            f"Val Loss: {val_metrics['val_loss']:.4f} (ES {early_stopper.counter}/{early_stopper.patience}) | "
            f"Acc: {val_metrics['accuracy']:.4f} | Macro-F1: {val_metrics['macro_f1']:.4f} | "
            f"QWK: {val_metrics['qwk']:.4f}" + (" [BEST]" if is_best else "")
        )

        with open(os.path.join(exp_dir, "training_history.json"), "w") as f:
            json.dump(history, f, indent=2)

        if cfg["es_enabled"] and early_stopper.early_stop:
            print(
                f"  --> Early stopping triggered at epoch {epoch} "
                f"(best val_loss: {early_stopper.best_loss:.4f})."
            )
            break

    writer.flush()
    writer.close()
    return best_summary


def main() -> None:
    cfg = {
        "seed": SEED,
        "root_dir": ROOT_DIR,
        "splits_dir": SPLITS_DIR,
        "img_size": IMG_SIZE,
        "crop_black_border": CROP_BLACK_BORDER,
        "num_workers": NUM_WORKERS,
        "pretrained": PRETRAINED,
        "num_classes": NUM_CLASSES,
        "num_lesions": NUM_LESIONS,
        "dropout": DROPOUT,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "loss_type": LOSS_TYPE,
        "focal_gamma": FOCAL_GAMMA,
        "aux_lesion_weight": AUX_LESION_WEIGHT,
        "selection_metric": SELECTION_METRIC,
        "es_enabled": EARLY_STOPPING_ENABLED,
        "es_patience": EARLY_STOPPING_PATIENCE,
        "es_min_delta": EARLY_STOPPING_MIN_DELTA,
    }

    run_training(cfg, exp_dir=OUTPUT_DIR)


if __name__ == "__main__":
    main()