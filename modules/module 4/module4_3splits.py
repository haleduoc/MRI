from __future__ import annotations

import random
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = Path("/content/m4_cache")
OUT_DIR = ROOT / "results/module4"

SEED = 42
VARIANTS = [
    "P0_Baseline_MinMax",
    "P1_Pct1_99_ROI",
    "P2_BF_Low_sr5",
    "P3_BF_Std_sr25",
    "P4_BF_High_sr150",
    "P5_CLAHE_2.0",
]
EXPERIMENTS = [
    ("M0_Baseline_MinMax", ["P0_Baseline_MinMax"], "P0_Baseline_MinMax", 8, False, True),
    ("M1_Pct1_99_ROI",     ["P1_Pct1_99_ROI"],     "P1_Pct1_99_ROI",     8, False, False),
    ("M2_BF_Low_sr5",      ["P2_BF_Low_sr5"],      "P2_BF_Low_sr5",      8, False, False),
    ("M3_BF_Std_sr25",     ["P3_BF_Std_sr25"],     "P3_BF_Std_sr25",     8, False, True),
    ("M4_BF_High_sr150",   ["P4_BF_High_sr150"],   "P4_BF_High_sr150",   8, False, False),
    ("M5_CLAHE_2.0",       ["P5_CLAHE_2.0"],       "P5_CLAHE_2.0",       8, False, False),
    ("M6_MultiFilter_Aug", ["P1_Pct1_99_ROI", "P3_BF_Std_sr25", "P5_CLAHE_2.0"], "P3_BF_Std_sr25", 6, True, True),
]


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MultiFilterSliceDataset(Dataset):
    def __init__(self, mmaps: dict[str, np.memmap], row_indices, labels, variant_names, augment_flip: bool = False):
        self.mmaps = mmaps
        self.row_indices = np.asarray(row_indices, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.variant_names = list(variant_names)
        self.augment_flip = augment_flip
        self.n_base = len(self.row_indices)

    def __len__(self) -> int:
        return self.n_base * len(self.variant_names)

    def __getitem__(self, i: int):
        v_idx = i // self.n_base
        s_idx = i % self.n_base
        r_idx = self.row_indices[s_idx]
        img = self.mmaps[self.variant_names[v_idx]][r_idx].astype(np.float32) / 255.0
        if self.augment_flip and random.random() < 0.5:
            img = np.ascontiguousarray(img[:, ::-1])
        return torch.from_numpy(img).unsqueeze(0), torch.tensor(self.labels[s_idx], dtype=torch.float32)


class SliceClassifierCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.MaxPool2d(2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True)
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor, return_feat: bool = False):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        feat = self.block4(x)
        pooled = self.gap(feat).flatten(1)
        logit = self.fc(pooled).squeeze(1)
        if return_feat:
            return logit, feat
        return logit


@torch.no_grad()
def predict_probs(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    probs, ys = [], []
    t0 = time.perf_counter()
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(xb)
            p = torch.sigmoid(logits)
        probs.append(p.float().cpu().numpy())
        ys.append(yb.numpy())
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    probs = np.concatenate(probs)
    ys = np.concatenate(ys).astype(np.int32)
    return probs, ys, elapsed_ms / max(len(ys), 1)


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    best_t, best_f1 = 0.50, -1.0
    for t in np.arange(0.15, 0.86, 0.01):
        f1 = f1_score(y_true, (y_prob >= t).astype(np.int32), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return round(best_t, 2), float(best_f1)


def compute_all_metrics(df_split: pd.DataFrame, y_prob: np.ndarray, tau: float) -> dict[str, float]:
    y_true = df_split["label"].values.astype(np.int32)
    y_pred = (y_prob >= tau).astype(np.int32)
    tmp = pd.DataFrame({"pid": df_split["patient_id"].values, "y": y_true, "p": y_pred})
    p_f1s, p_precs, p_recs = [], [], []
    for _, g in tmp.groupby("pid"):
        p_f1s.append(f1_score(g["y"], g["p"], zero_division=0))
        p_precs.append(precision_score(g["y"], g["p"], zero_division=0))
        p_recs.append(recall_score(g["y"], g["p"], zero_division=0))
    return {
        "Slice_AUC": round(float(roc_auc_score(y_true, y_prob)), 4),
        "Slice_F1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "Slice_Prec": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "Slice_Rec": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "Patient_Macro_F1": round(float(np.mean(p_f1s)), 4),
        "Patient_Macro_Prec": round(float(np.mean(p_precs)), 4),
        "Patient_Macro_Rec": round(float(np.mean(p_recs)), 4),
    }


@torch.no_grad()
def get_cam_and_prob(model: SliceClassifierCNN, img_u8: np.ndarray, device: torch.device) -> tuple[float, np.ndarray]:
    model.eval()
    x_t = torch.from_numpy(img_u8.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    logit, feat = model(x_t, return_feat=True)
    prob = float(torch.sigmoid(logit).item())
    weights = model.fc.weight.squeeze(0)
    cam = torch.relu((weights[:, None, None] * feat.squeeze(0)).sum(dim=0)).cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam = cv2.resize(cam, (128, 128), interpolation=cv2.INTER_LINEAR)
    return prob, cam


def run_module4_3splits(cache_dir: Path = CACHE_DIR, out_dir: Path = OUT_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df_meta = pd.read_csv(cache_dir / "meta_valid_slices.csv")
    n_slices = len(df_meta)
    mmaps = {
        name: np.memmap(cache_dir / f"{name}.dat", dtype=np.uint8, mode="r", shape=(n_slices, 128, 128))
        for name in VARIANTS
    }
    train_df = df_meta[df_meta["split"] == "train"].reset_index(drop=True)
    val_df = df_meta[df_meta["split"] == "validation"].reset_index(drop=True)
    test_df = df_meta[df_meta["split"] == "test"].reset_index(drop=True)

    n_pos = int((train_df["label"] == 1).sum())
    n_neg = int((train_df["label"] == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)

    results = []
    saved_models = {}

    for exp_name, tr_vars, ev_var, eps, aug_flip, save_ckpt in EXPERIMENTS:
        seed_everything(SEED)
        ds_tr = MultiFilterSliceDataset(mmaps, train_df["row_idx"].values, train_df["label"].values, tr_vars, augment_flip=aug_flip)
        ds_va = MultiFilterSliceDataset(mmaps, val_df["row_idx"].values, val_df["label"].values, [ev_var], augment_flip=False)
        ds_te = MultiFilterSliceDataset(mmaps, test_df["row_idx"].values, test_df["label"].values, [ev_var], augment_flip=False)

        dl_tr = DataLoader(ds_tr, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
        dl_va = DataLoader(ds_va, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
        dl_te = DataLoader(ds_te, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

        model = SliceClassifierCNN().to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

        best_val_auc = -1.0
        best_state = None
        for _ in range(1, eps + 1):
            model.train()
            for xb, yb in dl_tr:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    loss = criterion(model(xb), yb)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            va_probs, va_ys, _ = predict_probs(model, dl_va, device)
            va_auc = float(roc_auc_score(va_ys, va_probs))
            if va_auc > best_val_auc:
                best_val_auc = va_auc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        if save_ckpt:
            torch.save(best_state, out_dir / f"{exp_name}_best.pt")

        va_probs, va_ys, _ = predict_probs(model, dl_va, device)
        tau_star, _ = find_best_threshold(va_ys, va_probs)
        val_met = compute_all_metrics(val_df, va_probs, tau_star)

        te_probs, _, inf_ms = predict_probs(model, dl_te, device)
        test_met = compute_all_metrics(test_df, te_probs, tau_star)

        results.append({
            "Mo_hinh": exp_name,
            "So_lat_Train": len(ds_tr),
            "Tap_Test_dau_vao": ev_var,
            "Tau_Val": tau_star,
            "Val_AUC": val_met["Slice_AUC"],
            "Val_F1": val_met["Slice_F1"],
            "Test_AUC": test_met["Slice_AUC"],
            "Test_F1": test_met["Slice_F1"],
            "Test_Prec": test_met["Slice_Prec"],
            "Test_Rec": test_met["Slice_Rec"],
            "Test_Patient_MacroF1": test_met["Patient_Macro_F1"],
            "Inf_ms": round(inf_ms, 3),
        })
        if save_ckpt:
            saved_models[exp_name] = (model, tau_star)

    df_results = pd.DataFrame(results)
    df_results.to_csv(out_dir / "Module4_7Models_TestResults.csv", index=False)

    cross_rows = []
    for m_name in ["M3_BF_Std_sr25", "M6_MultiFilter_Aug"]:
        mdl, tau_m = saved_models[m_name]
        for tv in ["P1_Pct1_99_ROI", "P2_BF_Low_sr5", "P3_BF_Std_sr25", "P4_BF_High_sr150", "P5_CLAHE_2.0"]:
            ds_t = MultiFilterSliceDataset(mmaps, test_df["row_idx"].values, test_df["label"].values, [tv], augment_flip=False)
            dl_t = DataLoader(ds_t, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
            probs_t, _, _ = predict_probs(mdl, dl_t, device)
            met = compute_all_metrics(test_df, probs_t, tau_m)
            cross_rows.append({
                "Mo_hinh_da_train": m_name,
                "Tham_so_loc_Tap_Test": tv,
                "Tau_khoa": tau_m,
                "Test_AUC": met["Slice_AUC"],
                "Test_F1": met["Slice_F1"],
                "Test_Prec": met["Slice_Prec"],
                "Test_Rec": met["Slice_Rec"],
                "Patient_Macro_F1": met["Patient_Macro_F1"],
            })

    df_cross = pd.DataFrame(cross_rows)
    df_cross.to_csv(out_dir / "Module4_CrossFilter_TestRobustness.csv", index=False)
    return df_results, df_cross


if __name__ == "__main__":
    run_module4_3splits()