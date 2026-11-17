"""Module 4 - Phần 2: Huấn luyện mô hình CNN 2D trên tập R, quét chọn ngưỡng quyết định tối ưu tau* trên tập S và đánh giá xác nhận trên tập C.

Thực nghiệm đối chứng công bằng với cùng SEED = 42 giữa 3 nhánh:
- Nhánh A (Baseline): Chuẩn hóa Min-Max toàn khung + Resize Bilinear 128x128.
- Nhánh B1 (Chốt): Cắt phân vị [P1, P99] vùng não + Lọc Bilateral (d=5, sigma_r=25, sigma_s=3) + Crop ROI cố định tập R + Resize 128x128.
- Nhánh B2 (Khảo sát phụ): Nhánh B1 có bật thêm CLAHE (clipLimit=2.0, tileGridSize=8x8).
"""

from __future__ import annotations

import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

SEED = 42


def seed_everything(seed: int = SEED) -> None:
    """Khóa toàn bộ hạt giống ngẫu nhiên để đảm bảo tái lập kết quả 100% giữa Nhánh A và Nhánh B."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class SliceClassifierCNN(nn.Module):
    """Kiến trúc CNN 2D 4 tầng tích chập (16 -> 32 -> 64 -> 128) + Global Average Pooling + Linear."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        f4 = self.conv4(f3)
        pooled = self.gap(f4).flatten(1)
        logit = self.fc(pooled).squeeze(1)
        return logit, f4


class SliceDataset(Dataset):
    """Dataset lưu cache các lát cắt 2D (1x128x128) kèm nhãn và định danh bệnh nhân."""

    def __init__(self, records: list[tuple[str, int, np.ndarray, int]], preprocess_fn) -> None:
        self.pids: list[str] = []
        self.zs: list[int] = []
        self.xs: list[np.ndarray] = []
        self.ys: list[float] = []
        t0 = time.perf_counter()
        for pid, z, f_sl, lb in records:
            im = preprocess_fn(f_sl)
            self.pids.append(pid)
            self.zs.append(z)
            self.xs.append(im[np.newaxis, :, :].astype(np.float32))
            self.ys.append(float(lb))
        self.prep_time_ms = (time.perf_counter() - t0) * 1000.0 / max(1, len(records))

    def __len__(self) -> int:
        return len(self.ys)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.xs[idx]),
            torch.tensor(self.ys[idx], dtype=torch.float32),
            self.pids[idx],
            self.zs[idx],
        )


def predict_loader(model: nn.Module, loader: DataLoader, device: torch.device):
    """Dự đoán xác suất p = sigmoid(z) trên toàn bộ DataLoader."""
    model.eval()
    probs, ys, pids, zs = [], [], [], []
    with torch.no_grad():
        for bx, by, bpid, bz in loader:
            bx = bx.to(device)
            logits, _ = model(bx)
            p = torch.sigmoid(logits).cpu().numpy()
            probs.extend(p.tolist())
            ys.extend(by.numpy().tolist())
            pids.extend(list(bpid))
            zs.extend(bz.numpy().tolist())
    return np.array(probs), np.array(ys, dtype=int), pids, zs


def select_optimal_threshold_on_S(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """Quét ngưỡng tau trong [0.15, 0.85] trên tập S (Selection) để cực đại hóa F1-score."""
    best_tau, best_f1 = 0.5, -1.0
    for tau in np.linspace(0.15, 0.85, 71):
        score = float(f1_score(y_true, (y_prob >= tau).astype(int), zero_division=0))
        if score > best_f1:
            best_f1, best_tau = score, float(tau)
    return best_tau, best_f1


def compute_slice_and_patient_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, patient_ids: list[str], tau: float
) -> dict[str, float]:
    """Tính AUC, F1, Precision, Recall theo lát và trung bình Macro theo từng bệnh nhân."""
    y_pred = (y_prob >= tau).astype(int)
    df_p = pd.DataFrame({"pid": patient_ids, "y": y_true, "pred": y_pred})
    pat_f1s, pat_precs, pat_recs = [], [], []
    for _, grp in df_p.groupby("pid"):
        if grp["y"].sum() > 0:
            pat_f1s.append(f1_score(grp["y"], grp["pred"], zero_division=0))
            pat_precs.append(precision_score(grp["y"], grp["pred"], zero_division=0))
            pat_recs.append(recall_score(grp["y"], grp["pred"], zero_division=0))
    return {
        "threshold_tau": round(float(tau), 2),
        "slice_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "slice_f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "slice_precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "slice_recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "patient_macro_f1": round(float(np.mean(pat_f1s)), 4),
        "patient_macro_precision": round(float(np.mean(pat_precs)), 4),
        "patient_macro_recall": round(float(np.mean(pat_recs)), 4),
    }


def train_and_evaluate_branch(
    ds_R: SliceDataset,
    ds_S: SliceDataset,
    ds_C: SliceDataset,
    device: torch.device,
    epochs: int = 8,
    batch_size: int = 64,
    lr: float = 1e-3,
):
    """Huấn luyện CNN trên tập R với Weighted BCE, chọn ngưỡng tau* trên tập S và xác nhận trên tập C."""
    seed_everything(SEED)
    loader_R = DataLoader(ds_R, batch_size=batch_size, shuffle=True)
    loader_S = DataLoader(ds_S, batch_size=batch_size, shuffle=False)
    loader_C = DataLoader(ds_C, batch_size=batch_size, shuffle=False)

    model = SliceClassifierCNN().to(device)
    y_r_np = np.array(ds_R.ys)
    pos_weight_val = float((y_r_np == 0).sum() / max(1, (y_r_np == 1).sum()))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val], device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_auc_s = -1.0
    best_state = None

    for _ in range(1, epochs + 1):
        model.train()
        for bx, by, _, _ in loader_R:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits, _ = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

        probs_s, y_s, _, _ = predict_loader(model, loader_S, device)
        auc_s = float(roc_auc_score(y_s, probs_s))
        if auc_s > best_auc_s:
            best_auc_s = auc_s
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    probs_s, y_s, pids_s, _ = predict_loader(model, loader_S, device)
    best_tau, _ = select_optimal_threshold_on_S(y_s, probs_s)

    t_inf0 = time.perf_counter()
    probs_c, y_c, pids_c, _ = predict_loader(model, loader_C, device)
    inf_ms_per_slice = (time.perf_counter() - t_inf0) * 1000.0 / max(1, len(ds_C))

    metrics_S = compute_slice_and_patient_metrics(y_s, probs_s, pids_s, best_tau)
    metrics_C = compute_slice_and_patient_metrics(y_c, probs_c, pids_c, best_tau)
    return model, best_tau, metrics_S, metrics_C, inf_ms_per_slice