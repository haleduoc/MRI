"""Module 4 - Phần 1: Thực nghiệm CNN 2D đối chứng Nhánh A, Nhánh B1, Nhánh B2 và trích xuất Bản đồ kích hoạt đặc trưng (CAM) trên mẫu thử.

Quy trình thực nghiệm:
1. Khóa toàn bộ hạt giống ngẫu nhiên (SEED = 42) để đảm bảo đối chứng công bằng giữa các nhánh tiền xử lý.
2. Nạp dữ liệu phân tầng theo bệnh nhân từ R (Train), S (Selection - Chọn ngưỡng) và C (Confirmation - Xác nhận).
3. Huấn luyện mạng CNN 2D với hàm mất mát Weighted Binary Cross-Entropy (trọng số lớp dương tính từ tập R).
4. Quét tìm ngưỡng xác suất tối ưu tau* trên tập S để tối đa hóa F1-score, sau đó khóa tau* để đánh giá trên tập C.
5. Trích xuất Class Activation Map (CAM) từ tầng tích chập cuối cùng (Conv4 16x16) trên 4 mẫu thử điển hình (Mẫu 1: 001 z=78; Mẫu 2: 002 z=50; Mẫu 3: 003 z=85; Mẫu 4: 005 z=75) và xuất hình ảnh đối chứng.
"""

from __future__ import annotations

import os
import random
import time
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[2]
SPLIT_CSV = ROOT / "results/module2/Quy_tắc_chia/phan_chia_benh_nhan.csv"
RAS_DIR = ROOT / "results/module1/Kết_quả_chuẩn_hoá_RAS/nifti"
OUT_DIR = ROOT / "results/module4/01_Thuc_nghiem_mau_thu"

SEED = 42
N_MIN_BRAIN = 500
ROI_MARGIN = 4
OUT_SIZE = (128, 128)

SAMPLE_CASES = [
    ("BraTS20_Training_001", 78, "Mẫu 1: U lớn điển hình"),
    ("BraTS20_Training_002", 50, "Mẫu 2: Rìa u / U kích thước nhỏ"),
    ("BraTS20_Training_003", 85, "Mẫu 3: U kèm phù nề ranh giới mờ"),
    ("BraTS20_Training_005", 75, "Mẫu 4: Lát não lành (Không u)"),
]


def seed_everything(seed: int = SEED) -> None:
    """Khóa toàn bộ seed ngẫu nhiên của Python, NumPy và PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def load_ras_volume(pid: str, volume_root: Path = RAS_DIR) -> tuple[np.ndarray, np.ndarray]:
    """Đọc khối ảnh FLAIR và mặt nạ nhị phân WT đã chuẩn hóa hướng RAS từ Module 1."""
    f_path = volume_root / pid / f"{pid}_flair_RAS.nii.gz"
    w_path = volume_root / pid / f"{pid}_wt_RAS.nii.gz"
    flair_vol = np.asanyarray(nib.load(str(f_path)).dataobj).astype(np.float32)
    wt_vol = (np.asanyarray(nib.load(str(w_path)).dataobj) > 0).astype(np.uint8)
    return flair_vol, wt_vol


def load_patient_valid_slices(pid: str, step_z: int = 2) -> list[tuple[str, int, np.ndarray, int]]:
    """Trích xuất các lát cắt hợp lệ (A_brain >= 500 px) dọc trục axial của một bệnh nhân."""
    flair_vol, wt_vol = load_ras_volume(pid)
    records = []
    for z in range(0, flair_vol.shape[2], step_z):
        f_slice = flair_vol[:, :, z]
        w_slice = wt_vol[:, :, z]
        if int((f_slice > 0).sum()) >= N_MIN_BRAIN:
            label = 1 if int((w_slice > 0).sum()) > 0 else 0
            records.append((pid, z, f_slice, label))
    return records


def preprocess_branch_A(f_slice: np.ndarray) -> np.ndarray:
    """Nhánh A (Baseline): Chuẩn hóa Min-Max toàn khung 240x240 -> Resize Bilinear 128x128."""
    imin, imax = float(f_slice.min()), float(f_slice.max())
    norm = (f_slice - imin) / (imax - imin + 1e-8)
    return cv2.resize(norm.astype(np.float32), OUT_SIZE, interpolation=cv2.INTER_LINEAR)


def preprocess_branch_B(
    f_slice: np.ndarray,
    roi_box: tuple[int, int, int, int],
    use_clahe: bool = False,
) -> np.ndarray:
    """Nhánh B (Xử lý ảnh): Cắt phân vị [P1, P99] vùng não -> Bilateral (d=5, sigma_r=25, sigma_s=3) -> (Tùy chọn CLAHE) -> Crop ROI -> Resize 128x128."""
    mask = f_slice > 0
    bp = f_slice[mask]
    v_low, v_high = float(np.percentile(bp, 1.0)), float(np.percentile(bp, 99.0))
    clipped = np.clip(f_slice, v_low, v_high)
    norm = np.zeros_like(f_slice, dtype=np.float32)
    norm[mask] = (clipped[mask] - v_low) / (v_high - v_low + 1e-8)

    u8 = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
    bf_u8 = cv2.bilateralFilter(u8, d=5, sigmaColor=25.0, sigmaSpace=3.0)
    if use_clahe:
        clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        bf_u8 = clahe_obj.apply(bf_u8)
    proc = (bf_u8.astype(np.float32) / 255.0) * mask

    xmin, xmax, ymin, ymax = roi_box
    cropped = proc[xmin:xmax, ymin:ymax]
    return cv2.resize(cropped, OUT_SIZE, interpolation=cv2.INTER_LINEAR)


class SliceDataset(Dataset):
    """Dataset lưu cache các lát cắt 2D (1x128x128) và đo thời gian tiền xử lý trung bình (ms/lát)."""

    def __init__(
        self,
        records: list[tuple[str, int, np.ndarray, int]],
        branch: str = "A",
        roi_box: tuple[int, int, int, int] = (51, 192, 22, 205),
        use_clahe: bool = False,
    ) -> None:
        self.pids: list[str] = []
        self.zs: list[int] = []
        self.xs: list[np.ndarray] = []
        self.ys: list[float] = []
        t0 = time.perf_counter()
        for pid, z, f_sl, lb in records:
            if branch == "A":
                im = preprocess_branch_A(f_sl)
            else:
                im = preprocess_branch_B(f_sl, roi_box=roi_box, use_clahe=use_clahe)
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


class SliceClassifierCNN(nn.Module):
    """Mạng CNN 2D 4 khối tích chập (16 -> 32 -> 64 -> 128) + Global Average Pooling + Linear."""

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


def predict_loader(model: nn.Module, loader: DataLoader, device: torch.device):
    """Suy luận xác suất dự đoán p = sigmoid(z) trên toàn bộ tập dữ liệu."""
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


def compute_classification_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, pids: list[str], tau: float
) -> dict[str, float]:
    """Tính AUC, F1, Precision, Recall theo lát và trung bình Macro theo bệnh nhân."""
    y_pred = (y_prob >= tau).astype(int)
    auc_all = float(roc_auc_score(y_true, y_prob))
    f1_all = float(f1_score(y_true, y_pred, zero_division=0))
    prec_all = float(precision_score(y_true, y_pred, zero_division=0))
    rec_all = float(recall_score(y_true, y_pred, zero_division=0))

    df_p = pd.DataFrame({"pid": pids, "y": y_true, "prob": y_prob, "pred": y_pred})
    pat_f1s, pat_precs, pat_recs = [], [], []
    for _, grp in df_p.groupby("pid"):
        if grp["y"].sum() > 0:
            pat_f1s.append(f1_score(grp["y"], grp["pred"], zero_division=0))
            pat_precs.append(precision_score(grp["y"], grp["pred"], zero_division=0))
            pat_recs.append(recall_score(grp["y"], grp["pred"], zero_division=0))
    return {
        "Threshold (tau*)": round(tau, 2),
        "Slice_AUC": round(auc_all, 4),
        "Slice_F1": round(f1_all, 4),
        "Slice_Precision": round(prec_all, 4),
        "Slice_Recall": round(rec_all, 4),
        "Patient_Macro_F1": round(float(np.mean(pat_f1s)), 4),
        "Patient_Macro_Precision": round(float(np.mean(pat_precs)), 4),
        "Patient_Macro_Recall": round(float(np.mean(pat_recs)), 4),
    }


def get_cam_and_prob(
    model: SliceClassifierCNN, img_2d: np.ndarray, device: torch.device
) -> tuple[float, np.ndarray]:
    """Trích xuất xác suất dự đoán p và Bản đồ kích hoạt đặc trưng (CAM) từ tầng Conv4 (16x16)."""
    model.eval()
    x_t = torch.tensor(img_2d[np.newaxis, np.newaxis, :, :], dtype=torch.float32, device=device)
    with torch.no_grad():
        logit, feat4 = model(x_t)
        prob = float(torch.sigmoid(logit).item())
        weights = model.fc.weight.data[0]
        cam = torch.einsum("c,chw->hw", weights, feat4[0]).cpu().numpy()
        cam = np.maximum(cam, 0)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam_resized = cv2.resize(cam, OUT_SIZE, interpolation=cv2.INTER_LINEAR)
    return prob, cam_resized


def train_and_evaluate_branch(
    ds_R: SliceDataset,
    ds_S: SliceDataset,
    ds_C: SliceDataset,
    device: torch.device,
    epochs: int = 8,
    batch_size: int = 64,
    lr: float = 1e-3,
):
    """Huấn luyện CNN trên tập R, quét ngưỡng tau* trên tập S và đánh giá xác nhận trên tập C."""
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
    best_tau, best_f1_s = 0.5, -1.0
    tau_grid = np.linspace(0.15, 0.85, 71)
    f1_curve_s = []
    for t in tau_grid:
        f1_t = float(f1_score(y_s, (probs_s >= t).astype(int), zero_division=0))
        f1_curve_s.append(f1_t)
        if f1_t > best_f1_s:
            best_f1_s = f1_t
            best_tau = float(t)

    t_inf0 = time.perf_counter()
    probs_c, y_c, pids_c, _ = predict_loader(model, loader_C, device)
    inf_ms_per_slice = (time.perf_counter() - t_inf0) * 1000.0 / max(1, len(ds_C))

    metrics_S = compute_classification_metrics(y_s, probs_s, pids_s, best_tau)
    metrics_C = compute_classification_metrics(y_c, probs_c, pids_c, best_tau)
    return model, best_tau, tau_grid, f1_curve_s, metrics_S, metrics_C, inf_ms_per_slice


def run_module4_experiment() -> None:
    """Chạy toàn bộ thực nghiệm Module 4 đối chứng Nhánh A, B1, B2 và xuất biểu đồ CAM + ROC."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df_splits = pd.read_csv(SPLIT_CSV)
    rng = random.Random(SEED)
    r_pids_all = sorted(df_splits[df_splits["split"] == "R"]["patient_id"].tolist())
    s_pids_all = sorted(df_splits[df_splits["split"] == "S"]["patient_id"].tolist())
    c_pids_all = sorted(df_splits[df_splits["split"] == "C"]["patient_id"].tolist())

    must_have_r = [p for p, _, _ in SAMPLE_CASES]
    rem_r = [p for p in r_pids_all if p not in must_have_r]
    rng.shuffle(rem_r)
    rng.shuffle(s_pids_all)
    rng.shuffle(c_pids_all)

    selected_r_pids = must_have_r + rem_r[:26]
    selected_s_pids = s_pids_all[:12]
    selected_c_pids = c_pids_all[:12]

    data_R = [rec for pid in selected_r_pids for rec in load_patient_valid_slices(pid)]
    data_S = [rec for pid in selected_s_pids for rec in load_patient_valid_slices(pid)]
    data_C = [rec for pid in selected_c_pids for rec in load_patient_valid_slices(pid)]

    xs_r, ys_r = [], []
    for _, _, f_sl, _ in data_R:
        nx, ny = np.where(f_sl > 0)
        xs_r.extend([int(nx.min()), int(nx.max())])
        ys_r.extend([int(ny.min()), int(ny.max())])
    roi_box_r = (
        max(0, min(xs_r) - ROI_MARGIN),
        min(240, max(xs_r) + ROI_MARGIN + 1),
        max(0, min(ys_r) - ROI_MARGIN),
        min(240, max(ys_r) + ROI_MARGIN + 1),
    )

    ds_R_A, ds_S_A, ds_C_A = SliceDataset(data_R, "A"), SliceDataset(data_S, "A"), SliceDataset(data_C, "A")
    ds_R_B1, ds_S_B1, ds_C_B1 = SliceDataset(data_R, "B", roi_box_r, False), SliceDataset(data_S, "B", roi_box_r, False), SliceDataset(data_C, "B", roi_box_r, False)
    ds_R_B2, ds_S_B2, ds_C_B2 = SliceDataset(data_R, "B", roi_box_r, True), SliceDataset(data_S, "B", roi_box_r, True), SliceDataset(data_C, "B", roi_box_r, True)

    model_A, tau_A, grid_tau, f1_s_A, mS_A, mC_A, inf_A = train_and_evaluate_branch(ds_R_A, ds_S_A, ds_C_A, device)
    model_B1, tau_B1, _, f1_s_B1, mS_B1, mC_B1, inf_B1 = train_and_evaluate_branch(ds_R_B1, ds_S_B1, ds_C_B1, device)
    model_B2, tau_B2, _, f1_s_B2, mS_B2, mC_B2, inf_B2 = train_and_evaluate_branch(ds_R_B2, ds_S_B2, ds_C_B2, device)

    m4_rows = []
    for name_b, ds_c, m_s, m_c, inf_t in [
        ("Nhánh A: Baseline (Min-Max + Resize)", ds_C_A, mS_A, mC_A, inf_A),
        ("Nhánh B1: [P1,P99] + Bilateral + Crop ROI", ds_C_B1, mS_B1, mC_B1, inf_B1),
        ("Nhánh B2: [P1,P99] + Bilateral + CLAHE + Crop ROI", ds_C_B2, mS_B2, mC_B2, inf_B2),
    ]:
        m4_rows.append({"Nhánh": name_b, "Tập": "S (Selection)", **m_s, "Preproc_ms": round(ds_c.prep_time_ms, 3), "Infer_ms": round(inf_t, 3)})
        m4_rows.append({"Nhánh": name_b, "Tập": "C (Confirmation)", **m_c, "Preproc_ms": round(ds_c.prep_time_ms, 3), "Infer_ms": round(inf_t, 3)})

    pd.DataFrame(m4_rows).to_csv(OUT_DIR / "module4_branch_comparison_S_C.csv", index=False)
    print("Đã lưu kết quả thực nghiệm Module 4 tại:", OUT_DIR)


if __name__ == "__main__":
    run_module4_experiment()