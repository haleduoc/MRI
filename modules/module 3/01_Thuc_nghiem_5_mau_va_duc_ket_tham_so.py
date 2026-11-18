"""Module 3 - Phần 1: Thực nghiệm từng bước tiền xử lý trên 5 lát cắt đại diện và khảo sát tham số Thấp/Chuẩn/Cao.

Quy trình gồm:
- Bước 0: Trích xuất 5 lát cắt đại diện từ đầu ra Module 2 (chi_muc_lat.csv) và đo thông số vật lý gốc.
- Bước 1: Lọc lát rỗng theo FLAIR (A_brain >= 500 px) và đối chứng Chuẩn hóa Min-Max vs Percentile [P1, P99] vs [P5, P90].
- Bước 2: Khử nhiễu bảo toàn biên bằng Bilateral Filter (Thấp σr=5 / Chuẩn d=5, σr=25, σs=3 / Cao d=9, σr=150) và Non-Local Means.
- Bước 3: Khảo sát tăng cường tương phản cục bộ CLAHE (clipLimit = 0.5, 2.0, 8.0) và chứng minh hiện tượng khuếch đại vùng sáng trên lát âm.
- Bước 4: Ước lượng khung bao hộp sọ cố định (Fixed Brain ROI) trên tập R và đối chứng nội suy Bilinear vs Nearest-Neighbor.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INDEX_CSV = ROOT / "results/module2/Chi_muc_lat/chi_muc_lat.csv"
RAS_DIR = ROOT / "results/module1/Kết_quả_chuẩn_hoá_RAS/nifti"
OUT_DIR = ROOT / "results/module3/01_Thuc_nghiem_5_mau"

N_MIN_BRAIN_PIXELS = 500
ROI_MARGIN = 4
OUT_SIZE = (128, 128)

SAMPLE_SLICES = [
    ("BraTS20_Training_001", 78, "Mẫu 1: U lớn điển hình"),
    ("BraTS20_Training_002", 50, "Mẫu 2: Rìa u / U kích thước nhỏ"),
    ("BraTS20_Training_003", 85, "Mẫu 3: U kèm phù nề ranh giới mờ"),
    ("BraTS20_Training_005", 75, "Mẫu 4: Lát não lành (Không u)"),
    ("BraTS20_Training_001", 2,  "Mẫu 5: Lát rỗng vùng đỉnh sọ"),
]


def load_ras_volume(pid: str, volume_root: Path = RAS_DIR) -> tuple[np.ndarray, np.ndarray]:
    """Đọc khối ảnh FLAIR và mặt nạ nhị phân WT đã chuẩn hóa hướng RAS từ Module 1."""
    f_path = volume_root / pid / f"{pid}_flair_RAS.nii.gz"
    w_path = volume_root / pid / f"{pid}_wt_RAS.nii.gz"
    flair_vol = np.asanyarray(nib.load(str(f_path)).dataobj).astype(np.float32)
    wt_vol = (np.asanyarray(nib.load(str(w_path)).dataobj) > 0).astype(np.uint8)
    return flair_vol, wt_vol


def step1_minmax(img: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Chuẩn hóa Min-Max toàn khung [0, I_max] (Baseline Nhánh A)."""
    imin, imax = float(np.min(img)), float(np.max(img))
    return ((img - imin) / (imax - imin + eps)).astype(np.float32)


def step1_percentile_brain(
    img: np.ndarray, p_low: float = 1.0, p_high: float = 99.0, eps: float = 1e-8
) -> tuple[np.ndarray, float, float]:
    """Cắt phân vị [P_low, P_high] trên vùng mô não (I > 0) và chuẩn hóa về [0, 1]."""
    mask = img > 0
    if not np.any(mask):
        return np.zeros_like(img, dtype=np.float32), 0.0, 0.0
    bp = img[mask]
    v_low, v_high = float(np.percentile(bp, p_low)), float(np.percentile(bp, p_high))
    clipped = np.clip(img, v_low, v_high)
    out = np.zeros_like(img, dtype=np.float32)
    out[mask] = (clipped[mask] - v_low) / (v_high - v_low + eps)
    return out, v_low, v_high


def step2_bilateral_filter(
    norm_img: np.ndarray, d: int = 5, sigma_color: float = 25.0, sigma_space: float = 3.0
) -> tuple[np.ndarray, float]:
    """Khử nhiễu bảo toàn biên bằng Bilateral Filter và đo thời gian thực thi (ms)."""
    mask = norm_img > 0
    u8 = np.clip(norm_img * 255.0, 0, 255).astype(np.uint8)
    t0 = time.perf_counter()
    bf = cv2.bilateralFilter(u8, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    return (bf.astype(np.float32) / 255.0) * mask, dt_ms


def step2_nlm_filter(norm_img: np.ndarray, h: float = 10.0) -> tuple[np.ndarray, float]:
    """Khử nhiễu bằng Non-Local Means (NLM) để đối chứng tốc độ và chất lượng với Bilateral."""
    mask = norm_img > 0
    u8 = np.clip(norm_img * 255.0, 0, 255).astype(np.uint8)
    t0 = time.perf_counter()
    nlm = cv2.fastNlMeansDenoising(u8, None, h=h, templateWindowSize=7, searchWindowSize=21)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    return (nlm.astype(np.float32) / 255.0) * mask, dt_ms


def step3_clahe(
    norm_img: np.ndarray, clip_limit: float = 2.0, tile_grid: tuple[int, int] = (8, 8)
) -> np.ndarray:
    """Tăng tương phản cục bộ có giới hạn (CLAHE) kèm khóa mặt nạ vùng não."""
    mask = norm_img > 0
    u8 = np.clip(norm_img * 255.0, 0, 255).astype(np.uint8)
    clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return (clahe_obj.apply(u8).astype(np.float32) / 255.0) * mask


def compute_edge_and_laplacian_metrics(
    img_norm: np.ndarray, brain_mask: np.ndarray, wt_mask: np.ndarray
) -> tuple[float, float]:
    """Đo độ lệch chuẩn Laplacian (nhiễu tần số cao σ_Lap) và độ dốc biên khối u Sobel (|∇I|)."""
    lap = cv2.Laplacian(img_norm, cv2.CV_32F, ksize=3)
    sigma_lap = float(np.std(lap[brain_mask]))
    gx = cv2.Sobel(img_norm, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_norm, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx**2 + gy**2)
    if np.sum(wt_mask > 0) > 0:
        edge_ring = cv2.morphologyEx(wt_mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        edge_grad = float(np.mean(grad[edge_ring]))
    else:
        edge_grad = 0.0
    return sigma_lap, edge_grad


def calc_entropy(img_norm: np.ndarray, mask: np.ndarray) -> float:
    """Tính Entropy thông tin Shannon H = -∑ p_i log2(p_i) trên vùng mô não."""
    vals = np.clip(img_norm[mask] * 255.0, 0, 255).astype(np.uint8)
    hist = np.bincount(vals, minlength=256).astype(np.float64)
    p = hist[hist > 0] / np.sum(hist)
    return float(-np.sum(p * np.log2(p)))


def run_ablation_5_samples() -> None:
    """Thực thi trọn vẹn Bước 0 -> Bước 4 trên 5 mẫu thử đại diện và lưu bảng số liệu."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_index = pd.read_csv(INDEX_CSV)

    valid_samples = []
    step1_rows, step2_rows, step3_rows, step4_rows = [], [], [], []

    for pid, z_idx, note in SAMPLE_SLICES:
        row_m2 = df_index[(df_index["patient_id"] == pid) & (df_index["slice_index"] == z_idx)].iloc[0]
        flair_vol, wt_vol = load_ras_volume(pid)
        raw = flair_vol[:, :, z_idx]
        wt = wt_vol[:, :, z_idx]
        n_brain = int((raw > 0).sum())
        if n_brain < N_MIN_BRAIN_PIXELS:
            print(f"[Bước 1A - Lọc lát rỗng] Loại bỏ {pid} z={z_idx} (A_brain = {n_brain} < {N_MIN_BRAIN_PIXELS})")
            continue

        valid_samples.append({
            "pid": pid, "z": z_idx, "note": note, "flair": raw, "wt": wt,
            "label": int(row_m2["label"]),
        })

    # Thực nghiệm Bước 1, Bước 2, Bước 3 trên các mẫu hợp lệ
    xs, ys = [], []
    for s in valid_samples:
        raw, wt = s["flair"], s["wt"]
        brain_mask = raw > 0
        healthy_mask = brain_mask & (wt == 0)
        tumor_mask = wt > 0
        nx, ny = np.where(brain_mask)
        xs.extend([int(nx.min()), int(nx.max())])
        ys.extend([int(ny.min()), int(ny.max())])

        im_minmax = step1_minmax(raw)
        im_p1_99, _, _ = step1_percentile_brain(raw, 1.0, 99.0)
        im_p5_90, _, _ = step1_percentile_brain(raw, 5.0, 90.0)
        s["norm_A"] = im_minmax
        s["norm_B"] = im_p1_99

        for m_name, im_cur in [
            ("1. Min-Max gốc [0, Max]", im_minmax),
            ("2. Percentile [P1, P99] (Chuẩn)", im_p1_99),
            ("3. Percentile [P5, P90] (Quá hẹp)", im_p5_90),
        ]:
            mu_h, std_h = float(im_cur[healthy_mask].mean()), float(im_cur[healthy_mask].std())
            mu_t = float(im_cur[tumor_mask].mean()) if tumor_mask.sum() > 0 else 0.0
            delta_i = (mu_t - mu_h) if tumor_mask.sum() > 0 else 0.0
            cnr = abs(delta_i) / (std_h + 1e-8) if tumor_mask.sum() > 0 else 0.0
            sat_pct = float((im_cur[tumor_mask] >= 0.999).mean() * 100) if tumor_mask.sum() > 0 else 0.0
            step1_rows.append({
                "Mẫu": s["note"].split(":")[0], "Patient": f"{s['pid'][-3:]} (z={s['z']})",
                "Phương pháp": m_name, "μ_0": round(mu_h, 4), "μ_1": round(mu_t, 4),
                "ΔI": round(delta_i, 4), "σ_0": round(std_h, 4), "CNR": round(cnr, 3),
                "% U cháy sáng": round(sat_pct, 2),
            })

        # Bước 2: Khử nhiễu Bilateral (Thấp / Chuẩn / Cao) vs NLM
        bf_low, t_low = step2_bilateral_filter(im_p1_99, d=5, sigma_color=5.0, sigma_space=3.0)
        bf_std, t_std = step2_bilateral_filter(im_p1_99, d=5, sigma_color=25.0, sigma_space=3.0)
        bf_high, t_high = step2_bilateral_filter(im_p1_99, d=9, sigma_color=150.0, sigma_space=9.0)
        nlm_std, t_nlm = step2_nlm_filter(im_p1_99, h=10.0)
        s["step2_bf_std"] = bf_std

        for d_name, im_d, dt in [
            ("0. Chưa lọc (Sau Bước 1)", im_p1_99, 0.0),
            ("1. Bilateral Thấp (d=5, σr=5)", bf_low, t_low),
            ("2. Bilateral Chuẩn (d=5, σr=25)", bf_std, t_std),
            ("3. Bilateral Cao (d=9, σr=150)", bf_high, t_high),
            ("4. Non-Local Means (h=10)", nlm_std, t_nlm),
        ]:
            s_lap, e_grad = compute_edge_and_laplacian_metrics(im_d, brain_mask, wt)
            std_h = float(im_d[healthy_mask].std())
            cnr = abs(float(im_d[tumor_mask].mean() - im_d[healthy_mask].mean())) / (std_h + 1e-8) if tumor_mask.sum() > 0 else 0.0
            step2_rows.append({
                "Mẫu": s["note"].split(":")[0], "Patient": f"{s['pid'][-3:]} (z={s['z']})",
                "Cấu hình": d_name, "σ_Lap": round(s_lap, 4), "σ_0": round(std_h, 4),
                "|∇I|": round(e_grad, 4), "CNR": round(cnr, 3), "Time_ms": round(dt, 3),
            })

        # Bước 3: CLAHE (Thấp / Chuẩn / Cao)
        c_low = step3_clahe(bf_std, clip_limit=0.5)
        c_std = step3_clahe(bf_std, clip_limit=2.0)
        c_high = step3_clahe(bf_std, clip_limit=8.0)
        s["step3_clahe_std"] = c_std

        for c_name, im_c in [
            ("0. Trước CLAHE (Sau Bước 2)", bf_std),
            ("1. CLAHE Thấp (clip=0.5)", c_low),
            ("2. CLAHE Chuẩn (clip=2.0)", c_std),
            ("3. CLAHE Cao (clip=8.0)", c_high),
        ]:
            _, e_grad = compute_edge_and_laplacian_metrics(im_c, brain_mask, wt)
            fp_bright = float((im_c[healthy_mask] > 0.75).mean() * 100)
            step3_rows.append({
                "Mẫu": s["note"].split(":")[0], "Patient": f"{s['pid'][-3:]} (z={s['z']})",
                "Cấu hình CLAHE": c_name, "Biên |∇I|": round(e_grad, 4),
                "Entropy": round(calc_entropy(im_c, brain_mask), 3),
                "% Mô lành > 0.75": round(fp_bright, 2),
            })

    # Bước 4: Cắt khung ROI cố định và đối chứng Bilinear vs Nearest-Neighbor trên Mask WT
    xmin_roi, xmax_roi = max(0, min(xs) - ROI_MARGIN), min(240, max(xs) + ROI_MARGIN + 1)
    ymin_roi, ymax_roi = max(0, min(ys) - ROI_MARGIN), min(240, max(ys) + ROI_MARGIN + 1)
    for s in valid_samples:
        im_b = s["step2_bf_std"]
        wt_f = s["wt"].astype(np.float32)
        im_crop = im_b[xmin_roi:xmax_roi, ymin_roi:ymax_roi]
        wt_crop = wt_f[xmin_roi:xmax_roi, ymin_roi:ymax_roi]
        im_res = cv2.resize(im_crop, OUT_SIZE, interpolation=cv2.INTER_LINEAR)
        wt_nearest = cv2.resize(wt_crop, OUT_SIZE, interpolation=cv2.INTER_NEAREST)
        wt_bilinear_wrong = cv2.resize(wt_crop, OUT_SIZE, interpolation=cv2.INTER_LINEAR)
        step4_rows.append({
            "Mẫu": s["note"].split(":")[0],
            "ROI": f"[{xmin_roi}:{xmax_roi}, {ymin_roi}:{ymax_roi}]",
            "% Não trước Crop": round(float((im_b > 0).mean() * 100), 2),
            "% Não sau Crop": round(float((im_res > 0).mean() * 100), 2),
            "Pixel Mask lỗi (Bilinear)": int(((wt_bilinear_wrong > 0) & (wt_bilinear_wrong < 1)).sum()),
            "Pixel Mask lỗi (Nearest)": int(((wt_nearest > 0) & (wt_nearest < 1)).sum()),
        })

    pd.DataFrame(step1_rows).to_csv(OUT_DIR / "step1_normalization_metrics.csv", index=False)
    pd.DataFrame(step2_rows).to_csv(OUT_DIR / "step2_denoising_metrics.csv", index=False)
    pd.DataFrame(step3_rows).to_csv(OUT_DIR / "step3_clahe_metrics.csv", index=False)
    pd.DataFrame(step4_rows).to_csv(OUT_DIR / "step4_roi_resize_metrics.csv", index=False)
    print("Đã hoàn tất thực nghiệm 5 mẫu thử và lưu kết quả tại:", OUT_DIR)


if __name__ == "__main__":
    run_ablation_5_samples()