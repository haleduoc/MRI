"""Module 3 - Phần 2: Áp dụng bộ tham số chuẩn đúc kết từ Phần 1 cho toàn bộ 368 bệnh nhân (57.040 lát cắt).

Quy trình thực thi toàn cục:
1. Quét toàn bộ 368 bệnh nhân hợp lệ từ bảng phan_chia_benh_nhan.csv (Module 2).
2. Lọc bỏ các lát rỗng gần như không có mô não theo điều kiện A_brain(z) < N_min = 500 pixels (chỉ xét trên FLAIR).
3. Khớp khung bao hộp sọ cố định (Fixed Brain ROI) kèm lề margin = 4 px duy nhất trên 206 bệnh nhân tập R -> thu được [41:201, 13:215] (160x202 px).
4. Ghi nhận diện tích vùng não (brain_pixels) và diện tích khối u (wt_pixels) cho toàn bộ 48.825 lát hợp lệ.
5. Cung cấp hàm tiền xử lý chuẩn Nhánh A (Baseline Min-Max) và Nhánh B ([P1, P99] + Bilateral d=5, σr=25, σs=3 + Crop ROI [41:201, 13:215] + Resize 128x128).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
import cv2
import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SPLIT_CSV = ROOT / "results/module2/Quy_tắc_chia/phan_chia_benh_nhan.csv"
RAS_DIR = ROOT / "results/module1/Kết_quả_chuẩn_hoá_RAS/nifti"
OUT_DIR = ROOT / "results/module3/02_Toan_bo_dataset"

MIN_BRAIN_PIXELS = 500
ROI_MARGIN = 4
FIXED_ROI_R206 = (41, 201, 13, 215)  # Khớp từ 206 bệnh nhân tập R (160x202 px)
OUT_SIZE = (128, 128)


def preprocess_branch_A(f_slice: np.ndarray, out_size: tuple[int, int] = OUT_SIZE) -> np.ndarray:
    """Pipeline Nhánh A (Baseline): Min-Max toàn khung 240x240 -> Resize Bilinear 128x128."""
    imin, imax = float(f_slice.min()), float(f_slice.max())
    norm = (f_slice - imin) / (imax - imin + 1e-8)
    return cv2.resize(norm.astype(np.float32), out_size, interpolation=cv2.INTER_LINEAR)


def preprocess_branch_B(
    f_slice: np.ndarray,
    roi_box: tuple[int, int, int, int] = FIXED_ROI_R206,
    out_size: tuple[int, int] = OUT_SIZE,
) -> np.ndarray:
    """Pipeline Nhánh B (Xử lý ảnh chuẩn): Percentile [P1, P99] -> Bilateral (d=5, sigma_r=25, sigma_s=3) -> Crop ROI -> Resize Bilinear."""
    mask = f_slice > 0
    bp = f_slice[mask]
    v_low, v_high = float(np.percentile(bp, 1.0)), float(np.percentile(bp, 99.0))
    clipped = np.clip(f_slice, v_low, v_high)
    norm = np.zeros_like(f_slice, dtype=np.float32)
    norm[mask] = (clipped[mask] - v_low) / (v_high - v_low + 1e-8)
    u8 = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
    bf = cv2.bilateralFilter(u8, d=5, sigmaColor=25.0, sigmaSpace=3.0)
    proc = (bf.astype(np.float32) / 255.0) * mask
    xmin, xmax, ymin, ymax = roi_box
    return cv2.resize(proc[xmin:xmax, ymin:ymax], out_size, interpolation=cv2.INTER_LINEAR)


def resize_wt_mask_nearest(
    wt_slice: np.ndarray,
    roi_box: tuple[int, int, int, int] = FIXED_ROI_R206,
    out_size: tuple[int, int] = OUT_SIZE,
) -> np.ndarray:
    """Cắt khung ROI và nội suy mặt nạ nhị phân WT bằng Nearest-Neighbor (INTER_NEAREST)."""
    xmin, xmax, ymin, ymax = roi_box
    return cv2.resize(wt_slice[xmin:xmax, ymin:ymax].astype(np.uint8), out_size, interpolation=cv2.INTER_NEAREST)


def run_full_dataset_scan(ras_base: Path = RAS_DIR) -> None:
    """Quét toàn bộ 368 bệnh nhân (57.040 lát), lọc lát rỗng, ước lượng ROI trên 206 ca tập R và xuất thống kê."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_splits_all = pd.read_csv(SPLIT_CSV)
    pid_to_split = dict(
        zip(df_splits_all["patient_id"].str.strip(), df_splits_all["split"].str.strip())
    )

    valid_slice_records = []
    patient_summary_rows = []
    xmin_R, xmax_R, ymin_R, ymax_R = 240, 0, 240, 0
    total_raw_slices = 0
    total_removed_slices = 0
    t0 = time.perf_counter()

    all_pids = sorted(pid_to_split.keys())
    for idx_p, pid in enumerate(all_pids, 1):
        sp = pid_to_split[pid]
        f_p = ras_base / pid / f"{pid}_flair_RAS.nii.gz"
        w_p = ras_base / pid / f"{pid}_wt_RAS.nii.gz"

        flair_vol = np.asanyarray(nib.load(str(f_p)).dataobj).astype(np.float32)
        wt_vol = (np.asanyarray(nib.load(str(w_p)).dataobj) > 0).astype(np.uint8)

        n_z = flair_vol.shape[2]
        total_raw_slices += n_z
        brain_counts = (flair_vol > 0).sum(axis=(0, 1))
        wt_counts = (wt_vol > 0).sum(axis=(0, 1))

        if sp == "R":
            valid_z_mask = brain_counts >= MIN_BRAIN_PIXELS
            if np.any(valid_z_mask):
                sub_vol = flair_vol[:, :, valid_z_mask] > 0
                xs, ys, _ = np.where(sub_vol)
                xmin_R = min(xmin_R, int(xs.min()))
                xmax_R = max(xmax_R, int(xs.max()))
                ymin_R = min(ymin_R, int(ys.min()))
                ymax_R = max(ymax_R, int(ys.max()))

        kept_pos, kept_neg, rem_z = 0, 0, 0
        for z in range(n_z):
            nb = int(brain_counts[z])
            nw = int(wt_counts[z])
            lb = 1 if nw > 0 else 0
            if nb < MIN_BRAIN_PIXELS:
                rem_z += 1
                total_removed_slices += 1
                continue
            if lb == 1:
                kept_pos += 1
            else:
                kept_neg += 1
            valid_slice_records.append({
                "patient_id": pid,
                "split": sp,
                "slice_index": z,
                "label": lb,
                "brain_pixels": nb,
                "wt_pixels": nw,
            })

        patient_summary_rows.append({
            "patient_id": pid,
            "split": sp,
            "raw_slices": n_z,
            "removed_empty_slices": rem_z,
            "valid_slices": kept_pos + kept_neg,
            "valid_positive_slices": kept_pos,
            "valid_negative_slices": kept_neg,
        })
        if idx_p % 50 == 0 or idx_p == len(all_pids):
            print(f"Đã xử lý {idx_p}/{len(all_pids)} bệnh nhân ({time.perf_counter() - t0:.1f}s)")

    full_roi_box = (
        max(0, xmin_R - ROI_MARGIN),
        min(240, xmax_R + ROI_MARGIN + 1),
        max(0, ymin_R - ROI_MARGIN),
        min(240, ymax_R + ROI_MARGIN + 1),
    )

    df_valid_all = pd.DataFrame(valid_slice_records)
    df_pat_summary = pd.DataFrame(patient_summary_rows)

    split_stats_m3 = []
    for sp_name in ["R", "S", "C", "test"]:
        sub_df = df_valid_all[df_valid_all["split"] == sp_name]
        sub_pat = df_pat_summary[df_pat_summary["split"] == sp_name]
        split_stats_m3.append({
            "tap": sp_name,
            "so_benh_nhan": len(sub_pat),
            "tong_lat_ban_dau_M2": int(sub_pat["raw_slices"].sum()),
            "so_lat_rong_bi_loai_M3": int(sub_pat["removed_empty_slices"].sum()),
            "tong_lat_hop_le_M3": len(sub_df),
            "lat_duong_hop_le": int((sub_df["label"] == 1).sum()),
            "lat_am_hop_le": int((sub_df["label"] == 0).sum()),
        })
    df_split_stats_m3 = pd.DataFrame(split_stats_m3)

    df_valid_all.to_csv(OUT_DIR / "chi_muc_lat_hop_le_toan_bo.csv", index=False)
    df_pat_summary.to_csv(OUT_DIR / "thong_ke_loc_lat_theo_benh_nhan.csv", index=False)
    df_split_stats_m3.to_csv(OUT_DIR / "thong_ke_lat_hop_le_theo_tap.csv", index=False)

    roi_str = f"[{full_roi_box[0]}:{full_roi_box[1]}, {full_roi_box[2]}:{full_roi_box[3]}]"
    roi_full_info = {
        "fitted_on_split": "R (206 patients)",
        "min_brain_pixels_threshold": MIN_BRAIN_PIXELS,
        "margin_pixels": ROI_MARGIN,
        "xmin": full_roi_box[0],
        "xmax": full_roi_box[1],
        "ymin": full_roi_box[2],
        "ymax": full_roi_box[3],
        "crop_height": full_roi_box[1] - full_roi_box[0],
        "crop_width": full_roi_box[3] - full_roi_box[2],
        "output_resize": list(OUT_SIZE),
        "selected_pipeline_B": {
            "step1_norm": "Brain Percentile [P1, P99] + Min-Max to [0, 1]",
            "step2_denoise": "Bilateral (d=5, sigmaColor=25, sigmaSpace=3)",
            "step3_clahe": "Disabled for slice classifier",
            "step4_roi_resize": f"Fixed ROI {roi_str} + Bilinear/Nearest",
        },
        "total_slices_module2": total_raw_slices,
        "removed_empty_slices_module3": total_removed_slices,
        "remaining_valid_slices_module3": len(df_valid_all),
    }
    with open(OUT_DIR / "roi_va_cau_hinh_chuan_toan_bo.json", "w", encoding="utf-8") as f:
        json.dump(roi_full_info, f, indent=2, ensure_ascii=False)

    print("Hoàn tất quét 368 bệnh nhân. Khung ROI tập R:", full_roi_box)


if __name__ == "__main__":
    run_full_dataset_scan()