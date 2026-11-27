from __future__ import annotations

import glob
import json
import time
from pathlib import Path
import cv2
import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
M2_INDEX_GLOB = ROOT / "results/module2/Chi_muc_lat/*.csv"
RAS_DIRS = [
    ROOT / "results/module1/Kết_quả_chuẩn_hoá_RAS/nifti",
    ROOT / "results/module1/Kết quả chuẩn hóa RAS/nifti",
]
OUT_DIR = ROOT / "results/module3"

SEED = 42
BRAIN_MIN_PX = 500
MARGIN_PX = 4
OUT_SIZE = (128, 128)
MULTI_FILTER_VARIANTS = [
    "P1_Pct1_99_ROI",
    "P3_BF_Std_sr25",
    "P5_CLAHE_2.0",
]


def load_ras_volume(pid: str, kaggle_root: str | Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    for folder in RAS_DIRS:
        f_p = folder / pid / f"{pid}_flair_RAS.nii.gz"
        w_p = folder / pid / f"{pid}_wt_RAS.nii.gz"
        if f_p.exists() and w_p.exists():
            return (
                nib.load(str(f_p)).get_fdata().astype(np.float32),
                (nib.load(str(w_p)).get_fdata() > 0).astype(np.uint8),
            )
    if kaggle_root is not None:
        k_dir = Path(kaggle_root) / f"BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/{pid}"
        f_nii = nib.as_closest_canonical(nib.load(str(k_dir / f"{pid}_flair.nii")))
        s_nii = nib.as_closest_canonical(nib.load(str(k_dir / f"{pid}_seg.nii")))
        wt_arr = np.isin(s_nii.get_fdata(), [1, 2, 4]).astype(np.uint8)
        return f_nii.get_fdata().astype(np.float32), wt_arr
    raise FileNotFoundError(f"Không tìm thấy dữ liệu NIfTI cho {pid}")


def make_square_roi(x0: int, x1: int, y0: int, y1: int, h_max: int = 240, w_max: int = 240) -> tuple[int, int, int, int]:
    h, w = x1 - x0, y1 - y0
    side = min(max(h, w), h_max, w_max)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    sx0 = max(0, min(h_max - side, cx - side // 2))
    sy0 = max(0, min(w_max - side, cy - side // 2))
    return sx0, sx0 + side, sy0, sy0 + side


def preprocess_slice_variants(f_slice: np.ndarray, square_roi: tuple[int, int, int, int], out_size: tuple[int, int] = OUT_SIZE) -> dict[str, np.ndarray]:
    mask_full = f_slice > 0
    mx = float(f_slice.max())
    p0_u8 = np.clip((f_slice / (mx + 1e-8)) * 255.0, 0, 255).astype(np.uint8)

    lo, hi = np.percentile(f_slice[mask_full], [1, 99]) if np.any(mask_full) else (0.0, 1.0)
    x0, x1, y0, y1 = square_roi
    sl_roi = f_slice[x0:x1, y0:y1]
    mask_roi = sl_roi > 0
    norm_roi = np.zeros_like(sl_roi, dtype=np.float32)
    if np.any(mask_roi):
        norm_roi[mask_roi] = (np.clip(sl_roi[mask_roi], lo, hi) - lo) / (hi - lo + 1e-8)
    p1_roi_u8 = np.clip(norm_roi * 255.0, 0, 255).astype(np.uint8)

    p2_roi_u8 = cv2.bilateralFilter(p1_roi_u8, 5, 5.0, 3.0)
    p2_roi_u8[~mask_roi] = 0

    p3_roi_u8 = cv2.bilateralFilter(p1_roi_u8, 5, 25.0, 3.0)
    p3_roi_u8[~mask_roi] = 0

    p4_roi_u8 = cv2.bilateralFilter(p1_roi_u8, 9, 150.0, 9.0)
    p4_roi_u8[~mask_roi] = 0

    clahe_op = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    p5_roi_u8 = clahe_op.apply(p3_roi_u8)
    p5_roi_u8[~mask_roi] = 0

    return {
        "P0_Baseline_MinMax": cv2.resize(p0_u8, out_size, interpolation=cv2.INTER_LINEAR),
        "P1_Pct1_99_ROI": cv2.resize(p1_roi_u8, out_size, interpolation=cv2.INTER_LINEAR),
        "P2_BF_Low_sr5": cv2.resize(p2_roi_u8, out_size, interpolation=cv2.INTER_LINEAR),
        "P3_BF_Std_sr25": cv2.resize(p3_roi_u8, out_size, interpolation=cv2.INTER_LINEAR),
        "P4_BF_High_sr150": cv2.resize(p4_roi_u8, out_size, interpolation=cv2.INTER_LINEAR),
        "P5_CLAHE_2.0": cv2.resize(p5_roi_u8, out_size, interpolation=cv2.INTER_LINEAR),
    }


def run_module3_3splits(kaggle_root: str | Path | None = None, out_dir: Path = OUT_DIR) -> tuple[pd.DataFrame, dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_files = sorted(glob.glob(str(M2_INDEX_GLOB)))
    assert len(csv_files) > 0, f"Không tìm thấy file CSV tại {M2_INDEX_GLOB}"
    df_index = pd.read_csv(csv_files[0])
    slice_col = next(c for c in ["slice_index", "slice_idx", "z", "slice"] if c in df_index.columns)

    t_start = time.time()
    all_rows = []
    xmin, xmax = 240, 0
    ymin, ymax = 240, 0

    for pid, grp in df_index.groupby("patient_id", sort=True):
        split_name = str(grp["split"].iloc[0])
        f_vol, w_vol = load_ras_volume(pid, kaggle_root=kaggle_root)
        brain_mask_3d = f_vol > 0
        brain_counts = brain_mask_3d.sum(axis=(0, 1))
        wt_counts = (w_vol > 0).sum(axis=(0, 1))

        if split_name == "train":
            valid_z_mask = brain_counts >= BRAIN_MIN_PX
            if np.any(valid_z_mask):
                union_2d = brain_mask_3d[:, :, valid_z_mask].any(axis=2)
                coords = np.argwhere(union_2d)
                if len(coords) > 0:
                    xmin = min(xmin, int(coords[:, 0].min()))
                    xmax = max(xmax, int(coords[:, 0].max()))
                    ymin = min(ymin, int(coords[:, 1].min()))
                    ymax = max(ymax, int(coords[:, 1].max()))

        for _, row in grp.iterrows():
            z = int(row[slice_col])
            b_px = int(brain_counts[z])
            w_px = int(wt_counts[z])
            all_rows.append({
                "patient_id": pid,
                "split": split_name,
                "slice_index": z,
                "label": int(row["label"]),
                "brain_pixels": b_px,
                "wt_pixels": w_px,
                "is_valid": int(b_px >= BRAIN_MIN_PX),
            })

    tight_x0 = max(0, xmin - MARGIN_PX)
    tight_x1 = min(240, xmax + MARGIN_PX + 1)
    tight_y0 = max(0, ymin - MARGIN_PX)
    tight_y1 = min(240, ymax + MARGIN_PX + 1)
    sq_x0, sq_x1, sq_y0, sq_y1 = make_square_roi(tight_x0, tight_x1, tight_y0, tight_y1)
    side = sq_x1 - sq_x0

    df_all = pd.DataFrame(all_rows)
    df_valid = df_all[df_all["is_valid"] == 1].drop(columns=["is_valid"]).reset_index(drop=True)

    stats_rows = []
    for sp in ["train", "validation", "test"]:
        sub_all = df_all[df_all["split"] == sp]
        sub_val = df_valid[df_valid["split"] == sp]
        n_tot = len(sub_all)
        n_keep = len(sub_val)
        n_rem = n_tot - n_keep
        n_pos = int((sub_val["label"] == 1).sum())
        n_neg = int((sub_val["label"] == 0).sum())
        stats_rows.append({
            "tap": sp,
            "so_benh_nhan": int(sub_all["patient_id"].nunique()),
            "tong_lat_M2": n_tot,
            "lat_rong_bi_loai": n_rem,
            "ty_le_loai_%": round(n_rem / max(n_tot, 1) * 100, 2),
            "lat_hop_le_M3": n_keep,
            "lat_duong_WT1": n_pos,
            "ty_le_duong_%": round(n_pos / max(n_keep, 1) * 100, 2),
            "lat_am_WT0": n_neg,
            "ty_le_am_%": round(n_neg / max(n_keep, 1) * 100, 2),
        })

    df_stats = pd.DataFrame(stats_rows)
    cfg = {
        "seed": SEED,
        "split_strategy": "3-split patient-level (70% train / 15% validation / 15% test)",
        "min_brain_pixels_threshold": BRAIN_MIN_PX,
        "margin_pixels": MARGIN_PX,
        "tight_roi": {"xmin": tight_x0, "xmax": tight_x1, "ymin": tight_y0, "ymax": tight_y1},
        "square_roi": {
            "xmin": sq_x0, "xmax": sq_x1, "ymin": sq_y0, "ymax": sq_y1,
            "crop_height": side, "crop_width": side,
            "aspect_ratio_preserved": True,
            "resize_scale_to_128": round(128.0 / side, 6),
            "pixel_area_factor_to_original": round((side / 128.0) ** 2, 6),
        },
        "output_size": list(OUT_SIZE),
        "multi_filter_augmentation_variants": MULTI_FILTER_VARIANTS,
        "elapsed_sec": round(time.time() - t_start, 1),
    }

    df_valid.to_csv(out_dir / "FullDataset_ValidSlices_3Splits.csv", index=False)
    df_all.to_csv(out_dir / "FullDataset_AllSlices_WithFlag_3Splits.csv", index=False)
    df_stats.to_csv(out_dir / "FullDataset_Module3_SplitStats_3Splits.csv", index=False)
    with open(out_dir / "FullDataset_Module3_Config_3Splits.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    return df_valid, cfg


if __name__ == "__main__":
    run_module3_3splits()