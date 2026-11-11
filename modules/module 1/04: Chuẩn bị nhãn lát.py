"""Chuẩn bị quy tắc nhãn lát từ mask WT theo volume; không tạo bảng lát."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WT_DIR = ROOT / "results/module1/Kết quả chuẩn hóa RAS/nifti"
DEFAULT_ORIENTATION_REPORT = ROOT / "results/module1/Kết quả chuẩn hóa RAS/orientation_report.csv"
DEFAULT_OUTPUT = ROOT / "results/module1/quy_tac_nhan_lat.txt"


def labels_from_wt(mask: np.ndarray, axis: int) -> np.ndarray:
    """Trả nhãn 0/1 cho mỗi lát theo trục đã chỉ định."""
    if mask.ndim != 3:
        raise ValueError("WT phải là volume 3D.")
    if axis not in range(3):
        raise ValueError("Trục lát phải là 0, 1 hoặc 2.")
    if not np.isin(mask, (0, 1)).all():
        raise ValueError("WT chỉ được chứa giá trị 0 và 1.")
    reduce_axes = tuple(i for i in range(3) if i != axis)
    return np.any(mask > 0, axis=reduce_axes).astype(np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wt-dir", type=Path, default=DEFAULT_WT_DIR)
    parser.add_argument("--orientation-report", type=Path, default=DEFAULT_ORIENTATION_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    with args.orientation_report.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    axes = {int(row["axial_voxel_axis"]) for row in rows if row.get("status") == "valid" and row.get("axial_voxel_axis", "").isdigit()}
    if len(axes) != 1:
        parser.error(f"Không xác định được một trục axial thống nhất trong {args.orientation_report}")
    axis = axes.pop()
    masks = sorted(args.wt_dir.rglob("*_wt_RAS.nii.gz"))
    if not masks:
        parser.error(f"Không tìm thấy WT RAS trong {args.wt_dir}")
    for path in masks:
        labels_from_wt(np.asanyarray(nib.load(str(path)).dataobj), axis)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "Module 1 - Phần 5: Chuẩn bị nhãn lát\n"
        f"Nguồn WT volume: {args.wt_dir}\n"
        f"Báo cáo hướng: {args.orientation_report}\n"
        f"Trục axial voxel: {axis}\n"
        "Quy tắc: label = 1 nếu lát có ít nhất một voxel WT; label = 0 nếu không có.\n"
        "Thứ tự: Module 2 khóa patient ID vào các tập trước; sau đó mới tạo lát 2D và gán nhãn.\n"
        f"Số volume WT đã kiểm tra: {len(masks)}\n"
        "Không tạo bảng lát 2D hoặc chia tập dữ liệu.\n",
        encoding="utf-8",
    )
    print(f"Đã ghi quy tắc: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
