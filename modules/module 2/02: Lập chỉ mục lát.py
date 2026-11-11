"""Lập chỉ mục lát axial từ FLAIR và WT sau khi đã khóa split bệnh nhân."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLITS = ROOT / "results/module2/Quy_tắc_chia/phan_chia_benh_nhan.csv"
DEFAULT_VOLUME_DIR = ROOT / "results/module1/Kết quả chuẩn hóa RAS/nifti"
DEFAULT_OUTPUT = ROOT / "results/module2/Chi_muc_lat/chi_muc_lat.csv"
AXIAL_AXIS = 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--volume-dir", type=Path, default=DEFAULT_VOLUME_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        with args.splits.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not {"patient_id", "split"}.issubset(reader.fieldnames or []):
                raise ValueError("Danh sách chia thiếu cột patient_id hoặc split.")
            assignments = {}
            for row in reader:
                pid, split = row["patient_id"].strip(), row["split"].strip()
                if not pid or split not in {"R", "S", "C", "test"} or pid in assignments:
                    raise ValueError(f"Dòng split không hợp lệ hoặc ID trùng: {pid}")
                assignments[pid] = split
        if not assignments:
            raise ValueError("Danh sách split rỗng.")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["patient_id", "split", "slice_index", "label", "flair_path", "wt_path"])
            for pid, split in sorted(assignments.items()):
                patient_dir = args.volume_dir / pid
                flair_path = patient_dir / f"{pid}_flair_RAS.nii.gz"
                wt_path = patient_dir / f"{pid}_wt_RAS.nii.gz"
                flair = nib.load(str(flair_path))
                wt = nib.load(str(wt_path))
                if len(flair.shape) != 3 or len(wt.shape) != 3 or flair.shape != wt.shape:
                    raise ValueError(f"FLAIR/WT phải là volume 3D cùng kích thước: {pid}")
                wt_data = np.asanyarray(wt.dataobj)
                if not np.isin(wt_data, (0, 1)).all():
                    raise ValueError(f"WT không phải mask nhị phân 0/1: {pid}")
                for slice_index in range(flair.shape[AXIAL_AXIS]):
                    label = int(np.any(np.take(wt_data, slice_index, axis=AXIAL_AXIS) > 0))
                    writer.writerow([pid, split, slice_index, label, flair_path.resolve(), wt_path.resolve()])
        print(f"Đã tạo chỉ mục cho {len(assignments)} bệnh nhân: {args.output}")
        return 0
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
