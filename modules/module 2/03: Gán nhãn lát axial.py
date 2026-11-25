"""Gán nhãn 0/1 cho từng lát axial từ WT đã kiểm tra."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QA = ROOT / "results/module2/Kiem_tra_volume/kiem_tra_flair_wt.csv"
DEFAULT_OUTPUT = ROOT / "results/module2/Nhan_lat/nhan_lat_axial.csv"
AXIAL_AXIS = 2


def main() -> int:
    parser = argparse.ArgumentParser(description="Tạo nhãn có/không WT cho từng lát axial.")
    parser.add_argument("--qa", type=Path, default=DEFAULT_QA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        # Chỉ đọc các bệnh nhân đạt kiểm tra volume ở Bước 2.
        with args.qa.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        required = {"patient_id", "split", "wt_path", "status"}
        if not rows or not required.issubset(rows[0]):
            raise ValueError("Báo cáo kiểm tra thiếu cột cần thiết hoặc không có dữ liệu.")

        # Tạo một dòng cho mỗi lát axial, bỏ qua bệnh nhân có volume lỗi.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["patient_id", "split", "slice_index", "label"])
            for row in rows:
                if row["status"] != "valid":
                    continue
                wt_data = np.asanyarray(nib.load(row["wt_path"]).dataobj)
                # Lát có ít nhất một voxel WT dương được gán nhãn 1, ngược lại là 0.
                for slice_index in range(wt_data.shape[AXIAL_AXIS]):
                    label = int(np.any(np.take(wt_data, slice_index, axis=AXIAL_AXIS) > 0))
                    writer.writerow([row["patient_id"], row["split"], slice_index, label])
        print(f"Đã ghi nhãn lát axial: {args.output}")
        return 0
    except (OSError, ValueError) as error:
        # Báo lỗi đầu vào/đầu ra và trả mã lỗi cho terminal.
        print(error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
