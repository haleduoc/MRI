"""Kiểm tra FLAIR và WT sau khi đã chia bệnh nhân."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLITS = ROOT / "results/module2/Quy_tắc_chia/phan_chia_benh_nhan.csv"
DEFAULT_VOLUME_DIR = ROOT / "results/module1/Kết_quả_chuẩn_hoá_RAS/nifti"
DEFAULT_OUTPUT = ROOT / "results/module2/Kiem_tra_volume/kiem_tra_flair_wt.csv"


def main() -> int:
    parser = argparse.ArgumentParser(description="Kiểm tra volume FLAIR/WT trước khi lập chỉ mục lát.")
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--volume-dir", type=Path, default=DEFAULT_VOLUME_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        # Đọc danh sách bệnh nhân và tập đã khóa ở Bước 1.
        with args.splits.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not {"patient_id", "split"}.issubset(reader.fieldnames or []):
                raise ValueError("Danh sách chia thiếu cột patient_id hoặc split.")
            assignments = {}
            for row in reader:
                pid, split = row["patient_id"].strip(), row["split"].strip()
                if not pid or split not in {"train", "validation", "test"} or pid in assignments:
                    raise ValueError(f"Dòng split không hợp lệ hoặc ID trùng: {pid}")
                assignments[pid] = split
        if not assignments:
            raise ValueError("Danh sách split rỗng.")

        # Đọc từng cặp FLAIR/WT và ghi trạng thái kiểm tra riêng cho bệnh nhân.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["patient_id", "split", "flair_path", "wt_path", "shape", "status", "reason"])
            for pid, split in sorted(assignments.items()):
                patient_dir = args.volume_dir / pid
                flair_path = patient_dir / f"{pid}_flair_RAS.nii.gz"
                wt_path = patient_dir / f"{pid}_wt_RAS.nii.gz"
                try:
                    # Kiểm tra volume 3D, kích thước khớp và mask chỉ có giá trị 0/1.
                    flair, wt = nib.load(str(flair_path)), nib.load(str(wt_path))
                    if len(flair.shape) != 3 or len(wt.shape) != 3:
                        raise ValueError("FLAIR và WT phải là volume 3D")
                    if flair.shape != wt.shape:
                        raise ValueError("Kích thước FLAIR và WT không khớp")
                    wt_data = np.asanyarray(wt.dataobj)
                    if not np.isin(wt_data, (0, 1)).all():
                        raise ValueError("WT không phải mask nhị phân 0/1")
                    # Ghi đường dẫn và kích thước để các bước sau dùng lại.
                    writer.writerow([pid, split, flair_path.resolve(), wt_path.resolve(), str(flair.shape), "valid", ""])
                except (OSError, ValueError) as error:
                    # Giữ lý do lỗi trong báo cáo, không đưa ca lỗi sang bước gán nhãn.
                    writer.writerow([pid, split, flair_path.resolve(), wt_path.resolve(), "", "invalid", str(error)])
        print(f"Đã ghi kết quả kiểm tra volume: {args.output}")
        return 0
    except (OSError, ValueError) as error:
        # Báo lỗi đầu vào/đầu ra và trả mã lỗi cho terminal.
        print(error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
