"""Xuất chỉ mục lát axial từ nhãn và đường dẫn volume đã kiểm tra."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QA = ROOT / "results/module2/Kiem_tra_volume/kiem_tra_flair_wt.csv"
DEFAULT_LABELS = ROOT / "results/module2/Nhan_lat/nhan_lat_axial.csv"
DEFAULT_OUTPUT = ROOT / "results/module2/Chi_muc_lat/chi_muc_lat.csv"


def main() -> int:
    parser = argparse.ArgumentParser(description="Xuất CSV chỉ mục các lát axial đã gán nhãn.")
    parser.add_argument("--qa", type=Path, default=DEFAULT_QA)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        # Tạo tra cứu bệnh nhân hợp lệ để bổ sung đường dẫn FLAIR/WT.
        with args.qa.open(encoding="utf-8-sig", newline="") as f:
            valid = {r["patient_id"]: r for r in csv.DictReader(f) if r["status"] == "valid"}
        with args.labels.open(encoding="utf-8-sig", newline="") as f:
            labels = list(csv.DictReader(f))
        if not valid or not labels:
            raise ValueError("Thiếu bệnh nhân hợp lệ hoặc nhãn lát để xuất chỉ mục.")

        # Ghép từng nhãn lát với split và đường dẫn tương ứng rồi ghi CSV cuối.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["patient_id", "split", "slice_index", "label", "flair_path", "wt_path"])
            for row in labels:
                patient = valid.get(row["patient_id"])
                if patient is None or patient["split"] != row["split"]:
                    raise ValueError(f"Nhãn không khớp báo cáo QA/split: {row['patient_id']}")
                writer.writerow([row["patient_id"], row["split"], row["slice_index"], row["label"], patient["flair_path"], patient["wt_path"]])
        print(f"Đã xuất chỉ mục lát: {args.output}")
        return 0
    except (OSError, KeyError, ValueError) as error:
        # Báo lỗi đầu vào/đầu ra và trả mã lỗi cho terminal.
        print(error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
