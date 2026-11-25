# kiểm tra manifest và QA; lọc bệnh nhân; chia tập theo seed; lưu split và thống kê.

from __future__ import annotations

import argparse
import csv
import random
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "results/module1/Kết quả kiểm kê/path.csv"
DEFAULT_QA = ROOT / "results/module1/Kết quả kiểm tra từng cặp/chất_lượng.csv"
DEFAULT_ORIENTATION = ROOT / "results/module1/Kết_quả_chuẩn_hoá_RAS/orientation_report.csv"
DEFAULT_SLICE_COUNTS = ROOT / "results/module1/thong_ke_lat_theo_benh_nhan.csv"
DEFAULT_OUTPUT = ROOT / "results/module2/Quy_tắc_chia"
SEED = 42


def read_csv(path: Path, required: set[str]) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: thiếu cột {', '.join(sorted(missing))}")
        rows: dict[str, dict[str, str]] = {}
        # Từ chối ID rỗng/trùng để phép chia và phép join luôn theo bệnh nhân duy nhất.
        for line, row in enumerate(reader, 2):
            patient_id = (row.get("patient_id") or "").strip()
            if not patient_id:
                raise ValueError(f"{path}: dòng {line} thiếu patient_id")
            if patient_id in rows:
                raise ValueError(f"{path}: patient ID trùng {patient_id}")
            rows[patient_id] = row
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Chia train/validation/test theo patient ID trước khi tạo lát 2D."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--qa", type=Path, default=DEFAULT_QA)
    parser.add_argument("--orientation", type=Path, default=DEFAULT_ORIENTATION)
    parser.add_argument("--slice-counts", type=Path, default=DEFAULT_SLICE_COUNTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        manifest = read_csv(args.manifest, {"patient_id", "flair_count", "segmentation_count", "flair_paths", "segmentation_paths", "status", "reason"})
        qa = read_csv(args.qa, {"patient_id", "status", "reason"})
        orient = read_csv(args.orientation, {"patient_id", "status", "reason"})
        slice_counts = read_csv(args.slice_counts, {"patient_id", "positive_slices", "negative_slices", "total_slices"})
        # Hợp các ID để bệnh nhân thiếu ở bất kỳ báo cáo nào cũng được ghi lý do loại.
        patients = sorted(set(manifest) | set(qa) | set(orient))
        eligible, excluded = [], []
        for pid in patients:
            m, q, o = manifest.get(pid), qa.get(pid), orient.get(pid)
            reasons = []
            if not m: reasons.append("Không có trong manifest Module 1")
            elif m["status"] != "valid" or m["flair_count"] != "1" or m["segmentation_count"] != "1":
                reasons.append(m.get("reason") or "Manifest không có đúng một cặp FLAIR-seg hợp lệ")
            if not q or q["status"] != "valid": reasons.append((q or {}).get("reason") or "QA cặp không hợp lệ")
            if not o or o["status"] != "valid": reasons.append((o or {}).get("reason") or "Chuẩn hóa RAS/WT không hợp lệ")
            (excluded if reasons else eligible).append((pid, "; ".join(reasons)))
        if len(eligible) < 2:
            raise ValueError("Cần ít nhất 2 bệnh nhân hợp lệ để chia tập")
        # Chia trên patient ID trước khi sinh lát để ngăn rò rỉ lát giữa các tập.
        random.Random(SEED).shuffle(eligible)
        n_test = round(len(eligible) * .40)
        development = eligible[n_test:]
        # Train chiếm 30% tổng số ca; validation nhận phần còn lại để đủ toàn bộ.
        n_train = round(len(eligible) * .30)
        assignments = [(pid, "test") for pid, _ in eligible[:n_test]]
        assignments += [(pid, "train") for pid, _ in development[:n_train]]
        assignments += [(pid, "validation") for pid, _ in development[n_train:]]
        split_ids = {name: [pid for pid, assigned in assignments if assigned == name] for name in ("train", "validation", "test")}
        assigned_ids = [pid for pid, _ in assignments]
        # Hai kiểm tra bảo đảm đủ ID và không bệnh nhân nào xuất hiện ở hai tập.
        if len(assigned_ids) != len(set(assigned_ids)) or set(assigned_ids) != {pid for pid, _ in eligible}:
            raise ValueError("Kiểm tra thất bại: patient ID bị lặp hoặc thiếu trong các tập.")
        if any(set(split_ids[a]) & set(split_ids[b]) for i, a in enumerate(split_ids) for b in list(split_ids)[i+1:]):
            raise ValueError("Kiểm tra thất bại: các tập có patient ID giao nhau.")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with (args.output_dir / "phan_chia_benh_nhan.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f); w.writerow(["patient_id", "split", "seed"]); w.writerows((pid, split, SEED) for pid, split in sorted(assignments))
        with (args.output_dir / "benh_nhan_bi_loai.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f); w.writerow(["patient_id", "reason"]); w.writerows(excluded)
        with (args.output_dir / "thong_ke_lat_theo_tap.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f); w.writerow(["tap", "so_benh_nhan", "lat_duong", "lat_am", "tong_lat"])
            for split in ("train", "validation", "test"):
                ids = [pid for pid, assigned in assignments if assigned == split]
                counts = [slice_counts[pid] for pid in ids if pid in slice_counts]
                w.writerow([split, len(ids), sum(int(r["positive_slices"]) for r in counts), sum(int(r["negative_slices"]) for r in counts), sum(int(r["total_slices"]) for r in counts)])
        (args.output_dir / "quy_tac_chia_du_lieu.txt").write_text(
            f"QUY TẮC CHIA DỮ LIỆU THEO BỆNH NHÂN\nSeed ngẫu nhiên: {SEED}\n"
            f"Ngày tạo split: {date.today().isoformat()}\n"
            f"Bệnh nhân hợp lệ: {len(eligible)}; bị loại: {len(excluded)}\n"
            f"Train/validation/test: 30/30/40 ({n_train}/{len(development)-n_train}/{n_test})\n"
            "Xáo trộn và chia ở cấp patient ID trước khi tạo lát 2D; mỗi bệnh nhân chỉ thuộc một tập.\n"
            "Không thay đổi danh sách test sau khi xem số lát dương/âm, vị trí u hoặc kết quả mô hình.\n"
            "Không dùng test để chọn tiền xử lý, ngưỡng hoặc mô hình; chỉ dùng cho đánh giá cuối cùng.\n"
            "Thống kê lát theo tập chỉ để mô tả; không chuyển bệnh nhân giữa các tập.\n"
            "Train dùng để huấn luyện; validation dùng để lựa chọn cấu hình; test chỉ dùng đánh giá cuối cùng.\n",
            encoding="utf-8")
        print(f"Đã chia {len(eligible)} ca; loại {len(excluded)}; seed={SEED}; output={args.output_dir}")
        return 0
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
