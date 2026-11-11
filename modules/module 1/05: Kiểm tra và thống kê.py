"""Tổng hợp kiểm tra Module 1 và tạo overlay QC đại diện từ FLAIR–WT RAS."""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/module1"
QA = RESULTS / "Kết quả kiểm tra từng cặp/chất_lượng.csv"
ORIENTATION = RESULTS / "Kết quả chuẩn hóa RAS/orientation_report.csv"
OUT = RESULTS / "kiem_tra_bang_mat"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def overlay(flair_path: Path, wt_path: Path, patient: str, axis: int, index: int, title: str, dest: Path) -> None:
    flair_img, wt_img = nib.load(str(flair_path)), nib.load(str(wt_path))
    if flair_img.shape != wt_img.shape or not np.allclose(flair_img.affine, wt_img.affine):
        raise ValueError(f"FLAIR/WT khác lưới: {patient}")
    flair = np.take(np.asanyarray(flair_img.dataobj), index, axis=axis)
    wt = np.take(np.asanyarray(wt_img.dataobj), index, axis=axis) > 0
    flair = np.rot90(flair)
    wt = np.rot90(wt)
    valid = flair[np.isfinite(flair) & (flair > 0)]
    low, high = np.percentile(valid, [1, 99]) if valid.size else (0, 1)
    gray = np.zeros(flair.shape, dtype=np.uint8) if high <= low else np.clip((flair-low)*255/(high-low),0,255).astype(np.uint8)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    rgb[wt] = (255, 55, 55)
    image = Image.fromarray(rgb).resize((512, 512))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 511, 24), fill="black")
    draw.text((5, 5), f"{patient} | axis {axis} slice {index} | {title}", fill="white")
    dest.parent.mkdir(parents=True, exist_ok=True)
    image.save(dest)


def main() -> None:
    qa = {r["patient_id"]: r for r in read_csv(QA)}
    ori = {r["patient_id"]: r for r in read_csv(ORIENTATION)}
    patients = []
    smallest = largest = negative = None
    for pid, row in sorted(ori.items()):
        if row.get("status") != "valid" or qa.get(pid, {}).get("status") not in {"valid", "resampled"}:
            continue
        wt_path, flair_path = Path(row["wt_output"]), Path(row["flair_output"])
        axis = int(row["axial_voxel_axis"])
        wt_img = nib.load(str(wt_path))
        wt = np.asanyarray(wt_img.dataobj) > 0
        slice_counts = np.count_nonzero(wt, axis=tuple(i for i in range(3) if i != axis))
        positive = np.flatnonzero(slice_counts)
        total = wt.shape[axis]
        patients.append((pid, len(positive), total-len(positive), total))
        if len(positive):
            local_min, local_max = int(positive[np.argmin(slice_counts[positive])]), int(np.argmax(slice_counts))
            candidate_min = (int(slice_counts[local_min]), pid, row, axis, local_min)
            candidate_max = (int(slice_counts[local_max]), pid, row, axis, local_max)
            if smallest is None or candidate_min[0] < smallest[0]: smallest = candidate_min
            if largest is None or candidate_max[0] > largest[0]: largest = candidate_max
        if negative is None and len(positive) < total:
            positive_set = set(positive.tolist())
            negative_indices = np.array([i for i in range(total) if i not in positive_set])
            flair_img = nib.load(str(row["flair_output"]))
            flair_data = np.asanyarray(flair_img.dataobj)
            visible_counts = np.count_nonzero(np.take(flair_data, negative_indices, axis=axis), axis=tuple(i for i in range(3) if i != axis))
            best = int(np.argmax(visible_counts))
            negative = (pid, row, axis, int(negative_indices[best]))

    with (RESULTS / "thong_ke_lat_theo_benh_nhan.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["patient_id", "positive_slices", "negative_slices", "total_slices"])
        for pid, positive_count, negative_count, total in patients: w.writerow([pid, positive_count, negative_count, total])

    valid_status = Counter(r["status"] for r in qa.values())
    aligned = sum(bool(r.get("resampled_segmentation_path")) for r in qa.values())
    pos_count = sum(p[1] for p in patients)
    neg_count = sum(p[2] for p in patients)
    summary = (f"Ca hợp lệ sau chuẩn hóa RAS: {len(patients)}\n"+
      f"Ca thiếu/trùng theo QA: {sum(v for k,v in valid_status.items() if k not in {'valid','resampled'})}\n"+
      f"Ca phải căn chỉnh segmentation: {aligned}\nLát WT dương: {pos_count}\nLát WT âm: {neg_count}\n")
    (RESULTS / "summary_module1.txt").write_text(summary, encoding="utf-8")

    selections = []
    if smallest: selections.append((smallest, "WT nhỏ"))
    if largest: selections.append((largest, "WT lớn"))
    if negative: selections.append(((0, *negative), "WT âm"))
    for n, (item, title) in enumerate(selections, 1):
        _, pid, row, axis, idx = item
        overlay(Path(row["flair_output"]), Path(row["wt_output"]), pid, axis, idx, title, OUT / f"qc_{n}_{title}.png")
    print(f"Wrote {RESULTS / 'summary_module1.txt'} and per-patient stats; valid={len(patients)} positive={pos_count} negative={neg_count}")


if __name__ == "__main__": main()
