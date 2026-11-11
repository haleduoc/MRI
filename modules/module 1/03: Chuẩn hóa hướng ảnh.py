"""Chuẩn hóa cặp FLAIR–WT sang RAS+ và ghi nhận trục lát axial."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from PIL import Image, ImageDraw


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "results" / "module1" / "Kết quả chuẩn hóa RAS"
REPORT_FIELDS = [
    "patient_id", "status", "reason", "flair_source", "wt_source",
    "flair_orientation_before", "wt_orientation_before", "flair_shape_before",
    "wt_shape_before", "flair_spacing_before", "wt_spacing_before",
    "flair_affine_before", "wt_affine_before", "resampled_wt",
    "resampling_interpolation", "flair_orientation_after", "wt_orientation_after",
    "flair_shape_after", "wt_shape_after", "flair_spacing_after", "wt_spacing_after",
    "flair_affine_after", "wt_affine_after", "maximum_obliquity_degrees",
    "axial_voxel_axis", "axial_axis_note", "flair_output", "wt_output", "overlay_output",
]


# Đọc đầu vào và cho phép giới hạn patient ID để chạy thử một ca trước.
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chuẩn hóa các cặp FLAIR–WT BraTS sang RAS+ và ghi báo cáo hình học."
    )
    parser.add_argument("--manifest", required=True, type=Path, help="patient_manifest.csv")
    parser.add_argument("--qa-report", required=True, type=Path, help="qa_report.csv")
    parser.add_argument("--data-root", required=True, type=Path, help="Gốc để phân giải flair_paths")
    parser.add_argument("--wt-mask-dir", required=True, type=Path, help="Thư mục WT mask hiện có")
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Thư mục kết quả (mặc định: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--patient-id", action="append", default=[],
        help="Chỉ xử lý ID chỉ định; có thể lặp lại để chạy nhiều ca.",
    )
    parser.add_argument(
        "--overlay-patient-id", action="append", default=[],
        help="Tạo overlay kiểm tra cho ID chỉ định; có thể lặp lại.",
    )
    return parser.parse_args()


# Đọc CSV theo ID để mọi ghép cặp dựa trên patient ID, không dựa vào thứ tự dòng.
def read_rows_by_patient(path: Path) -> tuple[dict[str, dict[str, str]], str]:
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as csv_file:
            reader = csv.DictReader(csv_file)
            if not reader.fieldnames or "patient_id" not in reader.fieldnames:
                return {}, f"CSV thiếu cột patient_id: {path}"
            rows: dict[str, dict[str, str]] = {}
            for row in reader:
                patient_id = (row.get("patient_id") or "").strip()
                if patient_id:
                    rows[patient_id] = row
            return rows, ""
    except (OSError, UnicodeError, csv.Error) as error:
        return {}, f"Không đọc được CSV {path}: {error}"


# Tách các đường dẫn manifest được nối bằng dấu chấm phẩy.
def split_paths(value: str) -> list[str]:
    return [part.strip() for part in value.split(";") if part.strip()]


# Chuyển metadata thành JSON để affine và shape có thể đối chiếu trong CSV.
def json_value(value: object) -> str:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=lambda item: item.item())


# Kiểm tra affine trước khi dùng để biến đổi hoặc resample trong không gian vật lý.
def affine_is_valid(affine: np.ndarray) -> bool:
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        return False
    return abs(float(np.linalg.det(affine[:3, :3]))) > 1e-12


# Tạo một overlay PNG trên lát có nhiều voxel WT nhất để kiểm tra vị trí mask.
def save_overlay(flair: np.ndarray, wt: np.ndarray, patient_id: str, output_path: Path) -> None:
    mask_sums = np.count_nonzero(wt, axis=(0, 1))
    slice_index = int(np.argmax(mask_sums))
    flair_slice = np.rot90(flair[:, :, slice_index])
    wt_slice = np.rot90(wt[:, :, slice_index] > 0)
    finite_nonzero = flair_slice[np.isfinite(flair_slice) & (flair_slice > 0)]
    if finite_nonzero.size:
        low, high = np.percentile(finite_nonzero, [1, 99])
    else:
        low, high = 0.0, 1.0
    if high <= low:
        high = low + 1.0
    gray = np.clip((flair_slice - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    rgb[wt_slice] = (0.35 * rgb[wt_slice] + 0.65 * np.array([255, 40, 40])).astype(np.uint8)
    image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 20), fill=(0, 0, 0))
    draw.text((5, 4), f"{patient_id} | RAS+ axial axis 2 | slice {slice_index}", fill="white")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


# Ghi một patient thành hai file mới, dùng cùng orientation transform cho cả cặp.
def process_patient(
    patient_id: str,
    manifest_row: dict[str, str],
    qa_row: dict[str, str],
    data_root: Path,
    wt_mask_dir: Path,
    output_dir: Path,
    create_overlay: bool,
) -> dict[str, str]:
    report = {field: "" for field in REPORT_FIELDS}
    report["patient_id"] = patient_id
    manifest_status = (manifest_row.get("status") or "").strip().casefold()
    flair_paths = split_paths(manifest_row.get("flair_paths") or "")
    segmentation_paths = split_paths(manifest_row.get("segmentation_paths") or "")
    if len(flair_paths) == 1:
        report["flair_source"] = str((data_root / flair_paths[0]).resolve())
    report["wt_source"] = str((wt_mask_dir / f"{patient_id}_wt.nii.gz").resolve())
    if manifest_status != "valid" or len(flair_paths) != 1 or len(segmentation_paths) != 1:
        report["status"] = "not_eligible"
        report["reason"] = manifest_row.get("reason") or "Manifest không có đúng một FLAIR và segmentation."
        return report
    if (qa_row.get("status") or "").strip() != "valid":
        report["status"] = "not_eligible"
        report["reason"] = qa_row.get("reason") or "QA report không đánh dấu cặp hợp lệ."
        return report

    try:
        flair_path = (data_root / flair_paths[0]).resolve(strict=True)
        flair_path.relative_to(data_root.resolve())
        wt_path = (wt_mask_dir / f"{patient_id}_wt.nii.gz").resolve(strict=True)
        if not flair_path.is_file() or not wt_path.is_file():
            raise FileNotFoundError("Đường dẫn FLAIR hoặc WT không phải file.")
        report["flair_source"] = str(flair_path)
        report["wt_source"] = str(wt_path)
        flair_image = nib.load(str(flair_path))
        wt_image = nib.load(str(wt_path))
    except Exception as error:
        report["status"] = "input_error"
        report["reason"] = f"Không mở được file nguồn: {type(error).__name__}: {error}"
        return report

    flair_affine = np.asarray(flair_image.affine, dtype=float)
    wt_affine = np.asarray(wt_image.affine, dtype=float)
    report["flair_shape_before"] = json_value(list(flair_image.shape))
    report["wt_shape_before"] = json_value(list(wt_image.shape))
    report["flair_affine_before"] = json_value(flair_affine)
    report["wt_affine_before"] = json_value(wt_affine)
    report["flair_spacing_before"] = json_value(flair_image.header.get_zooms()[:3])
    report["wt_spacing_before"] = json_value(wt_image.header.get_zooms()[:3])

    if len(flair_image.shape) != 3 or len(wt_image.shape) != 3:
        report["status"] = "invalid_dimensions"
        report["reason"] = "FLAIR và WT phải là ảnh 3D."
        return report
    if not affine_is_valid(flair_affine) or not affine_is_valid(wt_affine):
        report["status"] = "invalid_affine"
        report["reason"] = "Affine FLAIR hoặc WT không hữu hạn hay bị suy biến."
        return report

    try:
        report["flair_orientation_before"] = json_value(nib.aff2axcodes(flair_affine))
        report["wt_orientation_before"] = json_value(nib.aff2axcodes(wt_affine))
        flair_data = np.asarray(flair_image.dataobj)
        wt_data = np.asarray(wt_image.dataobj)
        if not np.isfinite(flair_data).all():
            raise ValueError("FLAIR có NaN hoặc Inf.")
        if not np.isin(np.unique(wt_data), [0, 1]).all():
            report["status"] = "invalid_wt_values"
            report["reason"] = f"WT có giá trị khác 0/1: {np.unique(wt_data).tolist()}"
            return report
    except Exception as error:
        report["status"] = "value_error"
        report["reason"] = f"Lỗi kiểm tra dữ liệu: {type(error).__name__}: {error}"
        return report

    same_grid = flair_image.shape == wt_image.shape and np.allclose(
        flair_affine, wt_affine, atol=1e-5, rtol=0
    )
    report["resampled_wt"] = "no"
    if not same_grid:
        # Resample theo affine vật lý; không đổi shape mask bằng cách ép chỉ số voxel.
        wt_image = resample_from_to(wt_image, (flair_image.shape, flair_affine), order=0)
        wt_data = np.asarray(wt_image.dataobj)
        report["resampled_wt"] = "yes"
        report["resampling_interpolation"] = "nearest-neighbor (order=0)"
        if not np.isin(np.unique(wt_data), [0, 1]).all():
            report["status"] = "invalid_resampled_wt"
            report["reason"] = f"WT sau resample có giá trị khác 0/1: {np.unique(wt_data).tolist()}"
            return report

    # Một phép đổi trục từ FLAIR được áp dụng nguyên vẹn lên cả WT đã cùng lưới.
    source_orientation = nib.orientations.io_orientation(flair_affine)
    ras_orientation = nib.orientations.axcodes2ornt(("R", "A", "S"))
    transform = nib.orientations.ornt_transform(source_orientation, ras_orientation)
    flair_ras = nib.orientations.apply_orientation(flair_data, transform)
    wt_ras = nib.orientations.apply_orientation(wt_data, transform).astype(np.uint8, copy=False)
    ras_affine = flair_affine @ nib.orientations.inv_ornt_aff(transform, flair_image.shape)
    flair_output_dir = output_dir / "nifti" / patient_id
    flair_output_dir.mkdir(parents=True, exist_ok=True)
    flair_output = flair_output_dir / f"{patient_id}_flair_RAS.nii.gz"
    wt_output = flair_output_dir / f"{patient_id}_wt_RAS.nii.gz"
    flair_header = flair_image.header.copy()
    flair_header.set_data_dtype(flair_ras.dtype)
    wt_header = wt_image.header.copy()
    wt_header.set_data_dtype(np.uint8)
    flair_ras_image = nib.Nifti1Image(flair_ras, ras_affine, header=flair_header)
    wt_ras_image = nib.Nifti1Image(wt_ras, ras_affine, header=wt_header)
    flair_ras_image.set_qform(ras_affine, code=1)
    flair_ras_image.set_sform(ras_affine, code=1)
    wt_ras_image.set_qform(ras_affine, code=1)
    wt_ras_image.set_sform(ras_affine, code=1)
    nib.save(flair_ras_image, str(flair_output))
    nib.save(wt_ras_image, str(wt_output))

    output_spacing = flair_ras_image.header.get_zooms()[:3]
    orientation_after = nib.aff2axcodes(ras_affine)
    obliquity_degrees = np.degrees(nib.affines.obliquity(ras_affine))
    maximum_obliquity = float(np.max(obliquity_degrees))
    superior_direction = ras_affine[:3, :3] / np.linalg.norm(ras_affine[:3, :3], axis=0)
    axial_axis = int(np.argmax(np.abs(superior_direction[2, :])))
    if maximum_obliquity > 5.0:
        axial_note = f"Trục {axial_axis} gần +S nhất nhưng oblique {maximum_obliquity:.2f}°; lát không thuần axial."
    else:
        axial_note = f"RAS+ và oblique không quá 5°; dùng trục voxel {axial_axis} cho lát axial."

    overlay_path = output_dir / "overlays" / f"{patient_id}_axial_overlay.png" if create_overlay else None
    if overlay_path is not None:
        save_overlay(flair_ras, wt_ras, patient_id, overlay_path)
    report.update({
        "status": "valid_resampled" if not same_grid else "valid",
        "reason": "",
        "flair_orientation_after": json_value(orientation_after),
        "wt_orientation_after": json_value(nib.aff2axcodes(ras_affine)),
        "flair_shape_after": json_value(list(flair_ras.shape)),
        "wt_shape_after": json_value(list(wt_ras.shape)),
        "flair_spacing_after": json_value(output_spacing),
        "wt_spacing_after": json_value(wt_ras_image.header.get_zooms()[:3]),
        "flair_affine_after": json_value(ras_affine),
        "wt_affine_after": json_value(ras_affine),
        "maximum_obliquity_degrees": f"{maximum_obliquity:.6f}",
        "axial_voxel_axis": str(axial_axis),
        "axial_axis_note": axial_note,
        "flair_output": str(flair_output),
        "wt_output": str(wt_output),
        "overlay_output": str(overlay_path) if overlay_path else "",
    })
    return report


# Ghi báo cáo cho cả ID hợp lệ lẫn các ca bị loại do thiếu hoặc sai đầu vào.
def write_report(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


# Chạy các ca đã chọn, ghi báo cáo và in số ca theo trạng thái.
def main() -> int:
    arguments = parse_arguments()
    manifest_rows, manifest_error = read_rows_by_patient(arguments.manifest)
    qa_rows, qa_error = read_rows_by_patient(arguments.qa_report)
    if manifest_error or qa_error:
        for error in (manifest_error, qa_error):
            if error:
                print(error)
        return 2
    data_root = arguments.data_root.resolve()
    wt_mask_dir = arguments.wt_mask_dir.resolve()
    if not data_root.is_dir() or not wt_mask_dir.is_dir():
        print(f"Không tìm thấy data-root hoặc thư mục WT mask: {data_root} | {wt_mask_dir}")
        return 2

    patient_ids = sorted(manifest_rows)
    if arguments.patient_id:
        requested = set(arguments.patient_id)
        unknown = requested - set(patient_ids)
        if unknown:
            print("Patient ID không có trong manifest: " + ", ".join(sorted(unknown)))
            return 2
        patient_ids = [patient_id for patient_id in patient_ids if patient_id in requested]
    rows: list[dict[str, str]] = []
    overlay_ids = set(arguments.overlay_patient_id)
    for patient_id in patient_ids:
        qa_row = qa_rows.get(patient_id, {"status": "", "reason": "Không có patient ID trong QA report."})
        try:
            patient_row = process_patient(
                patient_id, manifest_rows[patient_id], qa_row, data_root, wt_mask_dir,
                arguments.output_dir, patient_id in overlay_ids,
            )
        except Exception as error:
            patient_row = {field: "" for field in REPORT_FIELDS}
            patient_row["patient_id"] = patient_id
            patient_row["status"] = "processing_error"
            patient_row["reason"] = f"{type(error).__name__}: {error}"
        rows.append(patient_row)
    report_path = arguments.output_dir / "orientation_report.csv"
    write_report(report_path, rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(f"Patient ID được ghi: {len(rows)}")
    print(f"Theo trạng thái: {counts}")
    print(f"Báo cáo hướng ảnh: {report_path.resolve()}")
    return 0 if all(row["status"] in {"valid", "valid_resampled", "not_eligible"} for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
