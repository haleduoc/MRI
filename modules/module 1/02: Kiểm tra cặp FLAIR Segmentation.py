"""Kiểm tra cặp FLAIR-segmentation NIfTI từ patient_manifest.csv."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

try:
    import nibabel as nib
    import numpy as np
except ImportError as error:
    raise SystemExit(
        "Thiếu thư viện cần thiết. Cài bằng: "
        "python -m pip install numpy nibabel scipy"
    ) from error


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "results" / "module1" / "nifti_validation"
ALLOWED_SEGMENTATION_LABELS = {0, 1, 2, 4}
REQUIRED_MANIFEST_COLUMNS = {
    "patient_id",
    "flair_paths",
    "segmentation_paths",
    "status",
    "reason",
}

FILE_REPORT_FIELDS = [
    "read_success",
    "ndim",
    "shape",
    "affine",
    "affine_valid",
    "voxel_spacing",
    "orientation",
    "qform",
    "qform_code",
    "sform",
    "sform_code",
    "has_nan",
    "has_inf",
    "minimum",
    "maximum",
    "nonzero_voxels",
    "labels_are_integer",
    "unexpected_labels",
    "labels_valid",
]


# Đọc các tham số và đặt thư mục kết quả mặc định trong results/module1.
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kiểm tra hình học và giá trị của các cặp NIfTI FLAIR-segmentation."
    )
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Đường dẫn đến patient_manifest.csv.",
    )
    parser.add_argument(
        "--data-root",
        required=True,
        type=Path,
        help="Thư mục gốc dataset, dùng để phân giải đường dẫn tương đối trong CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Thư mục lưu báo cáo và mask (mặc định: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--resample-to-flair",
        action="store_true",
        help="Căn segmentation về lưới FLAIR bằng nội suy nearest-neighbor.",
    )
    return parser.parse_args()


# Đọc manifest và gom các dòng theo patient ID để đầu ra chỉ có một dòng mỗi ID.
def read_manifest(
    manifest_path: Path,
) -> tuple[dict[str, list[dict[str, str]]], list[str]]:
    rows_by_patient: dict[str, list[dict[str, str]]] = {}
    csv_errors: list[str] = []

    try:
        with manifest_path.open("r", newline="", encoding="utf-8-sig") as csv_file:
            reader = csv.DictReader(csv_file)
            column_names = set(reader.fieldnames or [])
            missing_columns = sorted(REQUIRED_MANIFEST_COLUMNS - column_names)
            if missing_columns:
                return {}, [
                    "Manifest thiếu cột bắt buộc: " + ", ".join(missing_columns)
                ]

            for row_number, row in enumerate(reader, start=2):
                patient_id = (row.get("patient_id") or "").strip()
                if not patient_id:
                    csv_errors.append(
                        f"Dòng {row_number} trong manifest không có patient_id."
                    )
                    continue
                if None in row:
                    row["_csv_error"] = (
                        f"Dòng {row_number} có nhiều giá trị hơn số cột trong header."
                    )
                rows_by_patient.setdefault(patient_id, []).append(row)
            if not rows_by_patient and not csv_errors:
                csv_errors.append("Manifest không có patient ID nào để kiểm tra.")
    except (OSError, UnicodeError, csv.Error) as error:
        csv_errors.append(f"Không đọc được manifest {manifest_path}: {error}")

    return rows_by_patient, csv_errors


# Tách các đường dẫn được nối bằng dấu chấm phẩy trong manifest.
def split_paths(value: str | None) -> list[str]:
    if not value:
        return []
    return [path.strip() for path in value.split(";") if path.strip()]


# Ngăn đường dẫn trong CSV thoát khỏi thư mục dataset.
def resolve_dataset_path(data_root: Path, relative_path: str) -> tuple[Path | None, str]:
    try:
        candidate = (data_root / relative_path).resolve()
        candidate.relative_to(data_root)
        exists_as_file = candidate.is_file()
    except (OSError, RuntimeError, ValueError) as error:
        if isinstance(error, (OSError, RuntimeError)):
            return None, f"Không phân giải được đường dẫn {relative_path}: {error}"
        return None, f"Đường dẫn nằm ngoài data-root: {relative_path}"

    if not exists_as_file:
        return None, f"Không tìm thấy file: {relative_path}"
    if not candidate.name.casefold().endswith((".nii", ".nii.gz")):
        return None, f"File không có đuôi .nii hoặc .nii.gz: {relative_path}"
    return candidate, ""


# Kiểm tra ma trận affine có hữu hạn và không làm suy biến lưới không gian.
def is_valid_affine(affine: np.ndarray) -> bool:
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        return False
    try:
        determinant = float(np.linalg.det(affine[:3, :3]))
    except np.linalg.LinAlgError:
        return False
    return abs(determinant) > 1e-12


# Chuyển ma trận và danh sách số thành chuỗi JSON để ghi vào một ô CSV.
def to_json(value: Any) -> str:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        default=lambda item: item.item(),
    )


# Lấy thông tin hình học và dữ liệu voxel của một file, đồng thời giữ lỗi cụ thể.
def inspect_nifti(
    file_path: Path, modality: str
) -> tuple[Any | None, np.ndarray | None, dict[str, str], list[str]]:
    image_info = {field: "" for field in FILE_REPORT_FIELDS}
    image_info["read_success"] = "no"
    errors: list[str] = []

    try:
        image = nib.load(str(file_path))
        data = np.asanyarray(image.dataobj)
    except Exception as error:
        errors.append(f"{modality} read error: {type(error).__name__}: {error}")
        return None, None, image_info, errors

    image_info["read_success"] = "yes"
    image_info["ndim"] = str(data.ndim)
    image_info["shape"] = to_json(list(data.shape))
    affine = np.asarray(image.affine, dtype=float)
    affine_valid = is_valid_affine(affine)
    image_info["affine"] = to_json(affine)
    image_info["affine_valid"] = "yes" if affine_valid else "no"
    image_info["voxel_spacing"] = to_json(list(image.header.get_zooms()))

    if affine_valid:
        try:
            image_info["orientation"] = to_json(nib.aff2axcodes(affine))
        except Exception as error:
            errors.append(f"{modality} orientation error: {error}")
    else:
        errors.append(f"{modality} affine không hợp lệ.")

    try:
        qform, qform_code = image.get_qform(coded=True)
        if qform is not None:
            image_info["qform"] = to_json(qform)
        image_info["qform_code"] = str(int(qform_code or 0))
    except Exception as error:
        errors.append(f"{modality} qform error: {error}")
    try:
        sform, sform_code = image.get_sform(coded=True)
        if sform is not None:
            image_info["sform"] = to_json(sform)
        image_info["sform_code"] = str(int(sform_code or 0))
    except Exception as error:
        errors.append(f"{modality} sform error: {error}")

    if not np.issubdtype(data.dtype, np.number):
        image_info["has_nan"] = "unknown"
        image_info["has_inf"] = "unknown"
        errors.append(f"{modality} không chứa dữ liệu số.")
        if modality == "segmentation":
            image_info["labels_are_integer"] = "no"
            image_info["unexpected_labels"] = "[]"
            image_info["labels_valid"] = "no"
        return image, data, image_info, errors

    has_nan = bool(np.isnan(data).any())
    has_inf = bool(np.isinf(data).any())
    image_info["has_nan"] = "yes" if has_nan else "no"
    image_info["has_inf"] = "yes" if has_inf else "no"
    if has_nan:
        errors.append(f"{modality} có NaN.")
    if has_inf:
        errors.append(f"{modality} có Inf.")

    if np.iscomplexobj(data):
        errors.append(f"{modality} chứa dữ liệu phức, cần dữ liệu số thực.")
        image_info["nonzero_voxels"] = str(int(np.count_nonzero(data)))
        if modality == "segmentation":
            image_info["labels_are_integer"] = "no"
            image_info["unexpected_labels"] = "[]"
            image_info["labels_valid"] = "no"
        return image, data, image_info, errors

    finite_values = data[np.isfinite(data)]
    if finite_values.size:
        image_info["minimum"] = str(np.min(finite_values).item())
        image_info["maximum"] = str(np.max(finite_values).item())
    image_info["nonzero_voxels"] = str(int(np.count_nonzero(data)))

    if modality == "segmentation":
        finite_labels = np.unique(finite_values)
        labels_are_integer = bool(
            np.equal(finite_labels, np.rint(finite_labels)).all()
        )
        unexpected_labels = []
        for label in finite_labels:
            if label not in ALLOWED_SEGMENTATION_LABELS:
                unexpected_labels.append(float(label))

        labels_valid = (
            labels_are_integer
            and not unexpected_labels
            and not has_nan
            and not has_inf
        )
        image_info["labels_are_integer"] = "yes" if labels_are_integer else "no"
        image_info["unexpected_labels"] = to_json(unexpected_labels)
        image_info["labels_valid"] = "yes" if labels_valid else "no"
        if not labels_are_integer:
            errors.append("Segmentation chứa nhãn không nguyên.")
        if unexpected_labels:
            errors.append(
                "Segmentation có nhãn ngoài {0, 1, 2, 4}: "
                + to_json(unexpected_labels)
            )

    return image, data, image_info, errors


# Tạo một dòng CSV với đủ cột, kể cả khi một ca lỗi trước lúc đọc ảnh.
def new_report_row(patient_id: str) -> dict[str, str]:
    row = {
        "patient_id": patient_id,
        "status": "",
        "reason": "",
        "manifest_status": "",
        "manifest_reason": "",
        "flair_path": "",
        "segmentation_path": "",
        "same_grid": "",
        "resampled_segmentation_path": "",
        "resampling_interpolation": "",
        "target_shape": "",
        "target_affine": "",
        "resampled_segmentation_labels_valid": "",
        "wt_mask_path": "",
        "errors": "",
    }
    for modality in ("flair", "segmentation"):
        for field in FILE_REPORT_FIELDS:
            row[f"{modality}_{field}"] = ""
    return row


# Ghi chú lỗi theo bệnh nhân thay vì dừng toàn bộ lần kiểm tra.
def add_error(row: dict[str, str], errors: list[str], message: str) -> None:
    errors.append(message)
    row["errors"] = "; ".join(errors)
    row["reason"] = row["errors"]


# So sánh cả shape lẫn affine để xác định hai ảnh có cùng lưới hay không.
def images_share_grid(flair_image: Any, segmentation_image: Any) -> bool:
    same_shape = tuple(flair_image.shape) == tuple(segmentation_image.shape)
    same_affine = np.allclose(
        flair_image.affine,
        segmentation_image.affine,
        rtol=0.0,
        atol=1e-5,
    )
    return same_shape and same_affine


# Kiểm tra segmentation chỉ chứa nhãn nguyên trong tập {0, 1, 2, 4}.
def segmentation_label_errors(data: np.ndarray) -> list[str]:
    if not np.issubdtype(data.dtype, np.number) or np.iscomplexobj(data):
        return ["Segmentation không phải dữ liệu số thực."]

    if not np.isfinite(data).all():
        return ["Segmentation có NaN hoặc Inf."]

    labels = np.unique(data)
    errors = []
    if not np.equal(labels, np.rint(labels)).all():
        errors.append("Segmentation chứa nhãn không nguyên.")
    unexpected_labels = [float(label) for label in labels if label not in ALLOWED_SEGMENTATION_LABELS]
    if unexpected_labels:
        errors.append(
            "Segmentation có nhãn ngoài {0, 1, 2, 4}: "
            + to_json(unexpected_labels)
        )
    return errors


# Resample segmentation bằng nearest-neighbor về đúng shape và affine của FLAIR.
def resample_segmentation(
    segmentation_image: Any, flair_image: Any, output_path: Path
) -> tuple[Any, np.ndarray]:
    if len(flair_image.shape) != 3 or len(segmentation_image.shape) != 3:
        raise ValueError("Chỉ hỗ trợ resample cặp ảnh 3D.")

    from nibabel.processing import resample_from_to

    aligned_image = resample_from_to(
        segmentation_image,
        (flair_image.shape, flair_image.affine),
        order=0,
    )
    aligned_data = np.asanyarray(aligned_image.dataobj)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(aligned_image, str(output_path))
    return aligned_image, aligned_data


# Tạo mask WT uint8 trên cùng lưới với FLAIR, không thay đổi segmentation gốc.
def save_wt_mask(
    segmentation_data: np.ndarray, flair_image: Any, output_path: Path
) -> None:
    wt_mask = np.isin(segmentation_data, [1, 2, 4]).astype(np.uint8)
    mask_header = flair_image.header.copy()
    mask_header.set_data_dtype(np.uint8)
    mask_image = nib.Nifti1Image(wt_mask, flair_image.affine, header=mask_header)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(mask_image, str(output_path))


# Đóng các NIfTI phát sinh trong lần chạy thành một file ZIP duy nhất.
def package_nifti_outputs(output_dir: Path) -> Path | None:
    nifti_directories = [
        output_dir / "wt_masks",
        output_dir / "aligned_segmentations",
    ]
    nifti_files = []
    for directory in nifti_directories:
        if not directory.is_dir():
            continue
        for file_path in directory.rglob("*"):
            if file_path.is_file() and file_path.name.casefold().endswith(
                (".nii", ".nii.gz")
            ):
                nifti_files.append(file_path)

    archive_path = output_dir / "nifti_outputs.zip"
    if not nifti_files:
        if archive_path.exists():
            archive_path.unlink()
        return None

    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        for file_path in sorted(nifti_files):
            archive.write(file_path, arcname=file_path.relative_to(output_dir))

    # Chỉ xóa các thư mục NIfTI sau khi file ZIP đã được ghi thành công.
    for directory in nifti_directories:
        if directory.exists():
            shutil.rmtree(directory)
    return archive_path


# Ghi đường dẫn mask/alignment theo dạng tên ZIP và đường dẫn bên trong ZIP.
def update_report_archive_paths(
    report_rows: list[dict[str, str]], archive_path: Path | None
) -> None:
    if archive_path is None:
        return
    for row in report_rows:
        for field in ("wt_mask_path", "resampled_segmentation_path"):
            if row[field]:
                row[field] = f"{archive_path.name}::{row[field]}"


# Kiểm tra một bệnh nhân từ đường dẫn trong manifest đến mask WT.
def validate_patient(
    patient_id: str,
    manifest_row: dict[str, str],
    data_root: Path,
    output_dir: Path,
    resample_to_flair: bool,
) -> dict[str, str]:
    report_row = new_report_row(patient_id)
    errors: list[str] = []
    report_row["manifest_status"] = manifest_row.get("status", "") or ""
    report_row["manifest_reason"] = manifest_row.get("reason", "") or ""
    if manifest_row.get("_csv_error"):
        report_row["status"] = "csv_error"
        report_row["reason"] = manifest_row["_csv_error"]
        report_row["errors"] = report_row["reason"]
        return report_row
    flair_paths = split_paths(manifest_row.get("flair_paths"))
    segmentation_paths = split_paths(manifest_row.get("segmentation_paths"))
    report_row["flair_path"] = ";".join(flair_paths)
    report_row["segmentation_path"] = ";".join(segmentation_paths)

    has_duplicates = len(flair_paths) > 1 or len(segmentation_paths) > 1
    has_missing_paths = not flair_paths or not segmentation_paths
    if has_duplicates or has_missing_paths:
        if not flair_paths:
            add_error(report_row, errors, "Thiếu đường dẫn FLAIR trong manifest.")
        if len(flair_paths) > 1:
            add_error(report_row, errors, "Có nhiều đường dẫn FLAIR trong manifest.")
        if not segmentation_paths:
            add_error(report_row, errors, "Thiếu đường dẫn segmentation trong manifest.")
        if len(segmentation_paths) > 1:
            add_error(
                report_row, errors, "Có nhiều đường dẫn segmentation trong manifest."
            )
        report_row["status"] = "duplicate_file" if has_duplicates else "missing_file"
        return report_row

    flair_path, flair_path_error = resolve_dataset_path(data_root, flair_paths[0])
    segmentation_path, segmentation_path_error = resolve_dataset_path(
        data_root, segmentation_paths[0]
    )
    if flair_path_error:
        add_error(report_row, errors, "FLAIR: " + flair_path_error)
    if segmentation_path_error:
        add_error(report_row, errors, "Segmentation: " + segmentation_path_error)
    if flair_path is None or segmentation_path is None:
        report_row["status"] = "missing_file"
        return report_row

    flair_image, flair_data, flair_info, flair_errors = inspect_nifti(
        flair_path, "flair"
    )
    segmentation_image, segmentation_data, segmentation_info, segmentation_errors = (
        inspect_nifti(segmentation_path, "segmentation")
    )
    for field, value in flair_info.items():
        report_row[f"flair_{field}"] = value
    for field, value in segmentation_info.items():
        report_row[f"segmentation_{field}"] = value
    for error in flair_errors + segmentation_errors:
        add_error(report_row, errors, error)

    if flair_image is None or segmentation_image is None:
        report_row["status"] = "read_error"
        return report_row

    if any(
        " orientation error:" in error
        or " qform error:" in error
        or " sform error:" in error
        for error in errors
    ):
        report_row["status"] = "geometry_error"
        return report_row

    if flair_info["affine_valid"] != "yes" or segmentation_info["affine_valid"] != "yes":
        report_row["status"] = "invalid_affine"
        return report_row

    has_invalid_values = (
        flair_info["has_nan"] == "yes"
        or flair_info["has_inf"] == "yes"
        or segmentation_info["has_nan"] == "yes"
        or segmentation_info["has_inf"] == "yes"
    )
    if flair_data is None or segmentation_data is None:
        report_row["status"] = "read_error"
        return report_row

    if not np.issubdtype(flair_data.dtype, np.number) or np.iscomplexobj(flair_data):
        has_invalid_values = True
    if not np.issubdtype(segmentation_data.dtype, np.number) or np.iscomplexobj(
        segmentation_data
    ):
        has_invalid_values = True

    label_errors = segmentation_label_errors(segmentation_data)
    for error in label_errors:
        add_error(report_row, errors, error)

    same_grid = images_share_grid(flair_image, segmentation_image)
    report_row["same_grid"] = "yes" if same_grid else "no"
    aligned_segmentation = segmentation_image
    aligned_data = segmentation_data
    was_resampled = False
    aligned_label_errors: list[str] = []
    safe_patient_id = re.sub(r"[^A-Za-z0-9_.-]", "_", patient_id)

    if not same_grid:
        if not resample_to_flair:
            add_error(report_row, errors, "Ảnh không cùng shape và affine.")
            report_row["status"] = "requires_alignment"
            return report_row

        aligned_path = output_dir / "aligned_segmentations" / f"{safe_patient_id}_seg.nii.gz"
        try:
            aligned_segmentation, aligned_data = resample_segmentation(
                segmentation_image, flair_image, aligned_path
            )
        except Exception as error:
            add_error(report_row, errors, f"Không resample được segmentation: {error}")
            report_row["status"] = "requires_alignment"
            return report_row

        was_resampled = True
        report_row["resampled_segmentation_path"] = str(
            aligned_path.relative_to(output_dir)
        )
        report_row["resampling_interpolation"] = "nearest-neighbor (order=0)"
        report_row["target_shape"] = to_json(list(flair_image.shape))
        report_row["target_affine"] = to_json(np.asarray(flair_image.affine))

        aligned_label_errors = segmentation_label_errors(aligned_data)
        report_row["resampled_segmentation_labels_valid"] = (
            "no" if aligned_label_errors else "yes"
        )
        for error in aligned_label_errors:
            add_error(report_row, errors, "Sau resample: " + error)

    if has_invalid_values:
        report_row["status"] = "invalid_values"
        return report_row
    if label_errors or aligned_label_errors:
        report_row["status"] = "invalid_labels"
        return report_row

    mask_path = output_dir / "wt_masks" / f"{safe_patient_id}_wt.nii.gz"
    try:
        save_wt_mask(aligned_data, flair_image, mask_path)
    except Exception as error:
        add_error(report_row, errors, f"Không ghi được mask WT: {error}")
        report_row["status"] = "write_error"
        return report_row

    report_row["wt_mask_path"] = str(mask_path.relative_to(output_dir))
    report_row["status"] = "resampled" if was_resampled else "valid"
    return report_row


# Tạo các tên cột chi tiết cho metadata của cả FLAIR và segmentation.
def get_report_columns() -> list[str]:
    columns = [
        "patient_id",
        "status",
        "reason",
        "manifest_status",
        "manifest_reason",
        "flair_path",
        "segmentation_path",
    ]
    for modality in ("flair", "segmentation"):
        for field in FILE_REPORT_FIELDS:
            columns.append(f"{modality}_{field}")
    columns.extend(
        [
            "same_grid",
            "resampled_segmentation_path",
            "resampling_interpolation",
            "target_shape",
            "target_affine",
            "resampled_segmentation_labels_valid",
            "wt_mask_path",
            "errors",
        ]
    )
    return columns


# Tạo một dòng báo cáo cho lỗi manifest không gắn được với patient ID cụ thể.
def manifest_error_row(message: str) -> dict[str, str]:
    row = new_report_row("__manifest__")
    row["status"] = "manifest_error"
    row["reason"] = message
    row["errors"] = message
    return row


# Ghi số ca và các nhóm lỗi để người dùng đọc nhanh mà không mở CSV.
def write_summary(
    output_path: Path,
    report_rows: list[dict[str, str]],
    manifest_errors: list[str],
    nifti_archive: Path | None,
) -> None:
    total_patients = sum(1 for row in report_rows if row["patient_id"] != "__manifest__")
    status_counts: dict[str, int] = {}
    counts = {
        "valid_pairs": 0,
        "missing_or_duplicate": 0,
        "read_errors": 0,
        "invalid_values": 0,
        "nan_or_inf": 0,
        "invalid_labels": 0,
        "different_grids": 0,
        "requires_alignment": 0,
        "resampled": 0,
        "csv_errors": 0,
    }

    for row in report_rows:
        status = row["status"]
        status_counts[status] = status_counts.get(status, 0) + 1
        if status in {"valid", "resampled"}:
            counts["valid_pairs"] += 1
        if status in {"missing_file", "duplicate_file"}:
            counts["missing_or_duplicate"] += 1
        if status == "read_error":
            counts["read_errors"] += 1
        if status == "csv_error":
            counts["csv_errors"] += 1
        if status == "invalid_values" or any(
            row[f"{modality}_has_nan"] == "yes"
            or row[f"{modality}_has_inf"] == "yes"
            for modality in ("flair", "segmentation")
        ):
            counts["invalid_values"] += 1
        if (
            row["segmentation_labels_valid"] == "no"
            or row["resampled_segmentation_labels_valid"] == "no"
        ):
            counts["invalid_labels"] += 1
        if status == "requires_alignment":
            counts["requires_alignment"] += 1
        if row["resampling_interpolation"]:
            counts["resampled"] += 1
        if row["same_grid"] == "no":
            counts["different_grids"] += 1
        if any(
            row[f"{modality}_has_nan"] == "yes"
            or row[f"{modality}_has_inf"] == "yes"
            for modality in ("flair", "segmentation")
        ):
            counts["nan_or_inf"] += 1

    with output_path.open("w", encoding="utf-8") as summary_file:
        summary_file.write(f"Tổng số patient ID: {total_patients}\n")
        summary_file.write(f"Cặp hợp lệ: {counts['valid_pairs']}\n")
        summary_file.write(f"Ca thiếu hoặc trùng file: {counts['missing_or_duplicate']}\n")
        summary_file.write(f"Lỗi đọc NIfTI: {counts['read_errors']}\n")
        summary_file.write(
            f"Lỗi manifest hoặc đầu vào: {counts['csv_errors'] + len(manifest_errors)}\n"
        )
        summary_file.write(f"Giá trị dữ liệu không hợp lệ: {counts['invalid_values']}\n")
        summary_file.write(f"Ca có NaN hoặc Inf: {counts['nan_or_inf']}\n")
        summary_file.write(f"Ca có nhãn segmentation không hợp lệ: {counts['invalid_labels']}\n")
        summary_file.write(f"Cặp khác lưới: {counts['different_grids']}\n")
        summary_file.write(f"Cần căn chỉnh: {counts['requires_alignment']}\n")
        summary_file.write(f"Đã resample: {counts['resampled']}\n")
        summary_file.write(
            "Gói NIfTI: "
            + (nifti_archive.name if nifti_archive is not None else "không có")
            + "\n"
        )
        summary_file.write("\nSố ca theo trạng thái:\n")
        for status in sorted(status_counts):
            summary_file.write(f"- {status}: {status_counts[status]}\n")
        if manifest_errors:
            summary_file.write("\nLỗi manifest:\n")
            for error in manifest_errors:
                summary_file.write(f"- {error}\n")


# Điều phối đọc CSV, kiểm tra từng ca và ghi toàn bộ kết quả vào output-dir.
def main() -> int:
    arguments = parse_arguments()
    manifest_path = arguments.manifest.expanduser().resolve()
    data_root = arguments.data_root.expanduser().resolve()
    output_dir = arguments.output_dir.expanduser().resolve()

    # Không cho ghi báo cáo hoặc mask vào bên trong thư mục dữ liệu nguồn.
    try:
        output_dir.relative_to(data_root)
    except ValueError:
        pass
    else:
        print(
            f"Output directory không được nằm trong data-root: {output_dir}",
            file=sys.stderr,
        )
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_groups, manifest_errors = read_manifest(manifest_path)
    report_rows = []
    if not data_root.is_dir():
        manifest_errors.append(f"Không tồn tại data-root: {data_root}")
    else:
        for patient_id in sorted(manifest_groups):
            patient_rows = manifest_groups[patient_id]
            if len(patient_rows) != 1:
                row = new_report_row(patient_id)
                row["status"] = "csv_error"
                row["reason"] = "Patient ID xuất hiện nhiều hơn một lần trong manifest."
                row["errors"] = row["reason"]
            else:
                row = validate_patient(
                    patient_id,
                    patient_rows[0],
                    data_root,
                    output_dir,
                    arguments.resample_to_flair,
                )
            report_rows.append(row)

    for error in manifest_errors:
        report_rows.append(manifest_error_row(error))

    nifti_archive = package_nifti_outputs(output_dir)
    update_report_archive_paths(report_rows, nifti_archive)

    qa_report_path = output_dir / "qa_report.csv"
    with qa_report_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=get_report_columns())
        writer.writeheader()
        writer.writerows(report_rows)
    write_summary(
        output_dir / "summary.txt",
        report_rows,
        manifest_errors,
        nifti_archive,
    )

    patient_count = sum(
        1 for row in report_rows if row["patient_id"] != "__manifest__"
    )
    print(f"Số patient ID trong báo cáo: {patient_count}")
    print(f"Báo cáo QA: {qa_report_path}")
    print(f"Tóm tắt: {output_dir / 'summary.txt'}")
    if nifti_archive is not None:
        print(f"Gói NIfTI: {nifti_archive}")
    if manifest_errors:
        print(f"Có {len(manifest_errors)} lỗi manifest hoặc đầu vào; xem summary.txt.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
