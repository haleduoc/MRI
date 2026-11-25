# đọc manifest; kiểm tra hình học và nhãn; căn lưới segmentation; xuất QA và WT.

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from collections import Counter
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
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "results" / "module1" / "Kết quả kiểm tra từng cặp"
ALLOWED_SEGMENTATION_LABELS = {0, 1, 2, 4}
REQUIRED_MANIFEST_COLUMNS = {
    "patient_id",
    "flair_paths",
    "segmentation_paths",
    "status",
    "reason",
}

FILE_REPORT_FIELDS = """read_success ndim shape affine affine_valid voxel_spacing orientation
qform qform_code sform sform_code has_nan has_inf minimum maximum nonzero_voxels
labels_are_integer unexpected_labels labels_valid""".split()


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


def read_manifest(
    manifest_path: Path,
) -> tuple[dict[str, list[dict[str, str]]], list[str]]:
    rows_by_patient: dict[str, list[dict[str, str]]] = {}
    csv_errors: list[str] = []
    try:
        with manifest_path.open("r", newline="", encoding="utf-8-sig") as csv_file:
            reader = csv.DictReader(csv_file)
            missing_columns = sorted(REQUIRED_MANIFEST_COLUMNS - set(reader.fieldnames or []))
            if missing_columns:
                return {}, ["Manifest thiếu cột bắt buộc: " + ", ".join(missing_columns)]
            for row_number, row in enumerate(reader, start=2):
                patient_id = (row.get("patient_id") or "").strip()
                if not patient_id:
                    csv_errors.append(f"Dòng {row_number} trong manifest không có patient_id.")
                    continue
                if None in row:
                    # DictReader đưa phần dư vào khóa None; đánh dấu ca thay vì làm lệch cột.
                    row["_csv_error"] = f"Dòng {row_number} có nhiều giá trị hơn số cột trong header."
                rows_by_patient.setdefault(patient_id, []).append(row)
            if not rows_by_patient and not csv_errors:
                csv_errors.append("Manifest không có patient ID nào để kiểm tra.")
    except (OSError, UnicodeError, csv.Error) as error:
        csv_errors.append(f"Không đọc được manifest {manifest_path}: {error}")

    return rows_by_patient, csv_errors


def split_paths(value: str | None) -> list[str]:
    return [path.strip() for path in (value or "").split(";") if path.strip()]


def resolve_dataset_path(data_root: Path, relative_path: str) -> tuple[Path | None, str]:
    try:
        candidate = (data_root / relative_path).resolve()
        # CSV chỉ được mở trong data-root; resolve xử lý cả ../ và symlink.
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


def is_valid_affine(affine: np.ndarray) -> bool:
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        return False
    try:
        determinant = float(np.linalg.det(affine[:3, :3]))
    except np.linalg.LinAlgError:
        return False
    return abs(determinant) > 1e-12


def to_json(value: Any) -> str:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        default=lambda item: item.item(),
    )


def inspect_nifti(
    file_path: Path, modality: str
) -> tuple[Any | None, np.ndarray | None, dict[str, str], list[str]]:
    image_info = {field: "" for field in FILE_REPORT_FIELDS}
    image_info["read_success"] = "no"
    try:
        image = nib.load(str(file_path))
        data = np.asanyarray(image.dataobj)
    except Exception as error:
        return None, None, image_info, [f"{modality} read error: {type(error).__name__}: {error}"]

    errors: list[str] = []
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

    # Lưu cả hai transform NIfTI để QA có thể phát hiện metadata bất nhất.
    for form in ("qform", "sform"):
        try:
            matrix, code = getattr(image, f"get_{form}")(coded=True)
            if matrix is not None:
                image_info[form] = to_json(matrix)
            image_info[f"{form}_code"] = str(int(code or 0))
        except Exception as error:
            errors.append(f"{modality} {form} error: {error}")

    if not np.issubdtype(data.dtype, np.number):
        image_info["has_nan"] = "unknown"
        image_info["has_inf"] = "unknown"
        errors.append(f"{modality} không chứa dữ liệu số.")
        if modality == "segmentation":
            image_info.update(labels_are_integer="no", unexpected_labels="[]", labels_valid="no")
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
            image_info.update(labels_are_integer="no", unexpected_labels="[]", labels_valid="no")
        return image, data, image_info, errors

    finite_values = data[np.isfinite(data)]
    if finite_values.size:
        image_info["minimum"], image_info["maximum"] = (
            str(np.min(finite_values).item()), str(np.max(finite_values).item())
        )
    image_info["nonzero_voxels"] = str(int(np.count_nonzero(data)))

    if modality == "segmentation":
        # Mask BraTS cho phép nhãn nguyên 0/1/2/4; bỏ qua giá trị không hữu hạn ở thống kê.
        finite_labels = np.unique(finite_values)
        labels_are_integer = bool(
            np.equal(finite_labels, np.rint(finite_labels)).all()
        )
        unexpected_labels = [float(label) for label in finite_labels if label not in ALLOWED_SEGMENTATION_LABELS]
        labels_valid = labels_are_integer and not unexpected_labels and not has_nan and not has_inf
        image_info.update(
            labels_are_integer="yes" if labels_are_integer else "no",
            unexpected_labels=to_json(unexpected_labels),
            labels_valid="yes" if labels_valid else "no",
        )
        if not labels_are_integer:
            errors.append("Segmentation chứa nhãn không nguyên.")
        if unexpected_labels:
            errors.append(
                "Segmentation có nhãn ngoài {0, 1, 2, 4}: "
                + to_json(unexpected_labels)
            )

    return image, data, image_info, errors


def new_report_row(patient_id: str) -> dict[str, str]:
    row = {
        "patient_id": patient_id,
        **dict.fromkeys((
            "status", "reason", "manifest_status", "manifest_reason", "flair_path",
            "segmentation_path", "same_grid", "resampled_segmentation_path",
            "resampling_interpolation", "target_shape", "target_affine",
            "resampled_segmentation_labels_valid", "wt_mask_path", "errors",
        ), ""),
    }
    for modality in ("flair", "segmentation"):
        for field in FILE_REPORT_FIELDS:
            row[f"{modality}_{field}"] = ""
    return row


def add_error(row: dict[str, str], errors: list[str], message: str) -> None:
    errors.append(message)
    row["errors"] = "; ".join(errors)
    row["reason"] = row["errors"]


def images_share_grid(flair_image: Any, segmentation_image: Any) -> bool:
    same_shape = tuple(flair_image.shape) == tuple(segmentation_image.shape)
    same_affine = np.allclose(
        flair_image.affine,
        segmentation_image.affine,
        rtol=0.0,
        atol=1e-5,
    )
    return same_shape and same_affine


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


def resample_segmentation(
    segmentation_image: Any, flair_image: Any, output_path: Path
) -> tuple[Any, np.ndarray]:
    if len(flair_image.shape) != 3 or len(segmentation_image.shape) != 3:
        raise ValueError("Chỉ hỗ trợ resample cặp ảnh 3D.")

    from nibabel.processing import resample_from_to

    # Nearest-neighbor giữ nguyên nhãn rời rạc, không tạo giá trị mask trung gian.
    aligned_image = resample_from_to(
        segmentation_image,
        (flair_image.shape, flair_image.affine),
        order=0,
    )
    aligned_data = np.asanyarray(aligned_image.dataobj)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(aligned_image, str(output_path))
    return aligned_image, aligned_data


def save_wt_mask(
    segmentation_data: np.ndarray, flair_image: Any, output_path: Path
) -> None:
    # WT gộp ba vùng BraTS; nền 0 bị loại khỏi mask.
    wt_mask = np.isin(segmentation_data, [1, 2, 4]).astype(np.uint8)
    mask_header = flair_image.header.copy()
    mask_header.set_data_dtype(np.uint8)
    mask_image = nib.Nifti1Image(wt_mask, flair_image.affine, header=mask_header)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(mask_image, str(output_path))


def package_nifti_outputs(output_dir: Path) -> Path | None:
    folders = [output_dir / name for name in ("wt_masks", "aligned_segmentations")]
    nifti_files = [
        path for folder in folders for path in folder.rglob("*")
        if folder.is_dir() and path.is_file() and path.name.casefold().endswith((".nii", ".nii.gz"))
    ]
    archive_path = output_dir / "nifti_outputs.zip"
    if not nifti_files:
        if archive_path.exists():
            archive_path.unlink()
        return None
    # Xóa file trung gian chỉ sau khi ZIP đóng thành công, tránh mất đầu ra khi nén lỗi.
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        for file_path in sorted(nifti_files):
            archive.write(file_path, arcname=file_path.relative_to(output_dir))
    for folder in folders:
        if folder.exists(): shutil.rmtree(folder)
    return archive_path


def update_report_archive_paths(
    report_rows: list[dict[str, str]], archive_path: Path | None
) -> None:
    if archive_path is None:
        return
    for row in report_rows:
        for field in ("wt_mask_path", "resampled_segmentation_path"):
            if row[field]:
                row[field] = f"{archive_path.name}::{row[field]}"


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
    # Dừng trước khi mở ảnh vì cặp thiếu/trùng chưa xác định được đường dẫn duy nhất.
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

    # Tách lỗi số liệu khỏi lỗi nhãn để trạng thái báo cáo giữ đúng nguyên nhân.
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
            _, aligned_data = resample_segmentation(segmentation_image, flair_image, aligned_path)
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

        # Resample có thể làm phát sinh nhãn mới; kiểm tra lại trước khi dựng WT.
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


def get_report_columns() -> list[str]:
    base = """patient_id status reason manifest_status manifest_reason
    flair_path segmentation_path""".split()
    image_fields = [
        f"{modality}_{field}"
        for modality in ("flair", "segmentation")
        for field in FILE_REPORT_FIELDS
    ]
    results = """same_grid resampled_segmentation_path resampling_interpolation
    target_shape target_affine resampled_segmentation_labels_valid wt_mask_path errors""".split()
    return base + image_fields + results


def manifest_error_row(message: str) -> dict[str, str]:
    row = new_report_row("__manifest__")
    row["status"] = "manifest_error"
    row["reason"] = message
    row["errors"] = message
    return row


def write_summary(
    output_path: Path,
    report_rows: list[dict[str, str]],
    manifest_errors: list[str],
    nifti_archive: Path | None,
) -> None:
    total_patients = sum(1 for row in report_rows if row["patient_id"] != "__manifest__")
    status_counts = Counter(row["status"] for row in report_rows)
    counts = {
        # Một ca resampled vẫn là cặp dùng được; các nhóm lỗi bên dưới có thể chồng lấp.
        "valid_pairs": sum(row["status"] in {"valid", "resampled"} for row in report_rows),
        "missing_or_duplicate": sum(row["status"] in {"missing_file", "duplicate_file"} for row in report_rows),
        "read_errors": status_counts["read_error"],
        "csv_errors": status_counts["csv_error"],
        "invalid_values": 0,
        "nan_or_inf": 0,
        "invalid_labels": sum(
            row["segmentation_labels_valid"] == "no"
            or row["resampled_segmentation_labels_valid"] == "no"
            for row in report_rows
        ),
        "different_grids": sum(row["same_grid"] == "no" for row in report_rows),
        "requires_alignment": status_counts["requires_alignment"],
        "resampled": sum(bool(row["resampling_interpolation"]) for row in report_rows),
    }
    for row in report_rows:
        invalid = any(
            row[f"{modality}_has_nan"] == "yes" or row[f"{modality}_has_inf"] == "yes"
            for modality in ("flair", "segmentation")
        )
        counts["nan_or_inf"] += invalid
        counts["invalid_values"] += row["status"] == "invalid_values" or invalid

    with output_path.open("w", encoding="utf-8") as summary_file:
        summary_file.write(
            f"Tổng số patient ID: {total_patients}\n"
            f"Cặp hợp lệ: {counts['valid_pairs']}\n"
            f"Ca thiếu hoặc trùng file: {counts['missing_or_duplicate']}\n"
            f"Lỗi đọc NIfTI: {counts['read_errors']}\n"
            f"Lỗi manifest hoặc đầu vào: {counts['csv_errors'] + len(manifest_errors)}\n"
            f"Giá trị dữ liệu không hợp lệ: {counts['invalid_values']}\n"
            f"Ca có NaN hoặc Inf: {counts['nan_or_inf']}\n"
            f"Ca có nhãn segmentation không hợp lệ: {counts['invalid_labels']}\n"
            f"Cặp khác lưới: {counts['different_grids']}\n"
            f"Cần căn chỉnh: {counts['requires_alignment']}\n"
            f"Đã resample: {counts['resampled']}\n"
            f"Gói NIfTI: {nifti_archive.name if nifti_archive is not None else 'không có'}\n"
        )
        summary_file.write("\nSố ca theo trạng thái:\n")
        for status in sorted(status_counts):
            summary_file.write(f"- {status}: {status_counts[status]}\n")
        if manifest_errors:
            summary_file.write("\nLỗi manifest:\n")
            for error in manifest_errors:
                summary_file.write(f"- {error}\n")


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
