# Kiểm kê tên file NIfTI BraTS2020 TrainingData

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
# Chỉ định đúng thư mục TrainingData; không tự mở rộng sang thư mục archive khác.
DEFAULT_DATA_ROOT = (
    REPOSITORY_ROOT
    / "data"
    / "archive"
    / "BraTS2020_TrainingData"
    / "MICCAI_BraTS2020_TrainingData"
)

PATIENT_ID_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])BraTS20_Training_(\d{3})(?!\d)", re.IGNORECASE
)
FLAIR_PATTERN = re.compile(r"(?:^|[_-])flair(?:$|[_-])", re.IGNORECASE)
SEGMENTATION_PATTERN = re.compile(r"(?:^|[_-])seg(?:$|[_-])", re.IGNORECASE)
OTHER_MODALITY_PATTERN = re.compile(
    r"(?:^|[_-])(?:t1ce|t1|t2)(?:$|[_-])", re.IGNORECASE
)

# Đây là danh sách ID dự kiến để đối chiếu, không phải danh sách ca đã tìm thấy.
EXPECTED_PATIENT_IDS = [
    f"BraTS20_Training_{number:03d}" for number in range(1, 370)
]

PATIENT_MANIFEST_COLUMNS = [
    "patient_id",
    "expected_id",
    "flair_count",
    "segmentation_count",
    "flair_paths",
    "segmentation_paths",
    "status",
    "reason",
]
NIFTI_FILES_COLUMNS = ["patient_id", "modality", "path"]
UNMATCHED_FILES_COLUMNS = ["patient_id", "reason", "path"]


# Đọc thư mục dữ liệu từ dòng lệnh, nếu không có thì dùng đường dẫn mặc định.
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kiểm kê tên file NIfTI trong BraTS2020 TrainingData."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Thư mục TrainingData (mặc định: {DEFAULT_DATA_ROOT})",
    )
    return parser.parse_args()


# Chỉ lấy đường dẫn file NIfTI bên trong thư mục dữ liệu được chọn.
def find_nifti_files(data_root: Path) -> list[Path]:
    nifti_files = []
    for path in data_root.rglob("*"):
        if not path.is_file():
            continue

        lower_name = path.name.casefold()
        if lower_name.endswith(".nii") or lower_name.endswith(".nii.gz"):
            nifti_files.append(path)

    return sorted(nifti_files, key=lambda path: path.as_posix().casefold())


# Chuẩn hóa ID trong một đoạn tên để so sánh không phụ thuộc chữ hoa/thường.
def extract_patient_ids(text: str) -> set[str]:
    patient_ids = set()
    for match in PATIENT_ID_PATTERN.finditer(text):
        patient_ids.add(f"BraTS20_Training_{match.group(1)}")
    return patient_ids


# So ID ở tên file với ID trong các thư mục cha để phát hiện xung đột.
def get_patient_ids(file_path: Path, relative_path: Path) -> set[str]:
    patient_ids = extract_patient_ids(file_path.name)
    directory_ids = extract_patient_ids(relative_path.parent.as_posix())
    return patient_ids | directory_ids


# Nhận diện FLAIR/segmentation, bỏ qua modality khác và giữ lại tên chưa rõ.
def classify_modality(file_name: str) -> str:
    lower_name = file_name.casefold()
    if lower_name.endswith(".nii.gz"):
        file_stem = lower_name[:-7]
    elif lower_name.endswith(".nii"):
        file_stem = lower_name[:-4]
    else:
        file_stem = lower_name

    has_flair = FLAIR_PATTERN.search(file_stem) is not None
    has_segmentation = SEGMENTATION_PATTERN.search(file_stem) is not None

    if has_flair and has_segmentation:
        return "ambiguous"
    if has_flair:
        return "flair"
    if has_segmentation:
        return "segmentation"
    if OTHER_MODALITY_PATTERN.search(file_stem):
        return "ignored"
    return "unknown"


# Ghi file đã phân loại vào bệnh nhân hoặc thêm vào danh sách unmatched.
def process_nifti_files(
    data_root: Path,
    nifti_files: list[Path],
    files_by_patient: dict[str, dict[str, list[str]]],
    classified_files: list[dict[str, str]],
    unmatched_files: list[dict[str, str]],
    seen_patient_ids: set[str],
) -> None:
    for file_path in nifti_files:
        relative_path = file_path.relative_to(data_root)
        relative_name = relative_path.as_posix()
        patient_ids = get_patient_ids(file_path, relative_path)
        modality = classify_modality(file_path.name)
        seen_patient_ids.update(patient_ids)

        # ID mâu thuẫn giữa tên file và thư mục khiến file không thuộc bệnh nhân nào.
        if len(patient_ids) > 1:
            reason = "conflicting_patient_ids"
            if modality == "unknown":
                reason += ";modality_unclassifiable"
            elif modality == "ambiguous":
                reason += ";ambiguous_modality"
            unmatched_files.append(
                {
                    "patient_id": ";".join(sorted(patient_ids)),
                    "reason": reason,
                    "path": relative_name,
                }
            )
            continue

        if modality == "ignored":
            continue

        if not patient_ids:
            reason = "patient_id_not_found"
            if modality == "unknown":
                reason += ";modality_unclassifiable"
            elif modality == "ambiguous":
                reason += ";ambiguous_modality"
            unmatched_files.append(
                {"patient_id": "", "reason": reason, "path": relative_name}
            )
            continue

        patient_id = next(iter(patient_ids))
        if modality == "unknown" or modality == "ambiguous":
            reason = (
                "modality_unclassifiable"
                if modality == "unknown"
                else "ambiguous_modality"
            )
            unmatched_files.append(
                {"patient_id": patient_id, "reason": reason, "path": relative_name}
            )
            continue

        if patient_id not in files_by_patient:
            files_by_patient[patient_id] = {
                "flair": [],
                "segmentation": [],
            }
        files_by_patient[patient_id][modality].append(relative_name)
        classified_files.append(
            {"patient_id": patient_id, "modality": modality, "path": relative_name}
        )


# Lập manifest cho ID dự kiến lẫn ID mới phát hiện, không tự điền đường dẫn.
def create_manifest(
    files_by_patient: dict[str, dict[str, list[str]]],
    seen_patient_ids: set[str],
) -> list[dict[str, str | int]]:
    all_patient_ids = set(EXPECTED_PATIENT_IDS) | seen_patient_ids
    manifest = []

    for patient_id in sorted(all_patient_ids):
        patient_files = files_by_patient.get(
            patient_id, {"flair": [], "segmentation": []}
        )
        flair_paths = sorted(patient_files["flair"])
        segmentation_paths = sorted(patient_files["segmentation"])
        reasons = []

        if not flair_paths:
            reasons.append("missing_flair")
        elif len(flair_paths) > 1:
            reasons.append("duplicate_flair")

        if not segmentation_paths:
            reasons.append("missing_segmentation")
        elif len(segmentation_paths) > 1:
            reasons.append("duplicate_segmentation")

        has_missing_file = any(reason.startswith("missing_") for reason in reasons)
        has_duplicate_file = any(reason.startswith("duplicate_") for reason in reasons)
        if has_missing_file and has_duplicate_file:
            status = "missing_and_duplicate"
        elif has_duplicate_file:
            status = "duplicate"
        elif has_missing_file:
            status = "missing"
        else:
            status = "valid"

        # Chỉ cặp có đúng một file mỗi loại mới được đánh dấu hợp lệ.
        manifest.append(
            {
                "patient_id": patient_id,
                "expected_id": "yes" if patient_id in EXPECTED_PATIENT_IDS else "no",
                "flair_count": len(flair_paths),
                "segmentation_count": len(segmentation_paths),
                "flair_paths": ";".join(flair_paths),
                "segmentation_paths": ";".join(segmentation_paths),
                "status": status,
                "reason": ";".join(reasons),
            }
        )

    return manifest


# Ghi một báo cáo CSV với tên cột và thứ tự cột đã quy định.
def write_csv(
    output_path: Path, column_names: list[str], rows: list[dict[str, object]]
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=column_names)
        writer.writeheader()
        writer.writerows(rows)


# In số liệu tổng hợp và danh sách ID có vấn đề từ manifest.
def print_summary(
    manifest: list[dict[str, str | int]], seen_patient_ids: set[str]
) -> None:
    flair_patient_ids = []
    segmentation_patient_ids = []
    valid_patient_ids = []
    missing_flair_ids = []
    missing_segmentation_ids = []
    duplicate_patient_ids = []

    for row in manifest:
        patient_id = str(row["patient_id"])
        flair_count = int(row["flair_count"])
        segmentation_count = int(row["segmentation_count"])

        if flair_count > 0:
            flair_patient_ids.append(patient_id)
        else:
            missing_flair_ids.append(patient_id)

        if segmentation_count > 0:
            segmentation_patient_ids.append(patient_id)
        else:
            missing_segmentation_ids.append(patient_id)

        if row["status"] == "valid":
            valid_patient_ids.append(patient_id)
        if flair_count > 1 or segmentation_count > 1:
            duplicate_patient_ids.append(patient_id)

    missing_expected_ids = []
    for patient_id in EXPECTED_PATIENT_IDS:
        if patient_id not in seen_patient_ids:
            missing_expected_ids.append(patient_id)

    print(f"Expected TrainingData IDs: {len(EXPECTED_PATIENT_IDS)}")
    print(f"IDs with FLAIR: {len(flair_patient_ids)}")
    print(f"IDs with segmentation: {len(segmentation_patient_ids)}")
    print(f"Valid pairs (exactly one file per modality): {len(valid_patient_ids)}")
    print(f"Missing FLAIR: {', '.join(missing_flair_ids) or 'none'}")
    print(f"Missing segmentation: {', '.join(missing_segmentation_ids) or 'none'}")
    print(f"Duplicate files: {', '.join(duplicate_patient_ids) or 'none'}")
    print(f"Expected IDs not found in directory: {', '.join(missing_expected_ids) or 'none'}")


# Thực hiện lần lượt đọc cấu hình, quét, kiểm kê, ghi báo cáo và in kết quả.
def main() -> int:
    arguments = parse_arguments()
    data_root = arguments.data_root.expanduser().resolve()

    if not data_root.is_dir():
        print(
            f"Data directory not found: {data_root}\n"
            "Pass the correct TrainingData directory with --data-root PATH.",
            file=sys.stderr,
        )
        return 1

    nifti_files = find_nifti_files(data_root)
    if not nifti_files:
        print(
            f"No NIfTI files (.nii or .nii.gz) found under: {data_root}\n"
            "Pass the correct TrainingData directory with --data-root PATH.",
            file=sys.stderr,
        )
        return 1

    files_by_patient: dict[str, dict[str, list[str]]] = {}
    classified_files = []
    unmatched_files = []
    seen_patient_ids = set()
    process_nifti_files(
        data_root,
        nifti_files,
        files_by_patient,
        classified_files,
        unmatched_files,
        seen_patient_ids,
    )
    manifest = create_manifest(files_by_patient, seen_patient_ids)

    output_directory = REPOSITORY_ROOT / "results" / "module1"
    output_directory.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_directory / "patient_manifest.csv",
        PATIENT_MANIFEST_COLUMNS,
        manifest,
    )
    write_csv(
        output_directory / "nifti_files_found.csv",
        NIFTI_FILES_COLUMNS,
        classified_files,
    )
    write_csv(
        output_directory / "unmatched_files.csv",
        UNMATCHED_FILES_COLUMNS,
        unmatched_files,
    )

    print_summary(manifest, seen_patient_ids)
    print(f"Reports written to: {output_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
