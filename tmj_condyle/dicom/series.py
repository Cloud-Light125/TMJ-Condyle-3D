"""DICOM series reading through SimpleITK/GDCM.

The reader obtains the ordered filenames from
ImageSeriesReader.GetGDCMSeriesFileNames. No filename sorting is performed by
this project.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..utils.io import require_simpleitk


_KNOWN_NON_DICOM_SUFFIXES = {
    ".csv",
    ".gz",
    ".json",
    ".jpg",
    ".jpeg",
    ".md",
    ".nii",
    ".png",
    ".txt",
    ".yaml",
    ".yml",
}


def _may_contain_dicom(directory: Path) -> bool:
    """Avoid probing folders containing only obvious non-DICOM artifacts."""

    try:
        files = [path for path in directory.iterdir() if path.is_file()]
    except OSError:
        return False
    return any(path.suffix.lower() not in _KNOWN_NON_DICOM_SUFFIXES for path in files)


@dataclass(frozen=True)
class SeriesInfo:
    index: int
    series_uid: str
    file_count: int
    directory: Path

    def safe_dict(self) -> dict[str, int]:
        """Return an output-safe summary without exposing UIDs."""

        return {"series_index": self.index, "file_count": self.file_count}


def discover_series(directory: str | Path) -> list[SeriesInfo]:
    sitk = require_simpleitk()
    root = Path(directory)
    if not root.is_dir():
        raise NotADirectoryError(root)
    candidates = [root]
    candidates.extend(sorted(path for path in root.rglob("*") if path.is_dir()))
    discovered: list[tuple[Path, str, int]] = []
    for candidate in candidates:
        if not _may_contain_dicom(candidate):
            continue
        series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(candidate)) or []
        for series_uid in sorted(series_ids):
            filenames = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
                str(candidate), series_uid
            )
            if filenames:
                discovered.append((candidate, series_uid, len(filenames)))
    result: list[SeriesInfo] = []
    for index, (candidate, series_uid, file_count) in enumerate(discovered, start=1):
        result.append(
            SeriesInfo(
                index=index,
                series_uid=series_uid,
                file_count=file_count,
                directory=candidate,
            )
        )
    return result


def read_series(
    directory: str | Path,
    *,
    series_uid: str | None = None,
    series_index: int | None = None,
) -> tuple[Any, SeriesInfo]:
    """Read one DICOM series in GDCM-provided physical slice order."""

    sitk = require_simpleitk()
    series = discover_series(directory)
    if not series:
        raise ValueError("No DICOM series found in the selected input directory tree")
    if series_uid is not None and series_index is not None:
        raise ValueError("Choose either series_uid or series_index, not both")
    if series_index is not None:
        selected = next((item for item in series if item.index == series_index), None)
        if selected is None:
            raise ValueError(f"--series-index must be one of 1..{len(series)}")
    elif series_uid is None:
        if len(series) != 1:
            safe = ", ".join(
                f"series_{item.index} ({item.file_count} files)" for item in series
            )
            raise ValueError(
                "Multiple DICOM series were found. Re-run with --series-index "
                f"using one of the redacted entries. Found: {safe}"
            )
        selected = series[0]
    else:
        selected = next((item for item in series if item.series_uid == series_uid), None)
        if selected is None:
            raise ValueError("--series-uid does not match a series in the input directory")

    filenames = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
        str(selected.directory), selected.series_uid
    )
    if not filenames:
        raise ValueError("The selected DICOM series has no readable files")
    reader = sitk.ImageSeriesReader()
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    reader.SetFileNames(filenames)
    image = reader.Execute()
    return image, selected
