"""Read-only, manifest-driven I/O for the new AICSET 240 analysis.

One row is ONE simultaneous multichannel acquisition. A source file can occur in
several rows only with distinct record_id and explicit selectors. No file name is
treated as proof of health, damage, sampling rate, or a training/evaluation role.
The verified OpenLab convention plus exact CSV header, and the official Z24 PDT
scenario convention plus inspected MATLAB physical labels, are narrow exceptions.
CSV columns and MAT/NPZ/HDF5 variables must be selected explicitly when ambiguous.
Optional channel_names declares physical sensor labels in selection order; it is
mandatory for Z24 PDT matrix/position selectors because array indexes alone do not
identify a sensor location. Vector channel_keys and exact CSV names are explicit.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import re
import warnings
import zipfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


MANIFEST_COLUMNS = [
    "dataset", "record_id", "acquisition_id", "group_id", "timestamp_utc",
    "state_label", "scenario", "excitation", "setup_id", "path", "format",
    "fs_hz", "data_key", "channel_keys", "channel_columns", "channel_names", "sample_axis",
    "delimiter", "decimal", "skiprows", "header", "time_column",
]
SUPPORTED_INPUT_SUFFIXES = {".csv", ".txt", ".asc", ".dat", ".mat", ".npz", ".h5", ".hdf5"}
ALLOWED_DATASETS = {"openlab_2024", "z24_ems", "z24_pdt"}
OPENLAB_2024_VERTICAL_CHANNELS = [
    "G_ACCZ_PE11_CB0750_0", "G_ACCZ_PE12_CB0750_0", "G_ACCZ_PE13_CB0750_0",
    "G_ACCZ_PE22_CB0750_0", "G_ACCZ_PE23_CB0750_0",
]
Z24_PDT_REFERENCE_CHANNELS = ["R1V", "R2L", "R2T", "R2V", "R3V"]
Z24_PDT_PROTOCOL_SOURCE = "https://bwk.kuleuven.be/bwm/z24"
Z24_PDT_LABEL_SCHEMA_SOURCE = "https://github.com/AbdellahRiyahi/MultiBridge_FaultAware_Selective_SHM/blob/main/MultiBridge_FaultAware_Selective_SHM_V1_2.ipynb"


class ManifestValidationError(ValueError):
    """An invalid manifest; ``errors`` is an exportable pandas DataFrame."""

    def __init__(self, errors: pd.DataFrame):
        self.errors = errors
        examples = "; ".join(str(v) for v in errors["message"].head(6))
        super().__init__(f"Manifest validation failed ({len(errors)} issues): {examples}")


def _rio_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    try:
        result = pd.isna(value)
        return bool(result) if np.ndim(result) == 0 else False
    except (TypeError, ValueError):
        return False


def _rio_text(value: Any) -> str:
    return "" if _rio_empty(value) else str(value).strip()


def _rio_list(value: Any, field: str) -> list:
    if _rio_empty(value):
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        parsed = list(value)
    else:
        try:
            parsed = json.loads(str(value))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{field} must be a JSON list, e.g. [0, 1] or [\"R1V\", \"R2V\"]") from exc
    if not isinstance(parsed, list):
        raise ValueError(f"{field} must be a JSON list")
    if field in {"channel_keys", "channel_names"}:
        if not all(isinstance(v, str) and v.strip() for v in parsed):
            raise ValueError(f"{field} must contain nonempty strings")
        if any(v != v.strip() for v in parsed):
            raise ValueError(f"{field} entries must not contain surrounding whitespace")
    elif field == "channel_columns" and parsed:
        if all(isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_)) for v in parsed):
            parsed = [int(v) for v in parsed]
            if min(parsed) < 0:
                raise ValueError("channel_columns indexes cannot be negative")
        elif not all(isinstance(v, str) and v.strip() for v in parsed):
            raise ValueError("channel_columns must contain all nonnegative integer indexes or all exact nonempty string names")
    if len({str(v) for v in parsed}) != len(parsed):
        raise ValueError(f"{field} contains duplicate selectors")
    return parsed


def _rio_int(value: Any, field: str, default=None):
    if _rio_empty(value):
        if default is not None:
            return default
        raise ValueError(f"{field} must be stated explicitly")
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field} must be an integer, not a Boolean")
    number = float(value)
    if not math.isfinite(number) or number != int(number):
        raise ValueError(f"{field} must be an integer")
    return int(number)


def _rio_sampling_rate(value: Any) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("fs_hz must be a sampling rate, not a Boolean")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError("fs_hz must be a finite positive measured/documented sampling rate")
    return number


def _rio_timestamp(value: Any) -> pd.Timestamp:
    """Require an explicitly zoned ISO timestamp; never assume a local time is UTC."""
    if isinstance(value, (pd.Timestamp,)) or hasattr(value, "tzinfo"):
        stamp = pd.Timestamp(value)
    else:
        text = _rio_text(value)
        if not isinstance(value, str) or not re.fullmatch(
                r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})", text):
            raise ValueError("timestamp_utc requires an ISO date/time with explicit Z or UTC offset; numeric epochs and timezone-naive dates are ambiguous")
        stamp = pd.Timestamp(text)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError("timestamp_utc must include an explicit timezone; confirm the source time convention before adding Z")
    return stamp.tz_convert("UTC")


def _rio_header(value: Any):
    text = _rio_text(value).lower()
    if text in {"none", "null", "-1"}:
        return None
    if text == "":
        raise ValueError("header must be an explicit nonnegative row number or 'none'")
    number = _rio_int(value, "header")
    if number < 0:
        raise ValueError("header cannot be negative (use 'none' for no header)")
    return number


def _rio_root_for(dataset: str, roots: Mapping[str, Path]) -> Path:
    key = dataset if dataset in roots else ("z24" if dataset.startswith("z24_") else "openlab")
    if key not in roots:
        raise ValueError(f"No explicit input root configured for {dataset!r}")
    root = Path(roots[key]).expanduser()
    if not root.is_absolute():
        raise ValueError(f"Input root must be absolute: {root}")
    root = root.resolve(strict=True)
    if root == Path("/home") or Path("/home") in root.parents:
        raise ValueError(f"/home input roots are disabled; configure the actual /data location: {root}")
    if not root.is_dir():
        raise ValueError(f"Input root is not a directory: {root}")
    return root


def _rio_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _rio_safe_output(out_dir: Path, roots: Mapping[str, Path]) -> Path:
    output = Path(out_dir).expanduser().resolve()
    for value in roots.values():
        root = Path(value).expanduser().resolve()
        if _rio_within(output, root):
            raise ValueError(f"Write output outside the read-only input tree: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _rio_csv_sample(path: Path, skiprows: int = 0):
    with Path(path).open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
        for _ in range(skiprows):
            next(stream, None)
        return stream.read(16384)


def _rio_delimiter(value: Any, path: Path, skiprows: int = 0) -> str:
    text = _rio_text(value)
    if text.lower() in {"tab", "\\t"}:
        return "\t"
    if text.lower() in {"whitespace", "space", r"\s+"}:
        return r"\s+"
    if text and text.lower() != "auto":
        if len(text) != 1:
            raise ValueError("delimiter must be one character, tab, whitespace, or auto")
        return text
    sample = _rio_csv_sample(path, skiprows)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        rows = [line.strip() for line in sample.splitlines() if line.strip()][:12]
        widths = [len(re.split(r"\s+", line)) for line in rows]
        if widths and min(widths) >= 2 and len(set(widths)) == 1:
            return r"\s+"
        raise ValueError(f"Cannot infer a reliable delimiter for {path.name}; state it in the manifest")


def _rio_decimal(value: Any) -> str:
    """Honor an explicit CSV decimal mark; never infer a numeric locale."""
    mark = _rio_text(value) or "."
    if mark not in {".", ","}:
        raise ValueError("decimal must be an explicit '.' or ',' (default '.')")
    return mark


def _rio_csv_header(path: Path, delimiter: str, skiprows: int, header) -> list | None:
    """Read the physical header before pandas can rename duplicate column names."""
    if header is None:
        return None
    with path.open("r", encoding="utf-8-sig", errors="strict", newline="") as stream:
        for _ in range(skiprows):
            next(stream, None)
        records = (re.split(r"\s+", line.strip()) for line in stream if line.strip()) if delimiter == r"\s+" else csv.reader(stream, delimiter=delimiter)
        nonblank_index = -1
        for cells in records:
            if not cells or (len(cells) == 1 and not cells[0].strip()):
                continue
            nonblank_index += 1
            if nonblank_index == header:
                return cells
    raise ValueError("CSV header row is beyond the available nonblank records")


def _rio_file_identity(path: Path) -> tuple:
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def probe_input(path: Path) -> dict:
    """Inspect headers/container metadata, never infer scientific labels or fs.

    CSV probing reads at most 16 KiB. MAT uses whosmat; v7.3 needs optional h5py.
    NPZ never permits pickle. Probes are diagnostic suggestions, not selectors.
    """
    path = Path(path).resolve(strict=True)
    info: dict[str, Any] = {"path": str(path), "suffix": path.suffix.lower(), "size_bytes": path.stat().st_size}
    suffix = path.suffix.lower()
    try:
        if suffix in {".csv", ".txt", ".asc", ".dat"}:
            sample = _rio_csv_sample(path)
            nonempty = [line for line in sample.splitlines() if line.strip()]
            info["first_nonempty_lines"] = nonempty[:6]
            try:
                delimiter = _rio_delimiter("auto", path)
                info["candidate_delimiter"] = "whitespace" if delimiter == r"\s+" else delimiter
                if delimiter != r"\s+" and nonempty:
                    cells = next(csv.reader([nonempty[0]], delimiter=delimiter))
                else:
                    cells = re.split(r"\s+", nonempty[0].strip()) if nonempty else []
                info["first_row_cells"] = cells
                info["first_row_has_nonnumeric_tokens"] = any(not _rio_is_float(v) for v in cells)
            except ValueError as exc:
                info["delimiter_error"] = str(exc)
            # Headers can contain engineering units; report, but do not interpret them as fs.
            info["sampling_rate_inferred"] = False
            info["decimal_mark_inferred"] = False
        elif suffix == ".mat":
            from scipy.io import whosmat
            try:
                info["variables"] = [{"key": k, "shape": list(shape), "matlab_class": typ} for k, shape, typ in whosmat(path)]
                if {v["key"] for v in info["variables"]}.issuperset({"data", "labelshulp"}):
                    # Selective label loading enriches the schema without
                    # materialising the vibration matrix or estimating features.
                    info["z24_pdt_label_schema"] = _rio_pdt_mat_schema(path)
            except (NotImplementedError, ValueError, OSError) as exc:
                if not importlib.util.find_spec("h5py"):
                    raise ValueError("MAT metadata unavailable (possibly MATLAB v7.3); install h5py or use an explicit compatible export") from exc
                info["hdf5_datasets"] = _rio_h5_metadata(path)
        elif suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                info["variables"] = []
                for key in archive.files:
                    try:
                        array = archive[key]
                        info["variables"].append({"key": key, "shape": list(array.shape), "dtype": str(array.dtype)})
                    except ValueError as exc:
                        info["variables"].append({"key": key, "error": str(exc), "usable": False})
        elif suffix in {".h5", ".hdf5"}:
            if not importlib.util.find_spec("h5py"):
                raise ValueError("HDF5 input requires optional h5py")
            info["hdf5_datasets"] = _rio_h5_metadata(path)
        elif suffix == ".zip":
            with zipfile.ZipFile(path, "r") as archive:
                members = [{"path": item.filename, "size_bytes": item.file_size,
                            "compressed_bytes": item.compress_size, "directory": item.is_dir()}
                           for item in archive.infolist()]
            info["archive_members"] = members
            info["archive_extracted"] = False
            info["scientific_role_inferred"] = False
        elif suffix == ".aaa":
            info["first_nonempty_lines"] = [line for line in _rio_csv_sample(path).splitlines() if line.strip()][:12]
            info["supported_signal_container"] = False
            info["note"] = "Individual-channel source: diagnostic text only; not automatically merged or substituted for multichannel MAT acquisitions."
        else:
            info["supported_signal_container"] = False
    except Exception as exc:
        info["probe_error"] = f"{type(exc).__name__}: {exc}"
    return info


def _rio_is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def _rio_h5_metadata(path: Path) -> list[dict]:
    import h5py
    result = []
    with h5py.File(path, "r") as handle:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                result.append({"key": name, "shape": list(obj.shape), "dtype": str(obj.dtype)})
        handle.visititems(visit)
    return result


def _rio_mat_labels(value: Any) -> list[str]:
    """Decode explicit MATLAB string labels; never convert numeric indices to names."""
    array = np.asarray(value)
    if array.dtype.kind == "O":
        if array.ndim > 2 or (array.ndim == 2 and min(array.shape) > 1):
            raise ValueError(f"Ambiguous labelshulp cell layout {array.shape}; an ordered label vector is required")
        result = []
        for part in array.ravel():
            result.extend(_rio_mat_labels(part))
        return result
    if array.dtype.kind not in "US":
        raise ValueError("labelshulp must contain strings/characters, not numeric channel indices")
    char_width = array.dtype.itemsize // (4 if array.dtype.kind == "U" else 1)
    if array.ndim == 2 and char_width == 1:
        strings = ["".join(v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in row) for row in array]
    elif array.ndim <= 1 or (array.ndim == 2 and 1 in array.shape):
        strings = [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in array.ravel()]
    else:
        raise ValueError(f"Ambiguous labelshulp character layout {array.shape}; inspect the exported labels")
    return [value.strip(" \t\r\n\x00") for value in strings]


def _rio_pdt_mat_schema(path: Path) -> dict:
    """Inspect data dimensions and load ONLY channel/rate metadata from a classic MAT file."""
    from scipy.io import loadmat, whosmat
    path = Path(path)
    identity = _rio_file_identity(path)
    variables = {key: (shape, kind) for key, shape, kind in whosmat(path)}
    if not {"data", "labelshulp"}.issubset(variables):
        raise ValueError("Verified PDT schema requires both data and labelshulp")
    shape, kind = variables["data"]
    if len(shape) != 2 or shape[0] < 8 or shape[1] < 5 or kind not in {"double", "single", "int16", "int32", "int64", "uint16", "uint32", "uint64", "int8", "uint8"}:
        raise ValueError(f"PDT data must be a real samples-by-channels numeric matrix; metadata says {shape}, {kind}")
    values = loadmat(path, variable_names=["labelshulp", "fs", "Fs", "sample_rate", "sampling_rate"], simplify_cells=True)
    labels = _rio_mat_labels(values["labelshulp"])
    if len(labels) != shape[1]:
        raise ValueError(f"labelshulp has {len(labels)} labels but data has {shape[1]} columns; no orientation or positions are guessed")
    rates = {}
    for key in ["fs", "Fs", "sample_rate", "sampling_rate"]:
        if key not in values:
            continue
        value = np.asarray(values[key]).squeeze()
        if value.ndim != 0:
            raise ValueError(f"PDT sampling-rate metadata {key!r} is not a scalar")
        rate = _rio_sampling_rate(value.item())
        if not math.isclose(rate, 100.0, rel_tol=0, abs_tol=1e-9):
            raise ValueError(f"PDT file rate {key}={rate} contradicts the documented 100 Hz protocol")
        rates[key] = rate
    if _rio_file_identity(path) != identity:
        raise ValueError("PDT file changed during label/rate metadata inspection")
    return {"data_key": "data", "data_shape": list(shape), "data_class": kind,
            "label_key": "labelshulp", "channel_labels": labels, "sample_axis": 0,
            "embedded_rates_hz": rates, "signal_matrix_loaded": False}


def _rio_pdt_path_metadata(relative: str) -> dict:
    """Recognise the actual SS/AVT-or-FVT/SSsetupNN.mat PDT source convention."""
    match = re.fullmatch(r"(0[1-9]|1[0-7])/(avt|fvt)/\1setup(0[1-9])\.mat", relative.replace("\\", "/"), flags=re.I)
    if not match:
        return {}
    scenario, excitation, setup = match.groups()
    return {"scenario": scenario, "excitation": excitation.upper(), "setup_id": setup}


def _rio_pdt_manifest_metadata(inventory: pd.DataFrame) -> tuple[dict, dict]:
    """Verify a common physical reference layout before creating any PDT auto row.

    The nine source-01 AVT setups determine the common R-channel intersection.
    It must agree with the five explicitly documented reference component names.
    Every other acquisition then independently proves its selected column mapping
    from labelshulp. No signal matrix, synthetic label or fixed index is used.
    """
    audit = {"status": "NO_MATCHING_PDT_MAT_PATHS", "protocol_source": Z24_PDT_PROTOCOL_SOURCE,
             "label_schema_source": Z24_PDT_LABEL_SCHEMA_SOURCE,
             "documented_fs_hz": 100.0, "reference_components": Z24_PDT_REFERENCE_CHANNELS,
             "reference_components_are_not_all_vertical": True,
             "signal_matrices_loaded": False, "records": [], "errors": []}
    candidates = []
    for item in inventory.to_dict("records"):
        if str(item.get("root_label", "")).lower() not in {"z24", "z24_pdt"}:
            continue
        known = _rio_pdt_path_metadata(str(item.get("relative_path", "")))
        if not known:
            continue
        path = str(item["path"])
        record = {"path": path, **known, "status": "UNVERIFIED"}
        try:
            root = _rio_root_for("z24_pdt", {"z24": Path(str(item.get("input_root", "")))})
            if not _rio_within(Path(path).resolve(strict=True), root):
                raise ValueError("PDT metadata path resolves outside its explicit inventory input root")
            record.update(_rio_pdt_mat_schema(Path(path)))
            record["status"] = "LABEL_SCHEMA_INSPECTED"
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            audit["errors"].append({"path": path, "message": record["error"]})
        candidates.append(record)
    audit["records"] = candidates
    if not candidates:
        return {}, audit
    source = [r for r in candidates if r["scenario"] == "01" and r["excitation"] == "AVT"]
    setups = [r["setup_id"] for r in source]
    required_setups = {f"{n:02d}" for n in range(1, 10)}
    if set(setups) != required_setups or len(setups) != 9 or any(r["status"] != "LABEL_SCHEMA_INSPECTED" for r in source):
        audit["status"] = "BLOCKED_SOURCE01_NINE_AVT_LABEL_SCHEMAS_REQUIRED"
        audit["errors"].append({"path": "source01/AVT", "message": "All nine distinct source-01 AVT setup label schemas are required before verifying a common reference-channel intersection"})
        return {}, audit
    common = set.intersection(*(set(r["channel_labels"]) for r in source))
    reference_common = sorted(label for label in common if re.fullmatch(r"R\d+[VLT]", label))
    audit["source01_avt_common_labels"] = sorted(common)
    audit["source01_avt_common_reference_labels"] = reference_common
    if set(reference_common) != set(Z24_PDT_REFERENCE_CHANNELS):
        audit["status"] = "BLOCKED_REFERENCE_CHANNEL_INTERSECTION_AMBIGUOUS"
        audit["errors"].append({"path": "source01/AVT", "message": "The source-01 AVT physical R-channel intersection does not equal the five documented reference components; inspect labelshulp diagnostics rather than choosing five positions"})
        return {}, audit
    verified = {}
    for record in candidates:
        if record["status"] != "LABEL_SCHEMA_INSPECTED":
            continue
        labels = record["channel_labels"]
        if any(labels.count(name) != 1 for name in Z24_PDT_REFERENCE_CHANNELS):
            record["status"] = "BLOCKED_REFERENCE_LABEL_MISSING_OR_DUPLICATED"
            audit["errors"].append({"path": record["path"], "message": "Every common physical reference component must occur exactly once in labelshulp"})
            continue
        columns = [labels.index(name) for name in Z24_PDT_REFERENCE_CHANNELS]
        record["selected_column_indices"] = columns
        record["status"] = "VERIFIED_PDT_PATH_LABEL_SCHEMA_AND_COMMON_REFERENCE_COMPONENTS"
        scenario = record["scenario"]
        verified[record["path"]] = {
            "dataset": "z24_pdt", "group_id": "scenario_" + scenario,
            "scenario": scenario, "setup_id": record["setup_id"], "excitation": record["excitation"],
            "state_label": "reference" if scenario in {"01", "02", "08"} else "altered",
            "fs_hz": 100.0, "data_key": "data", "sample_axis": 0,
            "channel_columns": json.dumps(columns), "channel_names": json.dumps(Z24_PDT_REFERENCE_CHANNELS),
            "metadata_status": record["status"]}
    audit["verified_acquisitions"] = len(verified)
    audit["candidate_acquisitions"] = len(candidates)
    if len(verified) != len(candidates):
        audit["status"] = "BLOCKED_PDT_COHORT_HAS_UNVERIFIED_SCHEMAS"
        audit["automatic_manifest_rows"] = 0
        return {}, audit
    audit["status"] = "VERIFIED_ALL_PDT_CANDIDATES"
    audit["automatic_manifest_rows"] = len(verified)
    return verified, audit


def inventory_inputs(roots: dict[str, Path], out_dir: Path) -> pd.DataFrame:
    """Inventory every file without changing inputs; probe a bounded representative set.

    No acquisition count is capped: representative metadata probes are bounded
    (three per directory/format, 60 total), distributed across roots and formats.
    All recognised OpenLab environmental CSVs additionally receive header probes.
    """
    output = _rio_safe_output(out_dir, roots)
    rows, excluded_links = [], []
    seen = set()
    for label, candidate in roots.items():
        supplied_root = Path(candidate).expanduser()
        if not supplied_root.is_absolute():
            raise ValueError(f"Input root must be absolute: {supplied_root}")
        root = supplied_root.resolve(strict=True)
        if not root.is_dir():
            raise ValueError(f"Input root is not a directory: {root}")
        if root == Path("/home") or Path("/home") in root.parents:
            raise ValueError(f"Use /data inputs, not /home: {root}")
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            actual = path.resolve(strict=True)
            if not _rio_within(actual, root):
                excluded_links.append({"root_label": label, "link": str(path), "resolved_target": str(actual),
                                       "reason": "TARGET_OUTSIDE_DECLARED_INPUT_ROOT"})
                continue  # never read/probe a link outside the declared raw root
            if actual in seen:
                continue
            seen.add(actual)
            stat = actual.stat()
            rows.append({"root_label": label, "input_root": str(root), "path": str(actual),
                         "relative_path": str(actual.relative_to(root)), "suffix": actual.suffix.lower(),
                         "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                         "supported_signal_container": actual.suffix.lower() in SUPPORTED_INPUT_SUFFIXES})
    frame = pd.DataFrame(rows, columns=["root_label", "input_root", "path", "relative_path", "suffix", "size_bytes", "mtime_ns", "supported_signal_container"])
    frame.to_csv(output / "dataset_inventory.csv", index=False)
    probes, counts, buckets = [], {}, {}
    for row in frame.to_dict("records"):
        if not row["supported_signal_container"] and row["suffix"] not in {".aaa", ".zip"}:
            continue
        buckets.setdefault((row["root_label"], row["suffix"]), []).append(row)
    # Round-robin roots/formats prevents an early MAT tree from consuming all
    # diagnostics. Environmental CSVs precede acceleration records in their bucket.
    for values in buckets.values():
        values.sort(key=lambda row: (0 if "/02_environment/" in "/" + row["relative_path"].replace("\\", "/") else 1, row["relative_path"]))
    while any(buckets.values()) and len(probes) < 60:
        for bucket in sorted(buckets):
            values = buckets[bucket]
            while values:
                row = values.pop(0)
                key = (row["root_label"], str(Path(row["relative_path"]).parent), row["suffix"])
                if counts.get(key, 0) >= 3:
                    continue
                counts[key] = counts.get(key, 0) + 1
                probes.append(probe_input(Path(row["path"])))
                break
            if len(probes) >= 60:
                break
    environment_probes = [probe_input(Path(row["path"])) for row in frame.to_dict("records")
                          if row["suffix"] == ".csv" and "/02_environment/" in "/" + row["relative_path"].replace("\\", "/")]
    (output / "file_format_probes.json").write_text(json.dumps(probes, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "excluded_input_links.json").write_text(json.dumps(excluded_links, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "environmental_csv_probes.json").write_text(json.dumps(environment_probes, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    frame.attrs["probe_file"] = str(output / "file_format_probes.json")
    frame.attrs["probe_count"] = len(probes)
    frame.attrs["excluded_link_count"] = len(excluded_links)
    frame.attrs["environmental_csv_probe_count"] = len(environment_probes)
    return frame


def make_manifest_template(inventory: pd.DataFrame) -> pd.DataFrame:
    """Create candidate rows, leaving unverified scientific fields empty.

    ``candidate_*`` columns assist manual/schema-specific completion and are never
    consumed as confirmed labels, groups, fs, channels, or timestamps by validation.
    Environmental CSVs, tiltmeters, velocity and 2025 damage recordings are not
    silently added to the 2024 acceleration cohort. PDT auto rows require the
    exact source convention and independently inspected physical label mappings.
    ``frame.attrs['z24_pdt_schema_audit']`` must be exported before saving to CSV.
    """
    rows = []
    pdt_metadata, pdt_audit = _rio_pdt_manifest_metadata(inventory)
    for item in inventory.to_dict("records"):
        if not bool(item.get("supported_signal_container", False)):
            continue
        path = Path(item["path"])
        relative = str(item["relative_path"])
        low = relative.lower()
        root_label = str(item.get("root_label", "")).lower()
        candidate_dataset = ""
        if root_label in ALLOWED_DATASETS:
            candidate_dataset = root_label
        elif root_label == "z24":
            candidate_dataset = "z24_pdt" if (path.suffix.lower() == ".mat" or re.search(r"pdt|progressive", low)) else ""
        elif root_label == "openlab" and re.search(r"reference|2024", low) and not re.search(r"damage|2025|tilt|environment|climate", low):
            candidate_dataset = "openlab_2024"
        match = re.search(r"(?<!\d)(20\d{2})[-_](\d{2})[-_](\d{2})(?!\d)", relative)
        candidate_date = "-".join(match.groups()) if match else ""
        scenarios = re.findall(r"(?:scenario|state|damage)[ _-]*(0[1-9]|1[0-7])(?=\D|$)", low)
        scenario = scenarios[0] if len(set(scenarios)) == 1 else ""
        excitation = "AVT" if re.search(r"(?:^|[^a-z])avt(?:[^a-z]|$)", low) else ("FVT" if re.search(r"(?:^|[^a-z])fvt(?:[^a-z]|$)", low) else "")
        uid = hashlib.sha256((str(item["root_label"]) + ":" + relative).encode()).hexdigest()[:16]
        row = {c: "" for c in MANIFEST_COLUMNS}
        row.update({"record_id": uid, "acquisition_id": uid, "state_label": "unknown", "path": str(path),
                    "format": ("csv" if path.suffix.lower() in {".csv", ".txt", ".asc", ".dat"} else path.suffix.lstrip(".")),
                    "skiprows": 0, "decimal": ".", "candidate_dataset": candidate_dataset, "candidate_utc_date": candidate_date,
                    "candidate_scenario": scenario, "candidate_excitation": excitation,
                    "metadata_status": "UNVERIFIED_COMPLETE_REQUIRED_FIELDS_BEFORE_ANALYSIS"})
        # This single cohort has verified protocol metadata: exact channel names,
        # fs=500 Hz, UTC timestamp naming, and reference-state classification. It is
        # auto-populated only if BOTH the path convention and actual CSV header
        # agree. Other cohorts require explicit metadata rather than assumptions.
        known = _rio_known_openlab_2024(path, relative)
        if known:
            row.update(known)
        if str(path) in pdt_metadata:
            row.update(pdt_metadata[str(path)])
        rows.append(row)
    frame = pd.DataFrame(rows, columns=MANIFEST_COLUMNS + ["candidate_dataset", "candidate_utc_date", "candidate_scenario", "candidate_excitation", "metadata_status"])
    frame.attrs["z24_pdt_schema_audit"] = pdt_audit
    return frame


def _rio_known_openlab_2024(path: Path, relative: str) -> dict:
    normalised = relative.replace("\\", "/")
    verified_folder = "openlab_reference/2024-02-01_2024-10-31_ida_ki_export/01_acceleration_trigger/"
    if verified_folder not in normalised:
        return {}
    match = re.fullmatch(r"acc_(2024)_(\d{2})_(\d{2})_T(\d{2})_(\d{2})_(\d{2})\.csv", path.name)
    if not match:
        return {}
    try:
        yy, mm, dd, hh, mi, ss = match.groups()
        stamp = pd.Timestamp(f"{yy}-{mm}-{dd}T{hh}:{mi}:{ss}Z")
        if not pd.Timestamp("2024-02-01T00:00:00Z") <= stamp < pd.Timestamp("2024-11-01T00:00:00Z"):
            return {}
        sample = _rio_csv_sample(path)
        for line_index, line in enumerate(sample.splitlines()):
            if not all(name in line for name in OPENLAB_2024_VERTICAL_CHANNELS):
                continue
            candidates = []
            for delimiter in [",", ";", "\t", "|"]:
                cells = next(csv.reader([line], delimiter=delimiter))
                if all(cells.count(name) == 1 for name in OPENLAB_2024_VERTICAL_CHANNELS):
                    candidates.append(delimiter)
            if len(candidates) != 1:
                return {}
            return {"dataset": "openlab_2024", "timestamp_utc": stamp.isoformat(),
                    "group_id": stamp.strftime("%Y-%m-%d"), "state_label": "reference",
                    "fs_hz": 500.0, "sample_axis": 0, "header": 0, "skiprows": line_index,
                    "delimiter": candidates[0], "channel_columns": json.dumps(OPENLAB_2024_VERTICAL_CHANNELS),
                    "metadata_status": "VERIFIED_OPENLAB_2024_PROTOCOL_AND_EXACT_HEADER"}
    except (ValueError, OSError, StopIteration, csv.Error):
        return {}
    return {}


def validate_manifest(df: pd.DataFrame, roots: dict[str, Path]) -> pd.DataFrame:
    """Return a normalised manifest or raise ManifestValidationError with diagnostics.

    Roots are explicit, paths are resolved and contained, and /home roots are refused.
    The production notebook additionally enforces /data roots; explicit temporary
    roots remain possible for isolated unit tests. No signal arrays are loaded here.
    """
    if not df.columns.is_unique:
        raise ManifestValidationError(pd.DataFrame([{"row": None, "record_id": "", "field": "manifest", "message": "Manifest column names must be unique"}]))
    frame = df.copy().reset_index(drop=True)
    errors = []
    source_identities = {}
    if frame.empty:
        raise ManifestValidationError(pd.DataFrame([{"row": None, "record_id": "", "field": "manifest", "message": "No acquisition rows were supplied; complete the manifest before running scientific analysis"}]))
    required = ["dataset", "record_id", "acquisition_id", "group_id", "state_label", "path", "fs_hz"]
    for field in required:
        if field not in frame:
            errors.append({"row": None, "record_id": "", "field": field, "message": f"Missing required manifest column {field!r}"})
    if errors:
        raise ManifestValidationError(pd.DataFrame(errors))
    for field in MANIFEST_COLUMNS:
        if field not in frame:
            frame[field] = ""
        frame[field] = frame[field].astype(object)
    for field in ["dataset", "record_id", "acquisition_id", "group_id", "state_label", "scenario", "excitation", "setup_id", "data_key", "time_column"]:
        frame[field] = frame[field].map(_rio_text)
    frame["excitation"] = frame["excitation"].str.upper()
    for index, row in frame.iterrows():
        record = _rio_text(row.get("record_id"))
        def error(field, message):
            errors.append({"row": str(index), "record_id": record, "field": field, "message": str(message)})
        dataset = _rio_text(row["dataset"])
        if dataset not in ALLOWED_DATASETS:
            error("dataset", f"dataset must be one of {sorted(ALLOWED_DATASETS)}, not {dataset!r}")
        for field in ["record_id", "acquisition_id", "group_id"]:
            if not _rio_text(row[field]):
                error(field, f"{field} is required; one physical group must not span split roles")
        state = _rio_text(row["state_label"]).lower()
        if state not in {"reference", "altered", "unknown"}:
            error("state_label", "state_label must be reference, altered, or unknown")
        frame.at[index, "state_label"] = state
        if dataset in {"openlab_2024", "z24_ems"} or not _rio_empty(row["timestamp_utc"]):
            try:
                stamp = _rio_timestamp(row["timestamp_utc"])
                frame.at[index, "timestamp_utc"] = stamp.isoformat()
            except (ValueError, TypeError, OverflowError) as exc:
                error("timestamp_utc", exc)
        if dataset == "z24_pdt":
            for field in ["scenario", "setup_id", "excitation"]:
                if not _rio_text(row[field]):
                    error(field, f"z24_pdt requires explicit {field}; do not merge roving setups or excitation runs")
            if row["excitation"] and row["excitation"] not in {"AVT", "FVT"}:
                error("excitation", "z24_pdt excitation must be verified AVT or FVT; other/unknown protocols require a separately specified analysis")
            raw_scenario = _rio_text(row["scenario"])
            if raw_scenario:
                try:
                    number = _rio_int(raw_scenario, "scenario")
                    if not 1 <= number <= 17:
                        raise ValueError("Z24 PDT scenario must be 01–17")
                    frame.at[index, "scenario"] = f"{number:02d}"
                except ValueError as exc:
                    error("scenario", exc)
        try:
            fs = _rio_sampling_rate(row["fs_hz"])
            frame.at[index, "fs_hz"] = fs
        except (ValueError, TypeError) as exc:
            error("fs_hz", exc)
        try:
            root = _rio_root_for(dataset, roots)
            supplied = Path(_rio_text(row["path"])).expanduser()
            actual = (supplied if supplied.is_absolute() else root / supplied).resolve(strict=True)
            if not _rio_within(actual, root) or not actual.is_file():
                raise ValueError(f"Input file must resolve inside declared dataset root {root}: {actual}")
            frame.at[index, "path"] = str(actual)
            source_identities[index] = _rio_file_identity(actual)[:2]
        except (ValueError, OSError, RuntimeError) as exc:
            error("path", exc)
        try:
            keys = _rio_list(row["channel_keys"], "channel_keys")
            columns = _rio_list(row["channel_columns"], "channel_columns")
            physical_names = _rio_list(row["channel_names"], "channel_names")
            if keys and (columns or _rio_text(row["data_key"])):
                raise ValueError("Use channel_keys OR data_key/channel_columns, not both")
            fmt = _rio_text(row["format"]).lower() or Path(_rio_text(row["path"])).suffix.lstrip(".").lower()
            fmt = "csv" if fmt in {"txt", "asc", "dat"} else fmt
            if fmt not in {"csv", "mat", "npz", "h5", "hdf5"}:
                raise ValueError(f"Unsupported signal format {fmt!r}")
            frame.at[index, "format"] = fmt
            frame.at[index, "channel_keys"] = json.dumps(keys)
            frame.at[index, "channel_columns"] = json.dumps(columns)
            frame.at[index, "channel_names"] = json.dumps(physical_names)
            if physical_names and len(physical_names) < 2:
                raise ValueError("channel_names must declare at least two synchronised physical channels")
            if physical_names and (keys or columns) and len(physical_names) != len(keys or columns):
                raise ValueError("channel_names must have one physical label per selected channel, in selection order")
            if keys and len(keys) < 2:
                raise ValueError("Select at least two simultaneous channel_keys")
            if fmt == "csv":
                if keys or not columns or _rio_text(row["data_key"]):
                    raise ValueError("CSV requires explicit channel_columns (JSON list of column names or zero-based indexes)")
                _rio_header(row["header"])
                if _rio_header(row["header"]) is None and any(isinstance(v, str) for v in columns):
                    raise ValueError("CSV string channel_columns require an actual header row")
                frame.at[index, "decimal"] = _rio_decimal(row["decimal"])
                if _rio_text(row["delimiter"]) == "":
                    raise ValueError("CSV delimiter must be explicit or 'auto'")
                if len(columns) < 2:
                    raise ValueError("Select at least two simultaneous CSV channel_columns")
            elif not keys and not _rio_text(row["data_key"]):
                # Loading may infer a unique numeric matrix, but the manifest does not
                # silently certify the shape/meaning of that matrix.
                raise ValueError(f"{fmt} requires explicit data_key or channel_keys; inspect file_format_probes.json")
            elif columns and not all(isinstance(v, int) and not isinstance(v, bool) for v in columns):
                raise ValueError("MAT/NPZ/HDF5 matrix channel_columns must be nonnegative integer indexes")
            if dataset == "z24_pdt" and not keys and (fmt != "csv" or all(isinstance(v, int) for v in columns)) and not physical_names:
                raise ValueError("Z24 PDT matrix/index selectors require channel_names: explicit physical reference-channel labels; array positions do not identify sensor locations")
            if not keys:
                axis = _rio_int(row["sample_axis"], "sample_axis")
                if axis not in {0, 1}:
                    raise ValueError("sample_axis must be 0 (rows=samples) or 1 (columns=samples)")
                if fmt == "csv" and axis != 0:
                    raise ValueError("CSV is tabular: sample_axis must be 0")
                frame.at[index, "sample_axis"] = axis
            skip = _rio_int(row["skiprows"], "skiprows", default=0)
            if skip < 0:
                raise ValueError("skiprows cannot be negative")
            frame.at[index, "skiprows"] = skip
        except (ValueError, TypeError) as exc:
            error("selectors", exc)
    duplicate_ids = frame["record_id"].astype(str).duplicated(keep=False)
    for index in frame.index[duplicate_ids]:
        errors.append({"row": str(index), "record_id": str(frame.at[index, "record_id"]), "field": "record_id", "message": "record_id must uniquely identify a logical raw record"})
    duplicate_acquisitions = frame[["dataset", "acquisition_id"]].fillna("").astype(str).duplicated(keep=False)
    for index in frame.index[duplicate_acquisitions]:
        errors.append({"row": str(index), "record_id": str(frame.at[index, "record_id"]), "field": "acquisition_id", "message": "One manifest row must represent one simultaneous acquisition; duplicate dataset/acquisition_id rows require a corrected logical-record definition"})
    selectors = frame[["path", "data_key", "channel_keys", "channel_columns"]].fillna("").astype(str).copy()
    # Renaming logical IDs, reordering a selected channel set, or using a hard
    # link must not count the same digital acquisition twice.
    for index in selectors.index:
        selectors.at[index, "path"] = str(source_identities.get(index, selectors.at[index, "path"]))
        for field in ["channel_keys", "channel_columns"]:
            try:
                selectors.at[index, field] = json.dumps(sorted(_rio_list(frame.at[index, field], field), key=str))
            except (ValueError, TypeError):
                pass  # field-specific diagnostics are already recorded above
    duplicates = selectors.duplicated(keep=False)
    for index in frame.index[duplicates]:
        errors.append({"row": str(index), "record_id": str(frame.at[index, "record_id"]), "field": "selectors", "message": "Identical source path and selectors appear more than once; duplicate acquisitions would overweight evidence"})
    if errors:
        raise ManifestValidationError(pd.DataFrame(errors))
    return frame.reset_index(drop=True)


def _rio_lookup(container: Any, key: str):
    """Resolve exact top-level keys first, then explicit dotted/slash struct paths."""
    if isinstance(container, Mapping) and key in container:
        return container[key]
    value = container
    for part in re.split(r"[./]", key):
        if part == "":
            continue
        if isinstance(value, Mapping) and part in value:
            value = value[part]
        elif isinstance(value, np.ndarray) and value.dtype.names and part in value.dtype.names:
            value = value[part]
        elif hasattr(value, part):
            value = getattr(value, part)
        else:
            raise ValueError(f"Selector {key!r} cannot resolve component {part!r}; inspect container metadata")
    return value


def _rio_numeric_array(value: Any, selector: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "biuf":
        raise ValueError(f"Selector {selector!r} is not a real numeric array ({array.dtype})")
    return np.asarray(array, dtype=np.float64)


def _rio_select_arrays(getter, row: Mapping[str, Any]):
    keys = _rio_list(row.get("channel_keys"), "channel_keys")
    columns = _rio_list(row.get("channel_columns"), "channel_columns")
    if keys:
        vectors = []
        for key in keys:
            array = _rio_numeric_array(getter(str(key)), str(key)).squeeze()
            if array.ndim != 1:
                raise ValueError(f"channel_keys must select one vector each; {key!r} has shape {array.shape}")
            vectors.append(array)
        if len({len(v) for v in vectors}) != 1:
            raise ValueError("Selected channel vectors have unequal lengths; synchronisation must be resolved upstream")
        return np.column_stack(vectors), [str(k) for k in keys]
    key = _rio_text(row.get("data_key"))
    if not key:
        raise ValueError("An explicit data_key or channel_keys list is required")
    array = _rio_numeric_array(getter(key), key)
    if array.ndim != 2:
        raise ValueError(f"data_key {key!r} must select one 2D numeric array, got {array.shape}")
    axis = _rio_int(row.get("sample_axis"), "sample_axis")
    if axis not in {0, 1}:
        raise ValueError("sample_axis must be 0 or 1")
    if axis == 1:
        array = array.T
    if columns:
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in columns):
            raise ValueError("Container matrix channel_columns must contain zero-based integer indexes")
        if min(columns) < 0 or max(columns) >= array.shape[1]:
            raise ValueError("channel_columns index is outside the selected numeric matrix")
        array = array[:, columns]
        names = [f"{key}[{i}]" for i in columns]
    else:
        names = [f"{key}[{i}]" for i in range(array.shape[1])]
    return array, names


def _rio_h5_get(values, key: str):
    """Read one local numeric dataset, excluding untracked dependent files."""
    import h5py
    parts = [part for part in key.split("/") if part]
    if not parts or any(part in {".", ".."} for part in parts):
        raise ValueError("HDF5 selectors must be explicit dataset paths without '.' or '..'")
    current = values
    for part in parts:
        if not isinstance(current, (h5py.File, h5py.Group)):
            raise ValueError(f"HDF5 selector {key!r} traverses a non-group object")
        link = current.get(part, getlink=True)
        if isinstance(link, h5py.ExternalLink):
            raise ValueError(f"HDF5 selector {key!r} uses an external-file link; export a self-contained numeric dataset")
        current = current[part]
        if Path(current.file.filename).resolve() != Path(values.filename).resolve():
            raise ValueError(f"HDF5 selector {key!r} resolves into an untracked external file")
    if not isinstance(current, h5py.Dataset):
        raise ValueError(f"HDF5 selector {key!r} names a group, not a numeric dataset")
    if current.is_virtual or current.external:
        raise ValueError(f"HDF5 selector {key!r} uses virtual/external storage; export a self-contained numeric dataset")
    return current[()]


def _rio_declared_channel_names(row: Mapping[str, Any], inferred: list[str], count: int) -> list[str]:
    declared = _rio_list(row.get("channel_names"), "channel_names")
    if declared and len(declared) != count:
        raise ValueError("channel_names count does not match the loaded selected channels")
    return declared or inferred


def _rio_check_auto_pdt_labels(values: Mapping[str, Any], row: Mapping[str, Any]):
    """Recheck auto-discovered physical mappings against the file actually loaded."""
    if not _rio_text(row.get("metadata_status")).startswith("VERIFIED_PDT_"):
        return
    if "labelshulp" not in values or _rio_text(row.get("data_key")) != "data":
        raise ValueError("Auto-discovered PDT source no longer has its verified data/labelshulp schema")
    labels = _rio_mat_labels(values["labelshulp"])
    columns = _rio_list(row.get("channel_columns"), "channel_columns")
    names = _rio_list(row.get("channel_names"), "channel_names")
    if not columns or len(columns) != len(names) or any(not isinstance(i, int) or i >= len(labels) for i in columns):
        raise ValueError("Auto-discovered PDT physical channel mapping is incomplete")
    if [labels[i] for i in columns] != names or any(labels.count(name) != 1 for name in names):
        raise ValueError("PDT labelshulp changed or contradicts the auto-discovered physical channel mapping; rerun metadata discovery")
    for key in ["fs", "Fs", "sample_rate", "sampling_rate"]:
        if key in values:
            value = np.asarray(values[key]).squeeze()
            if value.ndim != 0 or not math.isclose(_rio_sampling_rate(value.item()), float(row["fs_hz"]), rel_tol=0, abs_tol=1e-9):
                raise ValueError("PDT embedded sampling-rate metadata contradicts the auto-discovered protocol")


def load_acquisition(row) -> tuple[np.ndarray, float, list[str]]:
    """Load exactly one declared simultaneous record as samples × channels.

    No resampling, smoothing, imputation, concatenation of acquisitions or hidden
    channel selection is performed. NaN/Inf, ambiguous/object arrays and mismatched
    channels are rejected. MATLAB v7.3/HDF5 numeric datasets need optional h5py.
    """
    row = row.to_dict() if hasattr(row, "to_dict") else dict(row)
    path = Path(_rio_text(row.get("path"))).resolve(strict=True)
    identity_before = _rio_file_identity(path)
    fs = _rio_sampling_rate(row.get("fs_hz", float("nan")))
    fmt = _rio_text(row.get("format")).lower() or path.suffix.lstrip(".").lower()
    fmt = "csv" if fmt in {"txt", "asc", "dat"} else fmt
    keys = _rio_list(row.get("channel_keys"), "channel_keys")
    columns = _rio_list(row.get("channel_columns"), "channel_columns")
    declared_names = _rio_list(row.get("channel_names"), "channel_names")
    if keys and (columns or _rio_text(row.get("data_key"))):
        raise ValueError("Use channel_keys OR data_key/channel_columns, not both")
    if _rio_text(row.get("dataset")) == "z24_pdt" and not keys and (fmt != "csv" or all(isinstance(v, int) for v in columns)) and not declared_names:
        raise ValueError("Z24 PDT matrix/index selectors require explicit physical channel_names")
    if fmt == "csv":
        if keys or _rio_text(row.get("data_key")):
            raise ValueError("CSV uses channel_columns, not data_key/channel_keys")
        skip = _rio_int(row.get("skiprows"), "skiprows", default=0)
        if skip < 0:
            raise ValueError("skiprows cannot be negative")
        delimiter = _rio_delimiter(row.get("delimiter"), path, skip)
        decimal = _rio_decimal(row.get("decimal"))
        header = _rio_header(row.get("header"))
        raw_header = _rio_csv_header(path, delimiter, skip, header)
        with warnings.catch_warnings():
            warnings.simplefilter("error", pd.errors.ParserWarning)
            # index_col=False prevents pandas from silently interpreting an
            # extra data field as a row index and shifting sensor columns.
            frame = pd.read_csv(path, sep=delimiter, decimal=decimal, header=header, skiprows=skip,
                                index_col=False, engine="python", encoding="utf-8-sig")
        selectors = _rio_list(row.get("channel_columns"), "channel_columns")
        if not selectors:
            raise ValueError("CSV channel_columns must be explicit")
        if all(isinstance(v, int) and not isinstance(v, bool) for v in selectors):
            if min(selectors) < 0 or max(selectors) >= frame.shape[1]:
                raise ValueError("CSV channel column index is out of bounds")
            selected = frame.iloc[:, selectors]
            selected_header = [raw_header[v] for v in selectors] if raw_header is not None else []
        elif all(isinstance(v, str) for v in selectors):
            if raw_header is None:
                raise ValueError("CSV string channel selectors require an actual header row")
            missing = [v for v in selectors if v not in frame.columns]
            if missing:
                raise ValueError(f"CSV channel names not found: {missing}; available columns={list(frame.columns)}")
            selected = frame.loc[:, selectors]
            selected_header = selectors
        else:
            raise ValueError("CSV channel_columns must be all integer indexes or all exact string names")
        if raw_header is not None and any(raw_header.count(name) != 1 for name in selected_header):
            raise ValueError("Selected CSV physical headers are duplicated; pandas header renaming must not conceal ambiguous channel identity")
        # Time is metadata and must not accidentally be included as a vibration channel.
        time_column = _rio_text(row.get("time_column"))
        if time_column and any(str(v) == time_column for v in selected.columns):
            raise ValueError("time_column was also selected as a vibration channel")
        try:
            array = selected.apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
        except (ValueError, TypeError) as exc:
            raise ValueError("Selected CSV signal columns contain nonnumeric cells; correct header/skiprows/selectors") from exc
        names = [str(v) for v in selected.columns]
    elif fmt == "mat":
        from scipy.io import loadmat
        is_h5 = False
        if importlib.util.find_spec("h5py"):
            import h5py
            is_h5 = h5py.is_hdf5(path)
        try:
            if is_h5:
                raise NotImplementedError("MATLAB v7.3/HDF5 container")
            values = loadmat(path, simplify_cells=True)
            _rio_check_auto_pdt_labels(values, row)
            array, names = _rio_select_arrays(lambda key: _rio_lookup(values, key), row)
        except NotImplementedError:
            if not importlib.util.find_spec("h5py"):
                raise ValueError("MATLAB v7.3 file requires h5py and explicit numeric HDF5 dataset selectors")
            import h5py
            with h5py.File(path, "r") as values:
                array, names = _rio_select_arrays(lambda key: _rio_h5_get(values, key), row)
    elif fmt == "npz":
        with np.load(path, allow_pickle=False) as values:
            array, names = _rio_select_arrays(lambda key: values[key], row)
    elif fmt in {"h5", "hdf5"}:
        if not importlib.util.find_spec("h5py"):
            raise ValueError("HDF5 file requires optional h5py")
        import h5py
        with h5py.File(path, "r") as values:
            array, names = _rio_select_arrays(lambda key: _rio_h5_get(values, key), row)
    else:
        raise ValueError(f"Unsupported acquisition format {fmt!r}")
    array = np.asarray(array, dtype=np.float64)
    names = _rio_declared_channel_names(row, names, array.shape[1] if array.ndim == 2 else 0)
    if _rio_file_identity(path) != identity_before:
        raise ValueError("Input file changed during acquisition loading; freeze inputs and rerun provenance hashing")
    if array.ndim != 2 or array.shape[0] < 8 or array.shape[1] < 2:
        raise ValueError(f"Expected ≥8 synchronised samples and ≥2 selected channels; got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("Selected raw channels contain NaN/Inf; no silent interpolation or raw-signal imputation is allowed")
    if len(names) != array.shape[1]:
        raise ValueError("Channel-name count does not match the loaded array")
    return np.ascontiguousarray(array), fs, names
