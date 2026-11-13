"""Measured openLAB 2024 air-temperature metadata, with an auditable UTC join.

This recogniser is confined to the published February--October 2024 export.
The provider documents UTC despite unzoned CSV strings; Table 3 identifies one
air-temperature channel in degrees Celsius. No alternative sensor, interpolation,
temperature proxy, outcome-dependent selection or environmental model is used.
Nearest-neighbour matching is retrospective and can use a measurement up to five
minutes after acquisition start; it does not establish prospective availability.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import re
import warnings
from typing import Any

import numpy as np
import pandas as pd


OPENLAB_ENVIRONMENT_COLUMNS = [
    "Timestamp", "G_HTST_ENVR_EN0000_0", "G_HTSH_ENVR_EN0000_0", "G_PYRS_ENVR_EN0000_0",
]
OPENLAB_ENVIRONMENT_RELATIVE = Path("openlab_reference/2024-02-01_2024-10-31_ida_ki_export/02_environment")
OPENLAB_ENVIRONMENT_EVIDENCE = {
    "dataset_doi": "10.25532/OPARA-660",
    "dataset_url": "https://opara.zih.tu-dresden.de/handle/123456789/1048",
    "data_article_doi": "10.1016/j.dib.2025.111624",
    "data_article_url": "https://doi.org/10.1016/j.dib.2025.111624",
    "schema_table": "Table 3, directory 02_environment",
    "temperature_channel": "G_HTST_ENVR_EN0000_0",
    "temperature_quantity": "MEASURED_AMBIENT_AIR_TEMPERATURE",
    "temperature_unit": "degree_Celsius",
    "naive_timestamp_timezone": "UTC",
    "timezone_evidence": "Provider documentation explicitly states that CSV timestamps omit timezone information but represent UTC.",
    "documented_interval_seconds": 600,
}


class EnvironmentMetadataError(ValueError):
    """Invalid or incomplete measured metadata; ``audit`` explains the blockage."""
    def __init__(self, message: str, audit: dict | None = None):
        self.audit = audit or {}
        super().__init__(message)


def _env_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _env_identity(path: Path) -> tuple:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _env_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _env_json_value(value: Any):
    if isinstance(value, dict):
        return {str(k): _env_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_env_json_value(v) for v in value]
    if isinstance(value, np.generic):
        return _env_json_value(value.item())
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is pd.NA or value is pd.NaT:
        return None
    return value


def _env_save_audit(out_dir: Path | None, audit: dict, matches: pd.DataFrame | None = None):
    if out_dir is None:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "openlab_temperature_join_audit.json").write_text(
        json.dumps(_env_json_value(audit), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    if matches is not None:
        matches.to_csv(out_dir / "openlab_temperature_acquisition_matches.csv", index=False)
        for field, name in [("acquisition_utc_date", "openlab_temperature_coverage_by_date.csv"),
                            ("partition", "openlab_temperature_coverage_by_partition.csv")]:
            if field in matches:
                summary = matches.groupby(field, dropna=False, sort=True).agg(
                    acquisitions=("acquisition_id", "size"),
                    matched=("temperature_match_accepted", "sum"),
                    maximum_absolute_time_offset_seconds=("temperature_abs_offset_seconds", "max"))
                summary["matched_fraction"] = summary["matched"] / summary["acquisitions"]
                summary.reset_index().to_csv(out_dir / name, index=False)


def _env_utc_acquisition(value: Any) -> pd.Timestamp:
    """Acquisition manifests retain the general explicit-timezone requirement."""
    if isinstance(value, (int, float, bool, np.number)) or value is None:
        raise EnvironmentMetadataError("Acquisition timestamps must be explicitly zoned ISO datetimes, not epochs or missing values")
    if isinstance(value, str) and not re.fullmatch(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})", value):
        raise EnvironmentMetadataError("Acquisition timestamps must use an ISO date/time with an explicit UTC offset")
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise EnvironmentMetadataError("Acquisition timestamp lacks an explicit timezone; validate the manifest before joining temperature")
    return stamp.tz_convert("UTC")


def load_openlab_2024_temperature(input_root: Path) -> tuple[pd.DataFrame, dict]:
    """Load the nine verified monthly environmental CSVs, read-only.

    Returned rows preserve measured values and source row numbers. Missing
    temperatures remain missing; duplicates and ambiguous schemas are rejected.
    Temporary absolute roots are allowed for isolated tests; the notebook enforces
    the production /data root. /home input roots are always refused.
    """
    supplied = Path(input_root).expanduser()
    if not supplied.is_absolute():
        raise EnvironmentMetadataError("Environmental input_root must be absolute")
    root = supplied.resolve(strict=True)
    if root == Path("/home") or Path("/home") in root.parents or not root.is_dir():
        raise EnvironmentMetadataError("Environmental inputs must use the declared read-only /data dataset root")
    folder = (root / OPENLAB_ENVIRONMENT_RELATIVE).resolve(strict=True)
    if not _env_within(folder, root) or not folder.is_dir():
        raise EnvironmentMetadataError("The verified 2024 environmental directory is absent or outside input_root")
    audit = {"status": "LOADING_VERIFIED_MEASURED_TEMPERATURE", "evidence": OPENLAB_ENVIRONMENT_EVIDENCE,
             "input_root": str(root), "source_files": [], "raw_inputs_written": False,
             "temperature_interpolated": False, "temperature_imputed": False}
    expected = {f"environment_2024_{month:02d}.csv" for month in range(2, 11)}
    available = {p.name for p in folder.glob("environment_*.csv")}
    if available != expected:
        audit["missing_monthly_files"] = sorted(expected - available)
        audit["unexpected_monthly_files"] = sorted(available - expected)
        raise EnvironmentMetadataError("The recognised export requires exactly the nine February--October 2024 environmental files", audit)
    pieces, source_identities = [], {}
    for name in sorted(expected):
        path = (folder / name).resolve(strict=True)
        if not _env_within(path, root):
            raise EnvironmentMetadataError(f"Environmental file resolves outside input_root: {name}", audit)
        before = _env_identity(path)
        source_identities[path] = before
        digest = _env_sha256(path)
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            header = next(csv.reader(stream), None)
        if header != OPENLAB_ENVIRONMENT_COLUMNS:
            audit["unrecognised_header"] = {"path": str(path), "header": header}
            raise EnvironmentMetadataError("Environmental header differs from the verified four-column schema; choose and document any alternative sensor explicitly", audit)
        with warnings.catch_warnings():
            warnings.simplefilter("error", pd.errors.ParserWarning)
            frame = pd.read_csv(path, sep=",", decimal=".", header=0, index_col=False,
                                encoding="utf-8-sig", engine="python", dtype={"Timestamp": str})
        if frame.empty:
            raise EnvironmentMetadataError(f"Environmental source has no measured rows: {name}", audit)
        text = frame["Timestamp"]
        if text.isna().any() or not text.str.fullmatch(r"2024-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?").all():
            raise EnvironmentMetadataError("Recognised source timestamps must be unzoned 2024 ISO datetimes; the documented UTC exception is confined to this exact schema", audit)
        # This UTC localisation is justified by the provider, not inferred from
        # geography, numerical offsets, file modification dates or model outcomes.
        stamp = pd.to_datetime(text, format="ISO8601", errors="raise", utc=True)
        month = int(re.fullmatch(r"environment_2024_(\d{2})\.csv", name).group(1))
        if not stamp.dt.month.eq(month).all():
            raise EnvironmentMetadataError(f"Environmental timestamps contradict their source month: {name}", audit)
        temperature = pd.to_numeric(frame["G_HTST_ENVR_EN0000_0"], errors="raise").astype(float)
        if np.isinf(temperature.to_numpy()).any():
            raise EnvironmentMetadataError("Measured environmental temperature contains infinity", audit)
        piece = pd.DataFrame({"temperature_timestamp_utc": stamp,
                              "measured_air_temperature_C": temperature,
                              "temperature_source_path": str(path),
                              "temperature_source_csv_row": np.arange(len(frame)) + 2})
        pieces.append(piece)
        if _env_identity(path) != before:
            raise EnvironmentMetadataError("An environmental input changed during hashing or loading", audit)
        audit["source_files"].append({"path": str(path), "sha256": digest, "rows": len(frame),
                                      "missing_temperature_rows": int(temperature.isna().sum()),
                                      "first_timestamp_utc": stamp.min().isoformat(),
                                      "last_timestamp_utc": stamp.max().isoformat(),
                                      "source_order_monotonic": bool(stamp.is_monotonic_increasing)})
    table = pd.concat(pieces, ignore_index=True).sort_values("temperature_timestamp_utc", kind="stable").reset_index(drop=True)
    duplicate = table.temperature_timestamp_utc.duplicated(keep=False)
    if duplicate.any():
        audit["duplicate_timestamp_examples"] = table.loc[duplicate, "temperature_timestamp_utc"].head(20).astype(str).tolist()
        raise EnvironmentMetadataError("Duplicate environmental UTC timestamps cannot be resolved by silently averaging or removing rows", audit)
    gaps = table.temperature_timestamp_utc.diff().dt.total_seconds().dropna()
    finite = table.measured_air_temperature_C.dropna()
    audit.update(status="VERIFIED_MEASURED_TEMPERATURE_LOADED", measured_rows=len(table),
                 missing_temperature_rows=int(table.measured_air_temperature_C.isna().sum()),
                 first_timestamp_utc=table.temperature_timestamp_utc.min().isoformat(),
                 last_timestamp_utc=table.temperature_timestamp_utc.max().isoformat(),
                 observed_minimum_interval_seconds=float(gaps.min()) if len(gaps) else None,
                 observed_maximum_interval_seconds=float(gaps.max()) if len(gaps) else None,
                 intervals_differing_from_documented_600_seconds=int(gaps.ne(600).sum()),
                 measured_temperature_minimum_C=float(finite.min()) if len(finite) else None,
                 measured_temperature_maximum_C=float(finite.max()) if len(finite) else None)
    if any(_env_identity(path) != identity for path, identity in source_identities.items()):
        raise EnvironmentMetadataError("An environmental source changed before the metadata snapshot completed", audit)
    return table, audit


def attach_openlab_2024_temperature(manifest: pd.DataFrame, input_root: Path, out_dir: Path | None,
                                    tolerance_seconds: float = 300, require_complete: bool = True) -> tuple[pd.DataFrame, dict]:
    """Attach one measured temperature per OpenLab acquisition, retaining all rows.

    The sole model predictor added is cov_temperature, in degrees Celsius. Exact
    nearest-time ties prefer the earlier measurement. All matching is independent
    of outcomes and partitions; predictors are not fitted here. With the default
    require_complete=True, a missing/nonfinite match blocks attachment so that a
    downstream regressor cannot silently replace an unmeasured temperature.
    """
    if isinstance(tolerance_seconds, bool) or not math.isfinite(float(tolerance_seconds)) or not 0 <= float(tolerance_seconds) <= 300:
        raise EnvironmentMetadataError("Temperature tolerance must be explicit and between 0 and 300 seconds for this protocol")
    root = Path(input_root).expanduser().resolve(strict=True)
    output = Path(out_dir).expanduser().resolve() if out_dir is not None else None
    if output is not None and _env_within(output, root):
        raise EnvironmentMetadataError("Environmental diagnostics must be written outside the raw input tree")
    required = {"dataset", "acquisition_id", "timestamp_utc"}
    if not required.issubset(manifest.columns):
        raise EnvironmentMetadataError("Temperature attachment requires a validated manifest with dataset, acquisition_id and timestamp_utc")
    result = manifest.copy().reset_index(drop=True)
    mask = result.dataset.eq("openlab_2024")
    if not mask.any():
        audit = {"status": "NO_OPENLAB_2024_ACQUISITIONS", "evidence": OPENLAB_ENVIRONMENT_EVIDENCE}
        _env_save_audit(output, audit)
        return result, audit
    existing = pd.to_numeric(result.loc[mask, "cov_temperature"], errors="raise") if "cov_temperature" in result else pd.Series(dtype=float)
    if existing.notna().any():
        raise EnvironmentMetadataError("OpenLab cov_temperature is already supplied; do not silently replace or mix measured metadata sources")
    try:
        environment, audit = load_openlab_2024_temperature(root)
        part = result.loc[mask].copy()
        if part.acquisition_id.isna().any() or part.acquisition_id.astype(str).str.strip().eq("").any() or part.acquisition_id.astype(str).str.strip().duplicated().any():
            raise EnvironmentMetadataError("OpenLab acquisition identifiers must be explicit and unique", audit)
        part["_environment_manifest_row"] = part.index
        part["acquisition_timestamp_utc"] = part.timestamp_utc.map(_env_utc_acquisition)
        order = part.sort_values("acquisition_timestamp_utc", kind="stable")
        matches = pd.merge_asof(order, environment, left_on="acquisition_timestamp_utc",
                                right_on="temperature_timestamp_utc", direction="nearest",
                                tolerance=pd.Timedelta(seconds=float(tolerance_seconds)))
        matches["temperature_signed_offset_seconds"] = (matches.temperature_timestamp_utc - matches.acquisition_timestamp_utc).dt.total_seconds()
        matches["temperature_abs_offset_seconds"] = matches.temperature_signed_offset_seconds.abs()
        matches["temperature_match_accepted"] = matches.temperature_timestamp_utc.notna() & np.isfinite(matches.measured_air_temperature_C)
        matches["temperature_match_status"] = np.where(matches.temperature_timestamp_utc.isna(), "NO_TIMESTAMP_WITHIN_TOLERANCE",
                                                       np.where(matches.temperature_match_accepted, "MEASURED_TEMPERATURE_MATCHED", "NEAREST_MEASUREMENT_TEMPERATURE_MISSING"))
        matches["acquisition_utc_date"] = matches.acquisition_timestamp_utc.dt.strftime("%Y-%m-%d")
        audit.update(status="COMPLETE_MEASURED_TEMPERATURE_ATTACHMENT" if matches.temperature_match_accepted.all() else "INCOMPLETE_MEASURED_TEMPERATURE_ATTACHMENT",
                     matching_rule="NEAREST_MEASURED_TIMESTAMP_TIE_TO_EARLIER", tolerance_seconds=float(tolerance_seconds),
                     matching_is_retrospective=True, nominal_prospective_availability_established=False,
                     acquisitions=len(matches), matched_acquisitions=int(matches.temperature_match_accepted.sum()),
                     unmatched_acquisitions=int((~matches.temperature_match_accepted).sum()),
                     matched_fraction=float(matches.temperature_match_accepted.mean()),
                     future_measurements_used=int((matches.temperature_signed_offset_seconds > 0).sum()),
                     maximum_absolute_time_offset_seconds=float(matches.temperature_abs_offset_seconds.max()) if matches.temperature_abs_offset_seconds.notna().any() else None,
                     all_acquisition_rows_retained=True, predictor_added="cov_temperature", predictor_unit="degree_Celsius")
        _env_save_audit(output, audit, matches.drop(columns=["_environment_manifest_row"]))
        if require_complete and not matches.temperature_match_accepted.all():
            raise EnvironmentMetadataError("Incomplete measured temperature coverage: attachment blocked rather than filling missing acquisition temperatures", audit)
        if "cov_temperature" not in result:
            result["cov_temperature"] = np.nan
        for _, row in matches.iterrows():
            result.at[int(row["_environment_manifest_row"]), "cov_temperature"] = float(row.measured_air_temperature_C) if row.temperature_match_accepted else np.nan
        return result, audit
    except Exception as exc:
        audit = getattr(exc, "audit", {}) or {"status": "TEMPERATURE_ATTACHMENT_BLOCKED", "evidence": OPENLAB_ENVIRONMENT_EVIDENCE}
        audit["error"] = f"{type(exc).__name__}: {exc}"
        _env_save_audit(output, audit)
        if isinstance(exc, EnvironmentMetadataError):
            raise
        raise EnvironmentMetadataError(str(exc), audit) from exc
