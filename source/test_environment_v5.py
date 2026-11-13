"""Measured-temperature metadata tests using isolated, labelled local fixtures."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import environment as env


class MeasuredEnvironmentContract(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.raw = self.base / "openlab"
        self.folder = self.raw / env.OPENLAB_ENVIRONMENT_RELATIVE
        self.folder.mkdir(parents=True)
        self.out = self.base / "derived"
        for month in range(2, 11):
            data = {"Timestamp": [f"2024-{month:02d}-01 00:{minute:02d}:00" for minute in [0,10,20]],
                    "G_HTST_ENVR_EN0000_0": [month, month+1, month+2],
                    "G_HTSH_ENVR_EN0000_0": [80, 81, 82],
                    "G_PYRS_ENVR_EN0000_0": [0, 1, 2]}
            pd.DataFrame(data).to_csv(self.folder / f"environment_2024_{month:02d}.csv", index=False)

    def tearDown(self):
        self.tmp.cleanup()

    def manifest(self, times=None):
        times = times or ["2024-05-01T00:00:00Z", "2024-05-01T00:05:00Z", "2024-05-01T00:08:00Z"]
        return pd.DataFrame({"dataset": ["openlab_2024"] * len(times),
                             "acquisition_id": [f"acq_{i}" for i in range(len(times))],
                             "timestamp_utc": times, "partition": ["fit"] * len(times)})

    def month(self, month=5):
        return self.folder / f"environment_2024_{month:02d}.csv"

    def rewrite(self, transform, month=5):
        path = self.month(month)
        frame = pd.read_csv(path)
        transform(frame).to_csv(path, index=False)

    def test_published_schema_localises_only_documented_environment_to_utc(self):
        table, audit = env.load_openlab_2024_temperature(self.raw)
        self.assertEqual(len(table), 27)
        self.assertEqual(str(table.temperature_timestamp_utc.dtype), "datetime64[ns, UTC]")
        self.assertEqual(audit["evidence"]["temperature_unit"], "degree_Celsius")
        self.assertEqual(audit["evidence"]["naive_timestamp_timezone"], "UTC")
        self.assertEqual(len(audit["source_files"]), 9)
        self.assertTrue(all(len(file["sha256"]) == 64 for file in audit["source_files"]))
        self.assertFalse(audit["temperature_imputed"])

    def test_nearest_measured_values_tie_to_past_and_audit_future_match(self):
        paths = list(self.folder.glob("*.csv"))
        before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        manifest = self.manifest()
        result, audit = env.attach_openlab_2024_temperature(manifest, self.raw, self.out)
        np.testing.assert_array_equal(result.cov_temperature, [5., 5., 6.])
        self.assertEqual([c for c in result if c.startswith("cov_")], ["cov_temperature"])
        self.assertEqual(audit["future_measurements_used"], 1)
        self.assertEqual(audit["maximum_absolute_time_offset_seconds"], 300.)
        self.assertEqual(audit["matched_fraction"], 1.)
        matches = pd.read_csv(self.out / "openlab_temperature_acquisition_matches.csv")
        np.testing.assert_array_equal(matches.temperature_signed_offset_seconds, [0., -300., 120.])
        self.assertEqual(manifest.columns.tolist(), ["dataset", "acquisition_id", "timestamp_utc", "partition"])
        self.assertTrue((self.out / "openlab_temperature_coverage_by_date.csv").exists())
        self.assertTrue((self.out / "openlab_temperature_coverage_by_partition.csv").exists())
        self.assertEqual(before, {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})

    def test_acquisition_timezone_offset_represents_same_instant(self):
        result, _ = env.attach_openlab_2024_temperature(self.manifest(["2024-05-01T01:00:00+01:00"]), self.raw, self.out)
        self.assertEqual(result.cov_temperature.tolist(), [5.])

    def test_naive_acquisition_dates_are_not_granted_the_provider_exception(self):
        for time in ["2024-05-01 00:00:00", "2024-05-01", 1714521600, "May 1 2024 00:00 UTC"]:
            with self.subTest(time=time), self.assertRaises(env.EnvironmentMetadataError):
                env.attach_openlab_2024_temperature(self.manifest([time]), self.raw, self.out)

    def test_partial_coverage_preserves_all_acquisitions_and_missingness(self):
        manifest = self.manifest(["2024-05-01T00:00:00Z", "2024-05-01T01:00:00Z"])
        with self.assertRaises(env.EnvironmentMetadataError) as exc:
            env.attach_openlab_2024_temperature(manifest, self.raw, self.out)
        self.assertEqual(exc.exception.audit["unmatched_acquisitions"], 1)
        result, audit = env.attach_openlab_2024_temperature(manifest, self.raw, self.out, require_complete=False)
        self.assertEqual(len(result), 2)
        self.assertEqual(result.cov_temperature.iloc[0], 5.)
        self.assertTrue(pd.isna(result.cov_temperature.iloc[1]))
        self.assertEqual(audit["matched_fraction"], .5)
        self.assertFalse(audit["temperature_imputed"])

    def test_missing_nearest_observation_is_not_replaced_with_another_temperature(self):
        def remove(frame):
            frame.loc[1, "G_HTST_ENVR_EN0000_0"] = np.nan
            return frame
        self.rewrite(remove)
        result, audit = env.attach_openlab_2024_temperature(self.manifest(["2024-05-01T00:08:00Z"]), self.raw, self.out, require_complete=False)
        self.assertTrue(result.cov_temperature.isna().all())
        self.assertEqual(audit["unmatched_acquisitions"], 1)
        matches = pd.read_csv(self.out / "openlab_temperature_acquisition_matches.csv")
        self.assertEqual(matches.temperature_match_status.iloc[0], "NEAREST_MEASUREMENT_TEMPERATURE_MISSING")

    def test_duplicate_timestamps_are_not_silently_averaged_or_removed(self):
        self.rewrite(lambda frame: pd.concat([frame, frame.iloc[:1]], ignore_index=True))
        with self.assertRaisesRegex(env.EnvironmentMetadataError, "Duplicate"):
            env.attach_openlab_2024_temperature(self.manifest(), self.raw, self.out)

    def test_multiple_or_changed_temperature_columns_require_explicit_schema_choice(self):
        self.rewrite(lambda frame: frame.assign(other_temperature=frame.G_HTST_ENVR_EN0000_0 + 1))
        with self.assertRaisesRegex(env.EnvironmentMetadataError, "header differs"):
            env.attach_openlab_2024_temperature(self.manifest(), self.raw, self.out)

    def test_existing_measured_covariate_is_not_overwritten(self):
        manifest = self.manifest().assign(cov_temperature=[10.,11.,12.])
        with self.assertRaisesRegex(env.EnvironmentMetadataError, "already supplied"):
            env.attach_openlab_2024_temperature(manifest, self.raw, self.out)

    def test_non_openlab_rows_and_existing_other_cohort_values_are_preserved(self):
        manifest = pd.concat([self.manifest().assign(cov_temperature=np.nan),
                              pd.DataFrame({"dataset":["z24_pdt"], "acquisition_id":["pdt_1"],
                                            "timestamp_utc":[""], "cov_temperature":[20.]})], ignore_index=True)
        result, audit = env.attach_openlab_2024_temperature(manifest, self.raw, self.out)
        self.assertEqual(len(result), 4)
        self.assertEqual(result.cov_temperature.tolist(), [5.,5.,6.,20.])
        self.assertEqual(audit["acquisitions"], 3)

    def test_outside_symlink_home_inputs_and_output_inside_raw_tree_rejected(self):
        source = self.month()
        outside = self.base / "outside.csv"
        outside.write_bytes(source.read_bytes())
        source.unlink()
        source.symlink_to(outside)
        with self.assertRaisesRegex(env.EnvironmentMetadataError, "outside"):
            env.attach_openlab_2024_temperature(self.manifest(), self.raw, self.out)
        with self.assertRaises(env.EnvironmentMetadataError):
            env.attach_openlab_2024_temperature(self.manifest(), Path("/home"), self.out)
        with self.assertRaisesRegex(env.EnvironmentMetadataError, "outside the raw"):
            env.attach_openlab_2024_temperature(self.manifest(), self.raw, self.raw / "derived")

    def test_file_changed_during_read_is_rejected(self):
        original = pd.read_csv
        def changed(*args, **kwargs):
            frame = original(*args, **kwargs)
            path = Path(args[0])
            path.write_text(path.read_text() + "\n")
            return frame
        with patch.object(env.pd, "read_csv", side_effect=changed):
            with self.assertRaisesRegex(env.EnvironmentMetadataError, "changed during"):
                env.attach_openlab_2024_temperature(self.manifest(), self.raw, self.out)

    def test_join_identifiers_tolerance_and_source_month_are_explicit(self):
        for tolerance in [-1, 301, True, float("nan")]:
            with self.subTest(tolerance=tolerance), self.assertRaises(env.EnvironmentMetadataError):
                env.attach_openlab_2024_temperature(self.manifest(), self.raw, self.out, tolerance_seconds=tolerance)
        manifest = self.manifest()
        manifest.loc[1, "acquisition_id"] = manifest.loc[0, "acquisition_id"]
        with self.assertRaisesRegex(env.EnvironmentMetadataError, "identifiers"):
            env.attach_openlab_2024_temperature(manifest, self.raw, self.out)
        self.rewrite(lambda frame: frame.assign(Timestamp=frame.Timestamp.str.replace("2024-05-", "2024-06-", regex=False)))
        with self.assertRaisesRegex(env.EnvironmentMetadataError, "source month"):
            env.attach_openlab_2024_temperature(self.manifest(), self.raw, self.out)


if __name__ == "__main__":
    unittest.main()
