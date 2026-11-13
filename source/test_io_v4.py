"""Isolated I/O contract checks; no HPC dataset or historical result is accessed.

Run: python -m unittest discover -s work/aicset_revision_v3 -p test_io_v3.py -v
HDF5 tests explicitly skip when optional h5py is unavailable.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import pandas as pd
from scipy.io import savemat

import raw_io as rio


class RawIOContract(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.inputs = self.base / "raw"
        self.inputs.mkdir()
        self.roots = {"z24": self.inputs, "openlab": self.inputs}
        self.signal = np.arange(48, dtype=float).reshape(16, 3) / 7

    def tearDown(self):
        self.tmp.cleanup()

    def csv(self, name="signal.csv"):
        path = self.inputs / name
        pd.DataFrame(self.signal, columns=["R1V", "R2V", "R3V"]).to_csv(path, index=False)
        return path

    def row(self, path, **changes):
        data = {key: "" for key in rio.MANIFEST_COLUMNS}
        data.update(dataset="z24_ems", record_id="record_1", acquisition_id="acq_1",
                    group_id="week_1", timestamp_utc="1998-02-01T00:00:00Z",
                    state_label="reference", path=str(path), format="csv", fs_hz=100.,
                    channel_columns=json.dumps(["R1V", "R2V", "R3V"]), sample_axis=0,
                    delimiter=",", decimal=".", skiprows=0, header=0)
        data.update(changes)
        return data

    def validated(self, row):
        return rio.validate_manifest(pd.DataFrame([row]), self.roots).iloc[0].to_dict()

    def test_readonly_csv_physical_names_and_values(self):
        path = self.csv()
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        x, fs, names = rio.load_acquisition(self.validated(self.row(path)))
        np.testing.assert_allclose(x, self.signal)
        self.assertEqual(fs, 100.)
        self.assertEqual(names, ["R1V", "R2V", "R3V"])
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)

    def test_manifest_normalizes_identity_fields_and_excitation(self):
        row = self.row(self.csv(), dataset=" z24_pdt ", record_id=" r ", acquisition_id=" a ",
                       group_id=" s ", scenario=" 1 ", setup_id=" 01 ", excitation=" avt ")
        out = self.validated(row)
        self.assertEqual([out[k] for k in ["dataset", "record_id", "acquisition_id", "group_id", "scenario", "setup_id", "excitation"]],
                         ["z24_pdt", "r", "a", "s", "01", "01", "AVT"])

    def test_duplicate_acquisitions_cannot_hide_in_whitespace(self):
        a = self.row(self.csv())
        b = dict(a, record_id="record_2", dataset=" z24_ems ", acquisition_id=" acq_1 ",
                 channel_columns=json.dumps(["R3V", "R2V", "R1V"]))
        with self.assertRaises(rio.ManifestValidationError) as exc:
            rio.validate_manifest(pd.DataFrame([a, b]), self.roots)
        self.assertIn("acquisition_id", exc.exception.errors.field.tolist())
        self.assertIn("selectors", exc.exception.errors.field.tolist())

    def test_distinct_records_in_one_container_remain_supported(self):
        path = self.inputs / "records.npz"
        np.savez(path, first=self.signal, second=self.signal + 1)
        a = self.row(path, format="npz", data_key="first", channel_columns="[]")
        b = dict(a, record_id="record_2", acquisition_id="acq_2", data_key="second")
        result = rio.validate_manifest(pd.DataFrame([a, b]), self.roots)
        self.assertEqual(len(result), 2)

    def test_utc_offset_converted_without_changing_instant(self):
        out = self.validated(self.row(self.csv(), timestamp_utc="1998-02-01T00:30:00+01:00"))
        self.assertEqual(out["timestamp_utc"], "1998-01-31T23:30:00+00:00")

    def test_naive_numeric_or_missing_timestamps_are_rejected(self):
        path = self.csv()
        for value in ["1998-02-01", "1998-02-01T12:00:00", 886291200, "886291200", ""]:
            with self.subTest(value=value), self.assertRaises(rio.ManifestValidationError):
                self.validated(self.row(path, timestamp_utc=value))

    def test_booleans_cannot_be_sampling_rates_or_axes(self):
        path = self.csv()
        for changes in [{"fs_hz": True}, {"sample_axis": False}, {"skiprows": True}]:
            with self.subTest(changes=changes), self.assertRaises(rio.ManifestValidationError):
                self.validated(self.row(path, **changes))

    def test_explicit_decimal_comma_is_honored_not_inferred(self):
        path = self.inputs / "locale.csv"
        text = "R1V;R2V\n" + "\n".join(f"{i},25;{i},75" for i in range(16)) + "\n"
        path.write_text(text)
        row = self.row(path, delimiter=";", decimal=",", channel_columns='["R1V","R2V"]')
        x, _, _ = rio.load_acquisition(self.validated(row))
        np.testing.assert_allclose(x[:, 0], np.arange(16) + .25)
        with self.assertRaises(ValueError):
            rio.load_acquisition(self.validated(dict(row, decimal=".")))

    def test_duplicate_physical_csv_headers_are_not_mangled_silently(self):
        path = self.inputs / "duplicate.csv"
        path.write_text("R1V,R1V,R2V\n" + "1,2,3\n" * 16)
        for cols in ['["R1V","R2V"]', '[0,2]', '["R1V.1","R2V"]']:
            with self.subTest(cols=cols), self.assertRaisesRegex(ValueError, "duplicated"):
                rio.load_acquisition(self.validated(self.row(path, channel_columns=cols)))

    def test_csv_extra_fields_cannot_silently_shift_sensor_columns(self):
        path = self.inputs / "shifted.csv"
        path.write_text("R1V,R2V\n" + "0,1,2\n" * 16)
        with self.assertRaises((ValueError, pd.errors.ParserWarning)):
            rio.load_acquisition(self.validated(self.row(path, channel_columns='["R1V","R2V"]')))

    def test_no_time_column_as_signal(self):
        with self.assertRaisesRegex(ValueError, "time_column"):
            rio.load_acquisition(self.validated(self.row(self.csv(), time_column="R1V")))

    def test_missing_fs_and_bad_selector_types_fail_closed(self):
        path = self.csv()
        for changes in [{"fs_hz": ""}, {"channel_columns": "[0,1.0]"},
                        {"channel_columns": "[true,1]"}, {"channel_keys": "[1,2]"},
                        {"channel_columns": '["R1V",null]'}]:
            with self.subTest(changes=changes), self.assertRaises(rio.ManifestValidationError):
                self.validated(self.row(path, **changes))

    def test_path_escape_symlink_and_home_input_rejected(self):
        outside = self.base / "outside.csv"
        outside.write_text("a,b\n1,2\n")
        link = self.inputs / "escape.csv"
        link.symlink_to(outside)
        for path in [outside, link, "../outside.csv"]:
            with self.subTest(path=path), self.assertRaises(rio.ManifestValidationError):
                self.validated(self.row(path))
        with self.assertRaises(ValueError):
            rio._rio_root_for("z24_ems", {"z24": Path("/home")})

    def test_inventory_excludes_external_links_and_preserves_sources(self):
        source = self.csv()
        outside = self.base / "outside.csv"
        outside.write_text("x,y\n1,2\n")
        (self.inputs / "external.csv").symlink_to(outside)
        before = source.read_bytes()
        out = self.base / "derived"
        inventory = rio.inventory_inputs({"z24": self.inputs}, out)
        self.assertEqual(inventory.path.tolist(), [str(source)])
        self.assertEqual(inventory.attrs["excluded_link_count"], 1)
        self.assertTrue(json.loads((out / "excluded_input_links.json").read_text()))
        self.assertEqual(source.read_bytes(), before)
        with self.assertRaises(ValueError):
            rio.inventory_inputs({"z24": self.inputs}, self.inputs / "results")

    def test_hardlinks_cannot_duplicate_one_digital_record(self):
        source = self.csv()
        linked = self.inputs / "same_bytes_hardlink.csv"
        os.link(source, linked)
        a = self.row(source)
        b = self.row(linked, record_id="record_2", acquisition_id="acq_2")
        with self.assertRaises(rio.ManifestValidationError) as exc:
            rio.validate_manifest(pd.DataFrame([a, b]), self.roots)
        self.assertIn("selectors", exc.exception.errors.field.tolist())

    def test_mat_vector_struct_selectors_and_orientation(self):
        path = self.inputs / "data.mat"
        savemat(path, {"record": {"R1V": self.signal[:, 0], "R2V": self.signal[:, 1]},
                       "matrix": self.signal.T})
        row = self.row(path, format="mat", channel_columns="[]", channel_keys='["record.R1V","record/R2V"]')
        x, _, names = rio.load_acquisition(self.validated(row))
        np.testing.assert_allclose(x, self.signal[:, :2])
        self.assertEqual(names, ["record.R1V", "record/R2V"])
        matrix = self.row(path, format="mat", data_key="matrix", channel_columns="[2,0]", sample_axis=1,
                          channel_names='["R3V","R1V"]')
        x, _, names = rio.load_acquisition(self.validated(matrix))
        np.testing.assert_allclose(x, self.signal[:, [2, 0]])
        self.assertEqual(names, ["R3V", "R1V"])

    def test_pdt_matrix_requires_physical_channel_labels(self):
        path = self.inputs / "data.npz"
        np.savez(path, data=self.signal)
        row = self.row(path, dataset="z24_pdt", scenario="01", setup_id="A", excitation="avt",
                       format="npz", data_key="data", channel_columns="[0,1]")
        with self.assertRaises(rio.ManifestValidationError):
            self.validated(row)
        with self.assertRaisesRegex(ValueError, "physical channel_names"):
            rio.load_acquisition(row)
        good = self.validated(dict(row, channel_names='["R1V","R2V"]'))
        x, _, names = rio.load_acquisition(good)
        np.testing.assert_allclose(x, self.signal[:, :2])
        self.assertEqual(names, ["R1V", "R2V"])
        for names in ['["R1V","R1V"]', '["R1V"]', '["R1V","R2V","R3V"]']:
            with self.subTest(names=names), self.assertRaises(rio.ManifestValidationError):
                self.validated(dict(row, channel_names=names))

    def test_pdt_unknown_excitation_is_not_silently_included(self):
        with self.assertRaises(rio.ManifestValidationError):
            self.validated(self.row(self.csv(), dataset="z24_pdt", scenario="01", setup_id="A", excitation="unknown"))

    def test_npz_pickle_objects_and_nans_are_rejected(self):
        path = self.inputs / "object.npz"
        np.savez(path, data=np.array([{"a": 1}], dtype=object))
        row = self.row(path, format="npz", data_key="data", channel_columns="[]")
        self.assertFalse(rio.probe_input(path)["variables"][0]["usable"])
        with self.assertRaises(ValueError):
            rio.load_acquisition(self.validated(row))
        bad = self.signal.copy()
        bad[3, 1] = np.nan
        np.savez(path, data=bad)
        with self.assertRaisesRegex(ValueError, "NaN/Inf"):
            rio.load_acquisition(self.validated(row))

    def test_changing_raw_input_during_load_is_detected(self):
        path = self.csv()
        row = self.validated(self.row(path))
        original = pd.read_csv
        def mutate_after_read(*args, **kwargs):
            frame = original(*args, **kwargs)
            path.write_text(path.read_text() + "\n")
            return frame
        with patch.object(rio.pd, "read_csv", side_effect=mutate_after_read):
            with self.assertRaisesRegex(ValueError, "changed during"):
                rio.load_acquisition(row)

    def test_verified_openlab_source_is_narrow_and_exact(self):
        folder = self.inputs / "openlab_reference/2024-02-01_2024-10-31_ida_ki_export/01_acceleration_trigger"
        folder.mkdir(parents=True)
        path = folder / "acc_2024_05_02_T06_16_01.csv"
        frame = pd.DataFrame(np.arange(80).reshape(16, 5), columns=rio.OPENLAB_2024_VERTICAL_CHANNELS)
        frame.to_csv(path, index=False)
        inventory = rio.inventory_inputs({"openlab": self.inputs}, self.base / "out")
        template = rio.make_manifest_template(inventory)
        self.assertEqual(template.iloc[0].dataset, "openlab_2024")
        self.assertEqual(template.iloc[0].timestamp_utc, "2024-05-02T06:16:01+00:00")
        self.assertEqual(template.iloc[0].state_label, "reference")
        x, fs, names = rio.load_acquisition(self.validated(template.iloc[0].to_dict()))
        self.assertEqual((x.shape, fs, names), ((16, 5), 500., rio.OPENLAB_2024_VERTICAL_CHANNELS))
        for name in ["acc_2025_05_02_T06_16_01.csv", "acc_2024_01_01_T06_16_01.csv"]:
            candidate = folder / name
            frame.to_csv(candidate, index=False)
            self.assertEqual(rio._rio_known_openlab_2024(candidate, str(candidate.relative_to(self.inputs))), {})
        missing = frame.iloc[:, :4]
        missing.to_csv(path, index=False)
        self.assertEqual(rio._rio_known_openlab_2024(path, str(path.relative_to(self.inputs))), {})

    @unittest.skipUnless(importlib.util.find_spec("h5py"), "optional h5py not installed")
    def test_hdf5_local_numeric_dataset_and_mat_v73(self):
        import h5py
        for suffix in [".h5", ".mat"]:
            path = self.inputs / ("data" + suffix)
            with h5py.File(path, "w") as f:
                f.create_dataset("signals/reference", data=self.signal)
            row = self.row(path, format=suffix[1:], data_key="signals/reference", channel_columns="[0,2]")
            x, _, _ = rio.load_acquisition(self.validated(row))
            np.testing.assert_allclose(x, self.signal[:, [0, 2]])

    @unittest.skipUnless(importlib.util.find_spec("h5py"), "optional h5py not installed")
    def test_hdf5_external_links_and_virtual_datasets_are_rejected(self):
        import h5py
        external = self.base / "outside.h5"
        with h5py.File(external, "w") as f:
            f.create_dataset("raw", data=self.signal)
        path = self.inputs / "links.h5"
        with h5py.File(path, "w", libver="latest") as f:
            f["external"] = h5py.ExternalLink(str(external), "/raw")
            layout = h5py.VirtualLayout(shape=self.signal.shape, dtype="f8")
            layout[:] = h5py.VirtualSource(str(external), "raw", shape=self.signal.shape)
            f.create_virtual_dataset("virtual", layout)
        for key in ["external", "virtual"]:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "external|virtual"):
                rio.load_acquisition(self.validated(self.row(path, format="h5", data_key=key, channel_columns="[]")))


class VerifiedPDTDiscovery(unittest.TestCase):
    """Test schema discovery using stored physical labels, never fixed positions."""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.raw = self.base / "z24"
        self.raw.mkdir()
        self.references = list(rio.Z24_PDT_REFERENCE_CHANNELS)

    def tearDown(self):
        self.tmp.cleanup()

    def source(self, scenarios=(1,), excitations=("avt",), setup_range=range(1, 10), mutate=None):
        for scenario in scenarios:
            for excitation in excitations:
                folder = self.raw / f"{scenario:02d}" / excitation
                folder.mkdir(parents=True, exist_ok=True)
                for setup in setup_range:
                    labels = self.references + [f"ROVING_{setup:02d}"]
                    if excitation.lower() == "fvt":
                        labels += ["SHAKER_FORCE"]
                    labels = list(np.roll(labels, setup % len(labels)))
                    if mutate:
                        labels = mutate(scenario, excitation, setup, labels)
                    values = np.column_stack([np.arange(16) + (self.references.index(label) * 100 if label in self.references else 999)
                                              for label in labels])
                    path = folder / f"{scenario:02d}setup{setup:02d}.mat"
                    savemat(path, {"data": values, "labelshulp": np.array([name + " " for name in labels])})

    def inventory(self):
        return rio.inventory_inputs({"z24": self.raw}, self.base / "diagnostics")

    def test_all_306_pdt_rows_have_proven_individual_reference_mapping(self):
        self.source(scenarios=range(1, 18), excitations=("avt", "fvt"))
        template = rio.make_manifest_template(self.inventory())
        audit = template.attrs["z24_pdt_schema_audit"]
        self.assertEqual(audit["status"], "VERIFIED_ALL_PDT_CANDIDATES")
        self.assertEqual(audit["verified_acquisitions"], 306)
        self.assertFalse(audit["signal_matrices_loaded"])
        self.assertEqual(set(audit["source01_avt_common_reference_labels"]), set(self.references))
        rows = rio.validate_manifest(template, {"z24": self.raw})
        self.assertEqual(len(rows), 306)
        self.assertEqual(rows.dataset.unique().tolist(), ["z24_pdt"])
        self.assertEqual(set(rows.excitation), {"AVT", "FVT"})
        self.assertEqual(set(rows[rows.state_label.eq("reference")].scenario), {"01", "02", "08"})
        self.assertEqual(set(rows[rows.state_label.eq("altered")].scenario), set(f"{n:02d}" for n in range(3,18)) - {"08"})
        selected = rows[(rows.scenario.eq("17")) & rows.excitation.eq("FVT")].iloc[7]
        x, fs, names = rio.load_acquisition(selected)
        self.assertEqual((fs, names), (100., self.references))
        np.testing.assert_array_equal(x, np.arange(16)[:, None] + np.arange(5)[None, :] * 100)
        self.assertGreater(len(set(rows.channel_columns)), 1)

    def test_metadata_discovery_never_requests_vibration_matrix(self):
        self.source()
        from scipy.io import loadmat
        requested = []
        def metadata_only(*args, **kwargs):
            requested.append(kwargs.get("variable_names"))
            self.assertNotIn("data", kwargs.get("variable_names", []))
            return loadmat(*args, **kwargs)
        with patch("scipy.io.loadmat", side_effect=metadata_only):
            template = rio.make_manifest_template(self.inventory())
        self.assertTrue(requested)
        self.assertEqual(template.attrs["z24_pdt_schema_audit"]["status"], "VERIFIED_ALL_PDT_CANDIDATES")

    def test_absent_or_ambiguous_source_setups_block_pdt_autodiscovery(self):
        self.source(setup_range=range(1, 9))
        template = rio.make_manifest_template(self.inventory())
        self.assertTrue(template.dataset.eq("").all())
        self.assertEqual(template.attrs["z24_pdt_schema_audit"]["status"], "BLOCKED_SOURCE01_NINE_AVT_LABEL_SCHEMAS_REQUIRED")

    def test_six_common_reference_components_are_not_arbitrarily_reduced_to_five(self):
        self.source(mutate=lambda s,e,n,labels: labels + ["R1L"])
        template = rio.make_manifest_template(self.inventory())
        self.assertTrue(template.dataset.eq("").all())
        self.assertEqual(template.attrs["z24_pdt_schema_audit"]["status"], "BLOCKED_REFERENCE_CHANNEL_INTERSECTION_AMBIGUOUS")

    def test_a_single_missing_later_physical_component_blocks_whole_auto_cohort(self):
        def mutate(scenario, excitation, setup, labels):
            return ["UNRESOLVED" if label == "R2T" else label for label in labels] if scenario == 9 and setup == 3 else labels
        self.source(scenarios=(1, 9), excitations=("avt", "fvt"), mutate=mutate)
        template = rio.make_manifest_template(self.inventory())
        audit = template.attrs["z24_pdt_schema_audit"]
        self.assertEqual(audit["status"], "BLOCKED_PDT_COHORT_HAS_UNVERIFIED_SCHEMAS")
        self.assertEqual(audit["automatic_manifest_rows"], 0)
        self.assertTrue(template.dataset.eq("").all())
        self.assertTrue(any(record["status"].startswith("BLOCKED_REFERENCE") for record in audit["records"]))

    def test_mismatched_label_count_or_documented_rate_is_not_accepted(self):
        self.source()
        path = self.raw / "01/avt/01setup09.mat"
        for changes in [{"labelshulp": np.array(self.references)}, {"fs": 200.}]:
            original = {"data": np.ones((16, 6)), "labelshulp": np.array(self.references + ["ROVING_09"])}
            savemat(path, {**original, **changes})
            template = rio.make_manifest_template(self.inventory())
            self.assertEqual(template.attrs["z24_pdt_schema_audit"]["status"], "BLOCKED_SOURCE01_NINE_AVT_LABEL_SCHEMAS_REQUIRED")

    def test_mat_label_strings_and_characters_are_decoded_without_numeric_names(self):
        self.assertEqual(rio._rio_mat_labels(np.array([b"R1V ", b"R2L "])), ["R1V", "R2L"])
        self.assertEqual(rio._rio_mat_labels(np.array([list("R1V "), list("R2L ")])), ["R1V", "R2L"])
        with self.assertRaisesRegex(ValueError, "not numeric"):
            rio._rio_mat_labels(np.arange(5))
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            rio._rio_mat_labels(np.array([["R1V", "R2L"], ["R2T", "R2V"]]))

    def test_loading_rechecks_discovered_physical_labels_after_source_changes(self):
        self.source()
        template = rio.make_manifest_template(self.inventory())
        row = rio.validate_manifest(template, {"z24": self.raw}).iloc[0]
        path = Path(row.path)
        schema = rio._rio_pdt_mat_schema(path)
        labels = schema["channel_labels"]
        labels[0], labels[1] = labels[1], labels[0]
        savemat(path, {"data": np.ones((16,len(labels))), "labelshulp": np.array(labels)})
        with self.assertRaisesRegex(ValueError, "channel mapping"):
            rio.load_acquisition(row)

    def test_actual_path_convention_is_required(self):
        self.assertEqual(rio._rio_pdt_path_metadata("17/FVT/17setup09.mat"), {"scenario":"17", "excitation":"FVT", "setup_id":"09"})
        for path in ["01/avt/02setup01.mat", "01/unknown/01setup01.mat", "01/avt/01setup10.mat", "ems/01/avt/01setup01.mat"]:
            self.assertEqual(rio._rio_pdt_path_metadata(path), {})

    def test_diagnostics_cover_environment_and_aaa_without_creating_ems_rows(self):
        self.source()
        (self.raw / "01/avt/01/R1V.aaa").parent.mkdir()
        (self.raw / "01/avt/01/R1V.aaa").write_text("Header: one physical PDT channel\n1.0\n2.0\n")
        archive_path = self.raw / "01/avt.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("avt/01/R1V.aaa", "header\n1\n2\n")
        openlab = self.base / "openlab"
        environment = openlab / "openlab_reference/2024-02-01_2024-10-31_ida_ki_export/02_environment"
        environment.mkdir(parents=True)
        for n in range(9):
            (environment / f"environment_{n}.csv").write_text("timestamp_utc;measured_temperature\n2024-05-02T06:00:00Z;14.2\n")
        inventory = rio.inventory_inputs({"z24": self.raw, "openlab": openlab}, self.base / "mixed_diagnostics")
        probes = json.loads((self.base / "mixed_diagnostics/file_format_probes.json").read_text())
        self.assertEqual(set(p["suffix"] for p in probes), {".csv", ".mat", ".aaa", ".zip"})
        self.assertEqual(inventory.attrs["environmental_csv_probe_count"], 9)
        environment_probes = json.loads((self.base / "mixed_diagnostics/environmental_csv_probes.json").read_text())
        self.assertEqual(len(environment_probes), 9)
        self.assertTrue(all("timestamp_utc" in p["first_row_cells"] for p in environment_probes))
        zip_probe = next(p for p in probes if p["suffix"] == ".zip")
        self.assertFalse(zip_probe["archive_extracted"])
        self.assertEqual(zip_probe["archive_members"][0]["path"], "avt/01/R1V.aaa")
        template = rio.make_manifest_template(inventory)
        self.assertNotIn("z24_ems", template.dataset.tolist())
        self.assertEqual(len(template[template.dataset.eq("z24_pdt")]), 9)
        self.assertNotIn("aaa", template['format'].tolist())


if __name__ == "__main__":
    unittest.main()
