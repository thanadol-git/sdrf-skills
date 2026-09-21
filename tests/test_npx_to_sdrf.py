"""Tests for tools.NPXtoSDRF."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.NPXtoSDRF import (
    ConversionDefaults,
    aggregate_samples,
    convert_npx_to_sdrf,
    infer_platform,
    map_sample_type,
    read_npx_parquet,
    render_sdrf,
)


@pytest.fixture
def sample_npx_rows() -> list[dict]:
    return [
        {
            "SampleID": "S001",
            "Sample Type": "SAMPLE",
            "Panel": "Explore HT",
            "PlateID": "Plate1",
            "WellID": "A01",
            "Normalization": "PCNormalizedNPX",
            "Panel_Lot_Nr": "LOT-123",
            "OlinkID": "OID1",
            "NPX": 1.2,
        },
        {
            "SampleID": "S001",
            "Sample Type": "SAMPLE",
            "Panel": "Explore HT",
            "PlateID": "Plate1",
            "WellID": "A01",
            "Normalization": "PCNormalizedNPX",
            "Panel_Lot_Nr": "LOT-123",
            "OlinkID": "OID2",
            "NPX": 2.3,
        },
        {
            "SampleID": "NEG1",
            "Sample Type": "NEGATIVE_CONTROL",
            "Panel": "Explore HT",
            "PlateID": "Plate1",
            "WellID": "H12",
            "Normalization": "PCNormalizedNPX",
            "OlinkID": "OID1",
            "NPX": 0.1,
        },
    ]


class TestNpxHelpers:
    def test_aggregate_samples(self, sample_npx_rows: list[dict]):
        samples = aggregate_samples(sample_npx_rows)
        assert len(samples) == 2
        by_id = {s.sample_id: s for s in samples}
        assert by_id["S001"].assay_count == 2
        assert by_id["S001"].panel == "Explore HT"
        assert by_id["NEG1"].sample_type == "NEGATIVE_CONTROL"

    def test_map_sample_type(self):
        assert map_sample_type("SAMPLE") == "study sample"
        assert map_sample_type("NEGATIVE_CONTROL") == "negative control"
        assert map_sample_type(None) == "study sample"

    def test_infer_platform(self):
        assert infer_platform("Explore HT") == "Olink Explore HT"
        assert infer_platform("Target 96 Inflammation") == "Olink Target 96"
        assert infer_platform("Unknown panel") is None


class TestRenderSdrf:
    def test_render_includes_template_columns(self, sample_npx_rows: list[dict]):
        samples = aggregate_samples(sample_npx_rows)
        defaults = ConversionDefaults(
            organism="homo sapiens",
            organism_part="blood",
            disease="normal",
            sample_matrix="plasma",
            platform="Olink Explore HT",
            data_file="demo.parquet",
        )
        rows = []
        for idx, sample in enumerate(samples, start=1):
            from tools.NPXtoSDRF import _build_row

            rows.append(
                _build_row(
                    sample,
                    defaults=defaults,
                    metadata={},
                    templates=("human",),
                    data_file="demo.parquet",
                    bio_rep=idx,
                )
            )

        content, columns = render_sdrf(rows, templates=("human",))
        assert "source name\t" in content
        assert "comment[platform]" in columns
        assert content.count("comment[sdrf template]") >= 2
        assert "protein expression profiling by antibody array" in content
        assert "S001" in content
        assert "negative control" in content


class TestParquetIntegration:
    @pytest.fixture
    def parquet_file(self, tmp_path: Path, sample_npx_rows: list[dict]):
        pa = pytest.importorskip("pyarrow")
        pq = pytest.importorskip("pyarrow.parquet")
        table = pa.Table.from_pylist(sample_npx_rows)
        path = tmp_path / "demo_npx.parquet"
        pq.write_table(table, path)
        return path

    def test_read_npx_parquet(self, parquet_file: Path):
        rows = read_npx_parquet(parquet_file)
        assert len(rows) == 3
        assert rows[0]["SampleID"] == "S001"

    def test_convert_npx_to_sdrf(self, parquet_file: Path):
        defaults = ConversionDefaults(
            organism="homo sapiens",
            organism_part="blood",
            disease="normal",
            sample_matrix="plasma",
        )
        content, report = convert_npx_to_sdrf(
            parquet_file,
            templates=("human",),
            defaults=defaults,
        )
        assert report.n_samples == 2
        assert report.n_assay_rows == 3
        assert "affinity-proteomics" in ",".join(report.templates)
        assert "S001" in content
        assert "NPX" in content
