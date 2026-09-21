"""Convert Olink NPX parquet exports to SDRF affinity-proteomics format.

Reads long-format NPX data (one row per sample × assay) and emits one SDRF row
per unique sample, using the affinity-proteomics template (optionally combined
with an organism template such as ``human``).

Usage:
  python tools/NPXtoSDRF.py data.parquet -o PXD000000.sdrf.tsv
  python -m tools npx-to-sdrf data.parquet -o out.sdrf.tsv --template human

NPX columns consumed when present (case/spacing insensitive):
  SampleID, Sample Type, Panel, PlateID, WellID, Normalization, Panel_Lot_Nr

Sample-level metadata can be supplied with ``--sample-metadata`` (TSV keyed by
SampleID) or global defaults via ``--organism``, ``--sample-matrix``, etc.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATES_DIR = REPO_ROOT / "spec" / "sdrf-proteomics" / "sdrf-templates"

TECHNOLOGY_TYPE = "protein expression profiling by antibody array"
QUANTIFICATION_UNIT = "NPX"
SDRF_VERSION = "v1.1.0"
ANNOTATION_TOOL = "NT=NPXtoSDRF;VV=v0.1.0"

# Olink NPX / NPX Map sample-type strings → SDRF characteristics[sample type] labels.
SAMPLE_TYPE_MAP: dict[str, str] = {
    "SAMPLE": "study sample",
    "STUDY SAMPLE": "study sample",
    "PLATE_CONTROL": "plate control",
    "PLATE CONTROL": "plate control",
    "NEGATIVE_CONTROL": "negative control",
    "NEGATIVE CONTROL": "negative control",
    "CONTROL": "quality control sample",
    "POSITIVE_CONTROL": "positive control",
    "POSITIVE CONTROL": "positive control",
    "CALIBRATOR": "calibrator",
    "REFERENCE": "reference",
    "BRIDGE": "bridge",
}

# Substrings in Panel names → comment[platform] when --platform is not set.
PANEL_PLATFORM_HINTS: tuple[tuple[str, str], ...] = (
    ("explore ht", "Olink Explore HT"),
    ("explore 384", "Olink Explore 384"),
    ("explore 1536", "Olink Explore HT"),
    ("explore", "Olink Explore HT"),
    ("target 48", "Olink Target 48"),
    ("target 96", "Olink Target 96"),
    ("reveal", "Olink Reveal"),
    ("focus", "Olink Focus"),
)

# Canonical NPX column aliases after normalisation.
COLUMN_ALIASES: dict[str, str] = {
    "sampleid": "SampleID",
    "sampletype": "Sample Type",
    "panel": "Panel",
    "plateid": "PlateID",
    "wellid": "WellID",
    "normalization": "Normalization",
    "panel_lot_nr": "Panel_Lot_Nr",
    "panellotnr": "Panel_Lot_Nr",
    "npx": "NPX",
    "olinkid": "OlinkID",
}


@dataclass
class NpxSample:
    """Aggregated metadata for one Olink sample."""

    sample_id: str
    sample_type: str | None = None
    panel: str | None = None
    plate_id: str | None = None
    well_id: str | None = None
    normalization: str | None = None
    lot_number: str | None = None
    assay_count: int = 0


@dataclass
class ConversionDefaults:
    """Global defaults applied when sample metadata does not override."""

    organism: str = "not available"
    organism_part: str = "not available"
    disease: str = "not available"
    sample_matrix: str = "not available"
    age: str = "not available"
    sex: str = "not available"
    platform: str | None = None
    data_file: str | None = None


@dataclass
class ConversionReport:
    """Summary of an NPX → SDRF conversion."""

    npx_path: str
    n_samples: int = 0
    n_assay_rows: int = 0
    panels: list[str] = field(default_factory=list)
    plates: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    templates: list[str] = field(default_factory=list)
    output_columns: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"NPX file: {self.npx_path}",
            f"Assay rows read: {self.n_assay_rows}",
            f"SDRF sample rows: {self.n_samples}",
            f"Templates: {', '.join(self.templates) or 'affinity-proteomics'}",
            f"Columns: {len(self.output_columns)}",
        ]
        if self.panels:
            lines.append(f"Panels: {', '.join(self.panels)}")
        if self.plates:
            lines.append(f"Plates: {', '.join(self.plates)}")
        if self.warnings:
            lines.append("Warnings:")
            lines.extend(f"  - {w}" for w in self.warnings)
        return "\n".join(lines)


def _normalize_column_name(name: str) -> str:
    key = re.sub(r"[\s_]+", "", name.strip().lower())
    return COLUMN_ALIASES.get(key, name.strip())


def _first_present(row: Mapping[str, Any], *keys: str) -> Any | None:
    for key in keys:
        if key in row:
            value = row[key]
            if value is not None and str(value).strip() != "":
                return value
    return None


def _normalize_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalised: list[dict[str, Any]] = []
    for row in rows:
        mapped: dict[str, Any] = {}
        for raw_key, value in row.items():
            mapped[_normalize_column_name(str(raw_key))] = value
        normalised.append(mapped)
    return normalised


def read_npx_parquet(path: str | Path) -> list[dict[str, Any]]:
    """Read an Olink NPX parquet file into normalised row dicts."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment specific
        raise ImportError(
            "pyarrow is required to read NPX parquet files. "
            "Install with: pip install pyarrow"
        ) from exc

    table = pq.read_table(str(path))
    rows = table.to_pylist()
    return _normalize_rows(rows)


def aggregate_samples(rows: Sequence[Mapping[str, Any]]) -> list[NpxSample]:
    """Collapse long-format NPX rows to one record per SampleID."""
    by_id: dict[str, NpxSample] = {}

    for row in rows:
        sample_id_raw = _first_present(row, "SampleID")
        if sample_id_raw is None:
            continue
        sample_id = str(sample_id_raw).strip()
        if not sample_id:
            continue

        sample = by_id.get(sample_id)
        if sample is None:
            sample = NpxSample(sample_id=sample_id)
            by_id[sample_id] = sample

        sample.assay_count += 1

        sample_type = _first_present(row, "Sample Type")
        if sample_type is not None and sample.sample_type is None:
            sample.sample_type = str(sample_type).strip()

        panel = _first_present(row, "Panel")
        if panel is not None and sample.panel is None:
            sample.panel = str(panel).strip()

        plate_id = _first_present(row, "PlateID")
        if plate_id is not None and sample.plate_id is None:
            sample.plate_id = str(plate_id).strip()

        well_id = _first_present(row, "WellID")
        if well_id is not None and sample.well_id is None:
            sample.well_id = str(well_id).strip()

        normalization = _first_present(row, "Normalization")
        if normalization is not None and sample.normalization is None:
            sample.normalization = str(normalization).strip()

        lot_number = _first_present(row, "Panel_Lot_Nr")
        if lot_number is not None and sample.lot_number is None:
            sample.lot_number = str(lot_number).strip()

    return [by_id[k] for k in sorted(by_id)]


def infer_platform(panel: str | None) -> str | None:
    if not panel:
        return None
    lowered = panel.lower()
    for needle, platform in PANEL_PLATFORM_HINTS:
        if needle in lowered:
            return platform
    return None


def map_sample_type(raw: str | None, *, is_control: bool = False) -> str:
    if not raw:
        return "plate control" if is_control else "study sample"
    key = re.sub(r"[\s_]+", " ", raw.strip().upper())
    return SAMPLE_TYPE_MAP.get(key, "study sample")


def _is_control_sample_type(label: str) -> bool:
    return label not in {"study sample"}


def load_sample_metadata(path: str | Path) -> dict[str, dict[str, str]]:
    """Load per-sample overrides from a TSV (first column must be SampleID)."""
    metadata: dict[str, dict[str, str]] = {}
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            return metadata
        id_col = reader.fieldnames[0]
        for row in reader:
            sample_id = (row.get(id_col) or "").strip()
            if not sample_id:
                continue
            overrides = {
                k: v.strip()
                for k, v in row.items()
                if k and k != id_col and v and v.strip()
            }
            metadata[sample_id] = overrides
            metadata[sample_id.lower()] = overrides
    return metadata


def _resolve_templates_dir(templates_dir: Path | None) -> Path:
    path = templates_dir or TEMPLATES_DIR
    if not path.is_dir():
        raise FileNotFoundError(
            f"SDRF templates directory not found: {path}. "
            "Run: git submodule update --init --recursive"
        )
    return path


def merged_template_columns(
    template_names: Sequence[str],
    *,
    templates_dir: Path | None = None,
) -> list[str]:
    """Return ordered column names for one or more SDRF templates."""
    sys.path.insert(0, str(REPO_ROOT / "spec" / "scripts"))
    from resolve_templates import load_manifest, resolve_template

    td = _resolve_templates_dir(templates_dir)
    manifest = load_manifest(td)
    order: list[str] = []
    seen: set[str] = set()
    for name in template_names:
        resolved = resolve_template(name, td, manifest)
        for col in resolved["all_columns"]:
            col_name = col["name"]
            if col_name not in seen:
                order.append(col_name)
                seen.add(col_name)
    return order


def _template_comment(name: str, version: str | None = None) -> str:
    if version is None:
        sys.path.insert(0, str(REPO_ROOT / "spec" / "scripts"))
        from resolve_templates import load_manifest, resolve_template

        manifest = load_manifest(_resolve_templates_dir(None))
        version = manifest[name]["latest"]
    return f"NT={name};VV=v{version}"


def _build_row(
    sample: NpxSample,
    *,
    defaults: ConversionDefaults,
    metadata: Mapping[str, str],
    templates: Sequence[str],
    data_file: str,
    bio_rep: int,
) -> dict[str, str]:
    sample_type_label = metadata.get("characteristics[sample type]") or map_sample_type(
        sample.sample_type,
    )
    is_control = _is_control_sample_type(sample_type_label)

    platform = (
        metadata.get("comment[platform]")
        or defaults.platform
        or infer_platform(sample.panel)
        or "not available"
    )

    assay_suffix = sample.plate_id or "1"
    assay_name = metadata.get("assay name") or f"{sample.sample_id}_plate_{assay_suffix}"

    row: dict[str, str] = {
        "source name": metadata.get("source name") or sample.sample_id,
        "assay name": assay_name,
        "technology type": TECHNOLOGY_TYPE,
        "comment[technical replicate]": metadata.get("comment[technical replicate]", "1"),
        "comment[data file]": metadata.get("comment[data file]") or data_file,
        "comment[sdrf version]": SDRF_VERSION,
        "comment[sdrf annotation tool]": ANNOTATION_TOOL,
        "characteristics[organism]": metadata.get(
            "characteristics[organism]",
            "not applicable" if is_control else defaults.organism,
        ),
        "characteristics[organism part]": metadata.get(
            "characteristics[organism part]",
            "not applicable" if is_control else defaults.organism_part,
        ),
        "characteristics[biological replicate]": metadata.get(
            "characteristics[biological replicate]",
            str(bio_rep),
        ),
        "characteristics[sample type]": sample_type_label,
        "characteristics[disease]": metadata.get(
            "characteristics[disease]",
            "not applicable" if is_control else defaults.disease,
        ),
        "comment[platform]": platform,
        "comment[quantification unit]": metadata.get(
            "comment[quantification unit]",
            QUANTIFICATION_UNIT,
        ),
    }

    if sample.panel:
        row["comment[panel name]"] = metadata.get("comment[panel name]", sample.panel)
    if sample.plate_id:
        row["comment[plate]"] = metadata.get("comment[plate]", sample.plate_id)
    if sample.normalization:
        row["comment[normalization method]"] = metadata.get(
            "comment[normalization method]",
            sample.normalization,
        )
    if sample.lot_number:
        row["comment[lot number]"] = metadata.get("comment[lot number]", sample.lot_number)

    matrix = metadata.get("characteristics[sample matrix]")
    if matrix:
        row["characteristics[sample matrix]"] = matrix
    elif defaults.sample_matrix != "not available" and not is_control:
        row["characteristics[sample matrix]"] = defaults.sample_matrix

    if "human" in templates:
        row["characteristics[age]"] = metadata.get(
            "characteristics[age]",
            "not applicable" if is_control else defaults.age,
        )
        row["characteristics[sex]"] = metadata.get(
            "characteristics[sex]",
            "not applicable" if is_control else defaults.sex,
        )

    for key, value in metadata.items():
        if key not in row and value:
            row[key] = value

    return row


def render_sdrf(
    rows: Sequence[Mapping[str, str]],
    *,
    templates: Sequence[str],
    templates_dir: Path | None = None,
) -> tuple[str, list[str]]:
    """Render SDRF TSV text and return (content, column_order)."""
    template_names = ["affinity-proteomics", *templates]
    all_columns = merged_template_columns(template_names, templates_dir=templates_dir)

    used_keys: set[str] = set()
    for row in rows:
        used_keys.update(row.keys())
    used_keys.update({"comment[sdrf version]", "comment[sdrf annotation tool]"})

    template_values = [_template_comment(name) for name in template_names]

    # Walk template column order; expand comment[sdrf template] to one column per template.
    logical_columns: list[str | tuple[str, int]] = []
    for col in all_columns:
        if col == "comment[sdrf template]":
            for idx in range(len(template_names)):
                logical_columns.append((col, idx))
        elif col in used_keys:
            logical_columns.append(col)

    extra_cols = sorted(used_keys - {c for c in all_columns if c != "comment[sdrf template]"})
    logical_columns.extend(extra_cols)

    out_columns: list[str] = []
    for item in logical_columns:
        if isinstance(item, tuple):
            out_columns.append(item[0])
        else:
            out_columns.append(item)

    out_rows: list[list[str]] = []
    for row in rows:
        values: list[str] = []
        for item in logical_columns:
            if isinstance(item, tuple):
                _, idx = item
                values.append(template_values[idx])
            else:
                values.append(row.get(item, ""))
        out_rows.append(values)

    lines = ["\t".join(out_columns)]
    lines.extend("\t".join(values) for values in out_rows)
    return "\n".join(lines) + "\n", out_columns


def convert_npx_to_sdrf(
    npx_path: str | Path,
    *,
    templates: Sequence[str] = ("human",),
    defaults: ConversionDefaults | None = None,
    sample_metadata: Mapping[str, Mapping[str, str]] | None = None,
    templates_dir: Path | None = None,
) -> tuple[str, ConversionReport]:
    """Convert an Olink NPX parquet file to SDRF TSV text."""
    path = Path(npx_path)
    defaults = defaults or ConversionDefaults()
    if defaults.data_file is None:
        defaults.data_file = path.name

    rows = read_npx_parquet(path)
    if not rows:
        raise ValueError(f"No rows found in NPX file: {path}")

    if "SampleID" not in rows[0]:
        raise ValueError(
            "NPX file is missing a SampleID column. "
            f"Columns found: {', '.join(sorted(rows[0]))}"
        )

    samples = aggregate_samples(rows)
    if not samples:
        raise ValueError(f"No samples with SampleID found in {path}")

    metadata_lookup = sample_metadata or {}
    sdrf_rows: list[dict[str, str]] = []
    bio_rep_counter = 0
    warnings: list[str] = []

    for sample in samples:
        meta = metadata_lookup.get(sample.sample_id) or metadata_lookup.get(
            sample.sample_id.lower(), {},
        )
        sample_type_label = meta.get("characteristics[sample type]") or map_sample_type(
            sample.sample_type,
        )
        if not _is_control_sample_type(sample_type_label):
            bio_rep_counter += 1
            bio_rep = int(meta.get("characteristics[biological replicate]", bio_rep_counter))
        else:
            bio_rep = int(meta.get("characteristics[biological replicate]", 1))

        platform = (
            meta.get("comment[platform]")
            or defaults.platform
            or infer_platform(sample.panel)
        )
        if platform is None:
            warnings.append(
                f"Sample {sample.sample_id}: could not infer comment[platform] "
                f"from panel {sample.panel!r}; using 'not available'"
            )

        sdrf_rows.append(
            _build_row(
                sample,
                defaults=defaults,
                metadata=meta,
                templates=templates,
                data_file=defaults.data_file or path.name,
                bio_rep=bio_rep,
            )
        )

    template_names = ["affinity-proteomics", *templates]
    content, columns = render_sdrf(
        sdrf_rows,
        templates=templates,
        templates_dir=templates_dir,
    )

    report = ConversionReport(
        npx_path=str(path),
        n_samples=len(sdrf_rows),
        n_assay_rows=len(rows),
        panels=sorted({s.panel for s in samples if s.panel}),
        plates=sorted({s.plate_id for s in samples if s.plate_id}),
        warnings=warnings,
        templates=template_names,
        output_columns=columns,
    )
    return content, report


def run_cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="NPXtoSDRF",
        description="Convert Olink NPX parquet exports to SDRF affinity-proteomics format",
    )
    parser.add_argument("npx_file", help="Path to Olink NPX parquet file")
    parser.add_argument("-o", "--output", help="Output SDRF TSV path (default: stdout)")
    parser.add_argument(
        "--template",
        action="append",
        default=["human"],
        help="Organism/add-on template to combine with affinity-proteomics (repeatable). "
        "Use --template none to emit technology columns only.",
    )
    parser.add_argument("--organism", default="not available")
    parser.add_argument("--organism-part", default="not available")
    parser.add_argument("--disease", default="not available")
    parser.add_argument("--sample-matrix", default="not available")
    parser.add_argument("--age", default="not available")
    parser.add_argument("--sex", default="not available")
    parser.add_argument(
        "--platform",
        help="Olink platform (e.g. 'Olink Explore HT'). Inferred from Panel when omitted.",
    )
    parser.add_argument(
        "--data-file",
        help="Value for comment[data file] (default: NPX parquet basename)",
    )
    parser.add_argument(
        "--sample-metadata",
        help="TSV with SampleID plus SDRF column overrides (tab-separated)",
    )
    parser.add_argument(
        "--templates-dir",
        type=Path,
        default=None,
        help="Override path to spec/sdrf-proteomics/sdrf-templates",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress summary on stderr")

    args = parser.parse_args(list(argv) if argv is not None else None)

    templates = [t for t in args.template if t.lower() != "none"]
    defaults = ConversionDefaults(
        organism=args.organism,
        organism_part=args.organism_part,
        disease=args.disease,
        sample_matrix=args.sample_matrix,
        age=args.age,
        sex=args.sex,
        platform=args.platform,
        data_file=args.data_file,
    )
    metadata = load_sample_metadata(args.sample_metadata) if args.sample_metadata else None

    try:
        content, report = convert_npx_to_sdrf(
            args.npx_file,
            templates=templates,
            defaults=defaults,
            sample_metadata=metadata,
            templates_dir=args.templates_dir,
        )
    except (FileNotFoundError, ImportError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.output:
        Path(args.output).write_text(content, encoding="utf-8")
        if not args.quiet:
            print(f"Wrote {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(content)

    if not args.quiet:
        print(report.summary(), file=sys.stderr)

    return 1 if report.warnings else 0


if __name__ == "__main__":
    raise SystemExit(run_cli())
