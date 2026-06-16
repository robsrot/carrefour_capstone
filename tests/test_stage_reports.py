from pathlib import Path

from src.config import PipelineConfig
from src.stage_reports import compact_stage_report_markdown, stage_report_path, write_stage_report


def _test_config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        values={"paths": {"outputs": "outputs"}},
        mode="dev",
        root=tmp_path,
    )


def test_stage_report_path_is_single_root_report_file(tmp_path: Path) -> None:
    cfg = _test_config(tmp_path)

    assert stage_report_path("1", cfg=cfg) == tmp_path / "outputs" / "dev" / "reports" / "stage_01.md"
    assert stage_report_path("00_5", cfg=cfg).name == "stage_00_5.md"


def test_write_stage_report_keeps_notebook_report_compact(tmp_path: Path) -> None:
    cfg = _test_config(tmp_path)

    report_path = write_stage_report(
        "04",
        "Customer Embeddings Coverage and Dominance Gates",
        summary=["Coverage gates passed.", "Dominance gates passed."],
        metrics={"line_coverage_pct": 99.5, "top_product_weight_share_pct": 4.2},
        figures={"Customer embedding diagnostics": tmp_path / "stage4.png"},
        artifacts={"Customer embeddings": tmp_path / "customer_embeddings.parquet"},
        cfg=cfg,
    )

    report = report_path.read_text(encoding="utf-8")
    assert "# Stage 04: Customer Embeddings Coverage and Dominance Gates" in report
    assert "| line_coverage_pct | 99.5 |" in report
    assert "## Visual Evidence" in report
    assert "stage4.png" in report
    assert report_path.parent == cfg.reports


def test_compact_stage_report_markdown_clips_long_reports(tmp_path: Path) -> None:
    report_path = tmp_path / "stage_99.md"
    report_path.write_text("\n".join(f"line {idx}" for idx in range(10)), encoding="utf-8")

    markdown = compact_stage_report_markdown(report_path, max_lines=3)

    assert "line 0" in markdown
    assert "line 4" not in markdown
    assert "Showing first 3 lines" in markdown
