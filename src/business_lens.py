"""Business-facing opportunity lens for the final customer tribes."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

import polars as pl

from src.config import CONFIG, PipelineConfig
from src.progress import log_event


def write_business_lens_artifacts(
    *,
    core_summary_path: str | Path,
    mission_summary_path: str | Path,
    profile_path: str | Path,
    output_dir: str | Path | None = None,
    cfg: PipelineConfig = CONFIG,
) -> dict[str, Path]:
    """Write the Stage 10 business lens artifacts.

    This layer does not feed back into modeling. It translates the selected
    product-first tribes into CRM, campaign, and opportunity-sizing language.
    """

    out_dir = Path(output_dir) if output_dir else cfg.reports / str(cfg.get("exports.presentation_dir", "presentation"))
    out_dir.mkdir(parents=True, exist_ok=True)

    core = _read_table(core_summary_path)
    missions = _read_table(mission_summary_path)
    profiles = _read_table(profile_path)

    campaign_matrix = _campaign_opportunity_matrix(core, missions, profiles)
    financial_sizing = _financial_sizing_table(core, profiles)

    campaign_path = out_dir / f"10_campaign_opportunity_matrix_{cfg.mode}.csv"
    sizing_path = out_dir / f"10_financial_opportunity_sizing_{cfg.mode}.csv"
    md_path = out_dir / f"10_business_lens_{cfg.mode}.md"
    html_path = out_dir / f"10_business_lens_{cfg.mode}.html"

    campaign_matrix.write_csv(campaign_path)
    financial_sizing.write_csv(sizing_path)
    md_path.write_text(
        _business_lens_markdown(
            campaign_matrix=campaign_matrix,
            financial_sizing=financial_sizing,
            campaign_path=campaign_path,
            sizing_path=sizing_path,
            base_dir=out_dir,
            cfg=cfg,
        ),
        encoding="utf-8",
    )
    html_path.write_text(
        _business_lens_html(
            campaign_matrix=campaign_matrix,
            financial_sizing=financial_sizing,
            campaign_path=campaign_path,
            sizing_path=sizing_path,
            cfg=cfg,
        ),
        encoding="utf-8",
    )

    log_event(
        "Stage 10 business lens",
        "wrote business opportunity artifacts",
        cfg=cfg,
        markdown=md_path,
        campaign_matrix=campaign_path,
        financial_sizing=sizing_path,
    )
    return {
        "markdown": md_path,
        "html": html_path,
        "campaign_matrix": campaign_path,
        "financial_sizing": sizing_path,
    }


def _campaign_opportunity_matrix(core: pl.DataFrame, missions: pl.DataFrame, profiles: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    profile_lookup = _profile_lookup(profiles)

    if not core.is_empty() and "tribe_id" in core.columns:
        for row in core.sort("population_share_pct", descending=True).iter_rows(named=True):
            tribe_id = _safe_int(row.get("tribe_id"))
            profile = profile_lookup.get(tribe_id, {})
            evidence = _first_nonempty(
                row.get("top_themes"),
                row.get("top_product_terms"),
                row.get("top_products"),
                profile.get("top_themes"),
                profile.get("top_products"),
            )
            rows.append(
                {
                    "audience_type": "Core tribe",
                    "audience": f"T{tribe_id} {_clean_text(row.get('suggested_tribe_name') or profile.get('suggested_tribe_name'))}",
                    "audience_size": _safe_int(row.get("n_customers")),
                    "population_share_pct": _safe_float(row.get("population_share_pct")),
                    "commercial_play": _core_commercial_play(profile),
                    "why_this_audience": _truncate(
                        f"Distinctive purchase evidence: {evidence}. "
                        f"Assignment provenance keeps core and soft-assigned customers separate.",
                        260,
                    ),
                    "activation_example": _core_activation_example(row, profile),
                    "primary_kpi": "Incremental sales per contacted customer; repeat purchase rate; basket attachment.",
                    "financial_sizing_logic": (
                        "Use observed-period spend as baseline, then test uplift with a holdout: "
                        "target_customers x avg_total_spend x incremental_lift_pct x margin_pct."
                    ),
                    "guardrail": "Use product evidence and assignment confidence; do not infer demographics.",
                }
            )

    if not missions.is_empty() and "mission_key" in missions.columns:
        for row in missions.sort("presentation_score", descending=True).iter_rows(named=True):
            rows.append(
                {
                    "audience_type": "Shopping mission",
                    "audience": _clean_text(row.get("mission_label")),
                    "audience_size": _safe_int(row.get("mission_customers")),
                    "population_share_pct": _safe_float(row.get("population_share_pct")),
                    "commercial_play": _mission_commercial_play(row.get("mission_family")),
                    "why_this_audience": _truncate(
                        f"Mission is backed by product evidence and overlaps with core tribes: {row.get('top_core_tribes')}.",
                        260,
                    ),
                    "activation_example": _mission_activation_example(row),
                    "primary_kpi": "Campaign conversion, category penetration, units per buyer, and incremental margin versus holdout.",
                    "financial_sizing_logic": (
                        "Use mission_customers as reachable audience, then size tests as "
                        "reachable_customers x expected_response_lift x incremental_basket_margin."
                    ),
                    "guardrail": "Mission membership is multi-label and purchase-based; validate with campaign controls.",
                }
            )

    return pl.DataFrame(rows) if rows else _empty_campaign_matrix()


def _financial_sizing_table(core: pl.DataFrame, profiles: pl.DataFrame) -> pl.DataFrame:
    profile_lookup = _profile_lookup(profiles)
    rows: list[dict[str, Any]] = []

    if core.is_empty() or "tribe_id" not in core.columns:
        return _empty_financial_sizing()

    for row in core.sort("tribe_id").iter_rows(named=True):
        tribe_id = _safe_int(row.get("tribe_id"))
        profile = profile_lookup.get(tribe_id, {})
        customers = _safe_int(row.get("n_customers"))
        avg_spend = _safe_float(profile.get("avg_total_spend"))
        avg_units = _safe_float(profile.get("avg_total_units"))
        avg_basket_value = _safe_float(profile.get("avg_avg_basket_value"))
        promo_share = _safe_float(_first_nonempty(profile.get("avg_promo_share"), profile.get("avg_promo_line_share")))
        observed_spend = customers * avg_spend if customers is not None and avg_spend is not None else None
        rows.append(
            {
                "tribe_id": tribe_id,
                "audience": f"T{tribe_id} {_clean_text(row.get('suggested_tribe_name') or profile.get('suggested_tribe_name'))}",
                "customers": customers,
                "population_share_pct": _safe_float(row.get("population_share_pct")),
                "avg_total_spend_observed_period": avg_spend,
                "avg_total_units_observed_period": avg_units,
                "avg_basket_value": avg_basket_value,
                "promo_share": promo_share,
                "observed_period_spend_proxy": observed_spend,
                "one_pct_incremental_spend_proxy": _scale(observed_spend, 0.01),
                "three_pct_incremental_spend_proxy": _scale(observed_spend, 0.03),
                "five_pct_incremental_spend_proxy": _scale(observed_spend, 0.05),
                "use_case": _core_commercial_play(profile),
                "sizing_note": (
                    "Gross spend proxy only. Convert to profit opportunity with margin, redemption cost, "
                    "contact cost, and measured incremental lift from a control group."
                ),
            }
        )

    return pl.DataFrame(rows).sort("observed_period_spend_proxy", descending=True)


def _business_lens_markdown(
    *,
    campaign_matrix: pl.DataFrame,
    financial_sizing: pl.DataFrame,
    campaign_path: Path,
    sizing_path: Path,
    base_dir: Path,
    cfg: PipelineConfig,
) -> str:
    return "\n".join(
        [
            "# Stage 10 Additional Business Lens",
            "",
            f"Run mode: `{cfg.mode}`",
            "",
            "## Business Question",
            "",
            "Carrefour's starting problem was not simply to find clusters. The business problem was that broad segmentation was not specific enough to target customer groups with relevant campaigns, offers, category messages, or retention plays.",
            "",
            "## How The Tribe Solution Answers It",
            "",
            "The selected clustering solution turns checkout history into three usable business layers:",
            "",
            "1. **Core tribes**: mutually exclusive product-purchase groups for strategic segment planning and reporting.",
            "2. **Confidence-scored assignment**: operational customer lists that separate strong core members from softer extensions.",
            "3. **Shopping missions**: overlapping product-backed activation audiences that are closer to campaign briefs than generic clusters.",
            "",
            "This means Carrefour can move from broad customer groups to product-specific audiences such as organic/bio buyers, drinks buyers, fresh food missions, family and kids missions, or world cuisine discovery groups. The solution is useful because each audience comes with product evidence, size, confidence, and a testable campaign hypothesis.",
            "",
            "## Financial Opportunity Logic",
            "",
            "The tribes do not create value by existing; they create value when they improve decisions that already have economics attached:",
            "",
            "- **Incremental sales**: send more relevant offers to an audience with lifted product evidence.",
            "- **Promo efficiency**: replace blanket discounts with targeted discounts where the customer is likely to respond.",
            "- **Basket expansion**: cross-sell adjacent products within a mission, such as meal bundles, pet replenishment, or special-diet complements.",
            "- **Retention and reactivation**: protect high-value or high-frequency tribes with timely, product-relevant touches.",
            "- **Category and private-label growth**: use product lifts to choose which Carrefour own-brand, premium, or replenishment products to promote.",
            "",
            "A conservative sizing formula is:",
            "",
            "`target_customers x observed_baseline_spend x measured_incremental_lift_pct x gross_margin_pct - campaign_cost`",
            "",
            "The sizing table includes 1%, 3%, and 5% gross spend uplift proxies by tribe. These are not forecasts. They are planning scenarios that must be validated with holdout tests.",
            "",
            "## Activation Tables",
            "",
            f"- Campaign opportunity matrix: `{_path_label(campaign_path, base_dir)}`",
            f"- Financial opportunity sizing template: `{_path_label(sizing_path, base_dir)}`",
            "",
            "## Campaign Opportunity Preview",
            "",
            _markdown_table(_campaign_preview(campaign_matrix)),
            "",
            "## Financial Sizing Preview",
            "",
            _markdown_table(financial_sizing.head(8)),
            "",
            "## Measurement Guardrails",
            "",
            "- Keep product identity and purchased quantities as the modeling signal.",
            "- Use spend, margin, promo cost, and response rates only after clustering for profiling and ROI measurement.",
            "- Run campaign holdouts before claiming financial impact.",
            "- Report core and soft-assigned customers separately when campaign risk or budget is high.",
            "- Treat labels as purchase-evidence summaries, not demographic or identity claims.",
            "",
        ]
    )


def _business_lens_html(
    *,
    campaign_matrix: pl.DataFrame,
    financial_sizing: pl.DataFrame,
    campaign_path: Path,
    sizing_path: Path,
    cfg: PipelineConfig,
) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Stage 10 Additional Business Lens</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 0; color: #17202a; background: #f6f7f9; }}
header {{ background: #ffffff; border-bottom: 1px solid #d0d5dd; padding: 26px 34px; }}
main {{ padding: 24px 34px 42px; }}
h1, h2 {{ margin: 0; }}
h2 {{ margin-top: 28px; margin-bottom: 12px; font-size: 20px; }}
p, li {{ line-height: 1.45; }}
.muted {{ color: #667085; font-size: 13px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 12px; margin-top: 16px; }}
.card {{ background: #ffffff; border: 1px solid #d0d5dd; border-radius: 8px; padding: 14px; }}
.card strong {{ display: block; margin-bottom: 7px; }}
code {{ background: #eef2f6; padding: 2px 5px; border-radius: 4px; }}
table {{ border-collapse: collapse; width: 100%; table-layout: fixed; background: #ffffff; }}
th, td {{ border: 1px solid #d0d5dd; padding: 9px; vertical-align: top; font-size: 13px; word-wrap: break-word; }}
th {{ background: #eef2f6; text-align: left; }}
a {{ color: #175cd3; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>
<header>
<h1>Stage 10 Additional Business Lens</h1>
<p class="muted">Run mode: {escape(str(cfg.mode))}. This layer translates product-first tribes into CRM, campaign, and opportunity-sizing language.</p>
</header>
<main>
<h2>How The Solution Answers Carrefour's Problem</h2>
<p>Carrefour needed better segmentation for targeted campaigns. The selected solution turns checkout behavior into product-backed audiences with size, evidence, confidence, and activation logic.</p>
<div class="grid">
<div class="card"><strong>Core tribes</strong><span class="muted">Mutually exclusive product-purchase groups for strategic planning and reporting.</span></div>
<div class="card"><strong>Confidence scoring</strong><span class="muted">Operational lists separate strongest core members from softer campaign reach.</span></div>
<div class="card"><strong>Shopping missions</strong><span class="muted">Overlapping product-backed audiences that map naturally to campaign briefs.</span></div>
<div class="card"><strong>Financial tests</strong><span class="muted">Revenue impact is estimated through controlled uplift, margin, and campaign cost.</span></div>
</div>
<h2>Financial Opportunity Logic</h2>
<p><code>target_customers x observed_baseline_spend x measured_incremental_lift_pct x gross_margin_pct - campaign_cost</code></p>
<p class="muted">The 1%, 3%, and 5% uplift columns are gross spend planning scenarios, not forecasts.</p>
<p><a href="{escape(campaign_path.name)}">Open campaign opportunity matrix</a> | <a href="{escape(sizing_path.name)}">Open financial sizing template</a></p>
<h2>Campaign Opportunity Preview</h2>
{_html_table(_campaign_preview(campaign_matrix))}
<h2>Financial Sizing Preview</h2>
{_html_table(financial_sizing.head(10))}
<h2>Measurement Guardrails</h2>
<ul>
<li>Keep product identity and purchased quantities as the modeling signal.</li>
<li>Use spend, margin, promo cost, and response rates only after clustering.</li>
<li>Run holdout tests before claiming financial impact.</li>
<li>Report core and soft-assigned customers separately when risk or budget is high.</li>
<li>Treat labels as purchase-evidence summaries, not demographic or identity claims.</li>
</ul>
</main>
</body>
</html>
"""


def _core_commercial_play(profile: dict[str, Any]) -> str:
    promo_share = _safe_float(_first_nonempty(profile.get("avg_promo_share"), profile.get("avg_promo_line_share")))
    avg_spend = _safe_float(profile.get("avg_total_spend"))
    frequency = _safe_float(profile.get("avg_frequency_per_30d"))
    if promo_share is not None and promo_share >= 0.25:
        return "Promo precision and discount efficiency"
    if avg_spend is not None and avg_spend >= 500:
        return "Retention and high-value basket growth"
    if frequency is not None and frequency >= 4:
        return "Frequency-based replenishment and cross-sell"
    return "Personalized product recommendation and basket expansion"


def _core_activation_example(row: dict[str, Any], profile: dict[str, Any]) -> str:
    evidence = _first_nonempty(row.get("top_products"), row.get("top_themes"), profile.get("top_products"), profile.get("top_themes"))
    return _truncate(
        f"Build a CRM audience from T{_safe_int(row.get('tribe_id'))}; use top lifted products/themes as offer creative and adjacent-category recommendations: {evidence}.",
        260,
    )


def _mission_commercial_play(mission_family: Any) -> str:
    family = _clean_text(mission_family).lower()
    if "fresh" in family:
        return "Fresh category frequency, meal bundles, and attachment"
    if "natural" in family or "wellness" in family or "special diet" in family or "health" in family:
        return "Premium, wellness, and relevance-led retention"
    if "family" in family:
        return "Lifecycle replenishment and family basket growth"
    if "pets" in family or "household" in family:
        return "Repeat replenishment and subscription-style reminders"
    if "drinks" in family:
        return "Occasion-led bundles and event timing"
    if "world" in family:
        return "Discovery, recipe-led cross-sell, and range awareness"
    return "Mission-led offer targeting and category growth"


def _mission_activation_example(row: dict[str, Any]) -> str:
    return _truncate(
        f"Create a multi-label audience for {row.get('mission_label')}; test mission-specific offers using the strongest product examples: {row.get('top_products')}.",
        260,
    )


def _profile_lookup(profiles: pl.DataFrame) -> dict[int, dict[str, Any]]:
    if profiles.is_empty() or "tribe_id" not in profiles.columns:
        return {}
    return {_safe_int(row.get("tribe_id")): row for row in profiles.iter_rows(named=True)}


def _read_table(path: str | Path) -> pl.DataFrame:
    path = Path(path)
    if not path.exists():
        return pl.DataFrame()
    if path.suffix.lower() == ".parquet":
        return pl.read_parquet(path)
    return pl.read_csv(path)


def _markdown_table(df: pl.DataFrame) -> str:
    if df.is_empty():
        return "No rows available."
    columns = df.columns
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in df.iter_rows(named=True):
        values = [_markdown_cell(row.get(col)) for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _campaign_preview(df: pl.DataFrame, limit_each: int = 5) -> pl.DataFrame:
    if df.is_empty() or "audience_type" not in df.columns:
        return df.head(limit_each * 2)
    previews = []
    for audience_type in ["Core tribe", "Shopping mission"]:
        subset = df.filter(pl.col("audience_type") == audience_type).head(limit_each)
        if not subset.is_empty():
            previews.append(subset)
    return pl.concat(previews, how="vertical") if previews else df.head(limit_each * 2)


def _html_table(df: pl.DataFrame) -> str:
    if df.is_empty():
        return "<p class='muted'>No rows available.</p>"
    header = "".join(f"<th>{escape(str(col).replace('_', ' ').title())}</th>" for col in df.columns)
    rows = []
    for row in df.iter_rows(named=True):
        cells = "".join(f"<td>{escape(_display_value(row.get(col)))}</td>" for col in df.columns)
        rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _empty_campaign_matrix() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "audience_type": pl.Utf8,
            "audience": pl.Utf8,
            "audience_size": pl.Int64,
            "population_share_pct": pl.Float64,
            "commercial_play": pl.Utf8,
            "why_this_audience": pl.Utf8,
            "activation_example": pl.Utf8,
            "primary_kpi": pl.Utf8,
            "financial_sizing_logic": pl.Utf8,
            "guardrail": pl.Utf8,
        }
    )


def _empty_financial_sizing() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "tribe_id": pl.Int64,
            "audience": pl.Utf8,
            "customers": pl.Int64,
            "population_share_pct": pl.Float64,
            "avg_total_spend_observed_period": pl.Float64,
            "avg_total_units_observed_period": pl.Float64,
            "avg_basket_value": pl.Float64,
            "promo_share": pl.Float64,
            "observed_period_spend_proxy": pl.Float64,
            "one_pct_incremental_spend_proxy": pl.Float64,
            "three_pct_incremental_spend_proxy": pl.Float64,
            "five_pct_incremental_spend_proxy": pl.Float64,
            "use_case": pl.Utf8,
            "sizing_note": pl.Utf8,
        }
    )


def _first_nonempty(*values: Any) -> str:
    for value in values:
        text = _clean_text(value)
        if text and text.lower() != "nan":
            return text
    return ""


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\n", " ").replace("\r", " ").strip()


def _safe_int(value: Any) -> int:
    try:
        if value is None or value == "":
            return 0
        return int(round(float(value)))
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _scale(value: float | None, factor: float) -> float | None:
    return None if value is None else value * factor


def _truncate(value: Any, max_len: int) -> str:
    text = _clean_text(value)
    return text if len(text) <= max_len else text[: max_len - 3].rstrip() + "..."


def _display_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:,.2f}"
    return str(value)


def _markdown_cell(value: Any) -> str:
    return _display_value(value).replace("|", "/").replace("\n", " ")


def _path_label(path: Path, base_dir: Path) -> str:
    try:
        return path.relative_to(base_dir).as_posix()
    except ValueError:
        return path.as_posix()
