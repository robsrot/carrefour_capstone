"""Optional post-profiling LLM interpretation helpers."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

import polars as pl
from dotenv import load_dotenv

from src.config import CONFIG, PipelineConfig
from src.progress import log_event


GeminiCaller = Callable[[str, str, dict[str, Any]], str]


def write_stage7_gemini_analysis(
    llm_evidence_path: str | Path,
    *,
    final_index_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    cfg: PipelineConfig = CONFIG,
    caller: GeminiCaller | None = None,
) -> dict[str, Any]:
    """Interpret final Stage 7 aggregate tribe evidence with Gemini.

    This is deliberately post hoc: it reads the Stage 7 LLM evidence table and
    final index only. It does not alter clustering, profile promotion, or tribe
    cards.
    """

    out_dir = Path(output_dir) if output_dir else cfg.artifacts / "stage7" / "llm_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / f"stage7_gemini_analysis_manifest_{cfg.mode}.json"

    enabled = bool(cfg.get("llm_analysis.enabled", False))
    if not enabled:
        return _write_gemini_manifest(
            manifest_path,
            cfg=cfg,
            status="disabled",
            reason="Set llm_analysis.enabled=true to run optional Gemini interpretation.",
        )

    provider = str(cfg.get("llm_analysis.provider", "gemini")).lower()
    if provider != "gemini":
        return _write_gemini_manifest(
            manifest_path,
            cfg=cfg,
            status="skipped",
            reason=f"Unsupported llm_analysis.provider={provider!r}.",
        )

    load_dotenv(cfg.root / ".env", override=False)
    api_key_env = str(cfg.get("llm_analysis.api_key_env", "GEMINI_API_KEY"))
    api_key = os.getenv(api_key_env)
    if not api_key and caller is None:
        status = _write_gemini_manifest(
            manifest_path,
            cfg=cfg,
            status="missing_api_key",
            reason=f"Set {api_key_env} in the environment or .env to run Gemini interpretation.",
        )
        if bool(cfg.get("llm_analysis.fail_on_missing_key", False)):
            raise RuntimeError(status["reason"])
        return status

    evidence = _read_csv(llm_evidence_path)
    final_index = _read_csv(final_index_path) if final_index_path else pl.DataFrame()
    if evidence.is_empty():
        return _write_gemini_manifest(
            manifest_path,
            cfg=cfg,
            status="empty_evidence",
            reason=f"No rows found in {llm_evidence_path}.",
        )

    system_instruction = _gemini_system_instruction()
    request_config = _gemini_request_config(cfg)
    api_caller = caller or _gemini_generate_content
    rows: list[dict[str, Any]] = []
    prompt_rows: list[dict[str, Any]] = []
    max_rows = int(cfg.get("llm_analysis.max_evidence_rows_per_tribe", 24))

    for tribe_id in sorted(evidence["tribe_id"].unique().to_list()):
        prompt = _tribe_analysis_prompt(
            int(tribe_id),
            evidence=evidence,
            final_index=final_index,
            max_evidence_rows=max_rows,
        )
        prompt_rows.append({"tribe_id": int(tribe_id), "prompt": prompt})
        try:
            raw_response = api_caller(prompt, system_instruction, request_config)
            parsed = _parse_gemini_json(raw_response)
            rows.append(_analysis_row(int(tribe_id), parsed, raw_response, "parsed"))
        except Exception as exc:
            if bool(cfg.get("llm_analysis.fail_on_error", False)):
                raise
            rows.append(
                {
                    "tribe_id": int(tribe_id),
                    "recommended_name": None,
                    "confidence": "error",
                    "one_sentence_read": None,
                    "why_this_tribe_exists": None,
                    "tribe_vs_rest": None,
                    "activation_ideas": None,
                    "caveats": str(exc),
                    "needs_human_review": True,
                    "raw_response": "",
                    "parse_status": "error",
                }
            )

    analysis = pl.DataFrame(rows).sort("tribe_id") if rows else _empty_analysis_table()
    analysis_csv = out_dir / f"stage7_gemini_tribe_analysis_{cfg.mode}.csv"
    analysis_json = out_dir / f"stage7_gemini_tribe_analysis_{cfg.mode}.json"
    analysis_md = out_dir / f"stage7_gemini_tribe_analysis_{cfg.mode}.md"
    analysis.write_csv(analysis_csv)
    analysis_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    analysis_md.write_text(_analysis_markdown(rows, cfg=cfg), encoding="utf-8")

    prompt_audit_path = None
    if bool(cfg.get("llm_analysis.write_prompt_audit", True)):
        prompt_audit_path = out_dir / f"stage7_gemini_prompt_audit_{cfg.mode}.json"
        prompt_audit_path.write_text(json.dumps(prompt_rows, indent=2, ensure_ascii=False), encoding="utf-8")

    status = _write_gemini_manifest(
        manifest_path,
        cfg=cfg,
        status="complete",
        reason="Gemini interpretation completed from final Stage 7 aggregate evidence.",
        analysis_csv=analysis_csv,
        analysis_json=analysis_json,
        analysis_markdown=analysis_md,
        prompt_audit_json=prompt_audit_path,
        tribes=analysis.height,
        model=str(cfg.get("llm_analysis.model", "gemini-3.5-flash")),
    )
    log_event(
        "Stage 7 Gemini analysis",
        "wrote optional LLM interpretation",
        cfg=cfg,
        tribes=analysis.height,
        markdown=analysis_md,
    )
    return status


def _gemini_generate_content(prompt: str, system_instruction: str, request_config: dict[str, Any]) -> str:
    api_key_env = str(request_config["api_key_env"])
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing {api_key_env}.")

    model = str(request_config["model"])
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    payload = {
        "system_instruction": {"parts": [{"text": system_instruction}]},
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": float(request_config["temperature"]),
            "topP": float(request_config["top_p"]),
            "maxOutputTokens": int(request_config["max_output_tokens"]),
        },
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=float(request_config["timeout_seconds"])) as response:
            response_json = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini API HTTP {exc.code}: {body[:500]}") from exc
    return _extract_gemini_text(response_json)


def _gemini_request_config(cfg: PipelineConfig) -> dict[str, Any]:
    return {
        "api_key_env": str(cfg.get("llm_analysis.api_key_env", "GEMINI_API_KEY")),
        "model": str(cfg.get("llm_analysis.model", "gemini-3.5-flash")),
        "temperature": float(cfg.get("llm_analysis.temperature", 0.2)),
        "top_p": float(cfg.get("llm_analysis.top_p", 0.9)),
        "max_output_tokens": int(cfg.get("llm_analysis.max_output_tokens", 1800)),
        "timeout_seconds": float(cfg.get("llm_analysis.request_timeout_seconds", 90)),
    }


def _read_csv(path: str | Path | None) -> pl.DataFrame:
    if path is None:
        return pl.DataFrame()
    csv_path = Path(path)
    return pl.read_csv(csv_path) if csv_path.exists() else pl.DataFrame()


def _tribe_analysis_prompt(
    tribe_id: int,
    *,
    evidence: pl.DataFrame,
    final_index: pl.DataFrame,
    max_evidence_rows: int,
) -> str:
    tribe_evidence = evidence.filter(pl.col("tribe_id") == tribe_id)
    if not tribe_evidence.is_empty():
        tribe_evidence = tribe_evidence.sort(
            by=[
                pl.col("proof_role").replace_strict(
                    {
                        "primary_actionability_proof": "0",
                        "primary_candidate": "1",
                        "supporting_category_context": "2",
                        "supplemental_curated_theme_context": "3",
                    },
                    default="9",
                ),
                "evidence_type",
                "evidence_rank",
            ]
        ).head(max_evidence_rows)
    index_row = _index_row(final_index, tribe_id)
    payload = {
        "project_context": _gemini_project_context(),
        "analysis_task": (
            "Name and interpret this already-discovered tribe from purchase evidence. "
            "The goal is a business-ready segment label that a stakeholder can defend from the supplied products, "
            "product terms, sectors, lifts, reach, q-values, confidence, and caveats."
        ),
        "evidence_priority": [
            "1. primary_actionability_proof rows, especially product_term and lifted_product evidence",
            "2. repeated lifted SKUs or product terms that describe the same purchase need",
            "3. sector/category context when it supports the product evidence",
            "4. curated_theme rows only as supporting context, never as the sole reason for a name",
        ],
        "naming_rules": [
            "Use only products, product terms, sectors, categories, reach, lift, q-values, and readiness evidence supplied here.",
            "Prefer names like '<specific product/category need> Buyers' or '<specific product/category need> Purchase Cluster'.",
            "Do not invent personas, motives, demographics, household structure, income, health status, religion, lifestyle, or intent.",
            "Do not call a tribe 'baby', 'organic', 'pet', etc. unless supplied product-term/SKU evidence clearly supports that read.",
            "If evidence is mixed, use a mixed-product name and set confidence to medium or low.",
            "If the tribe cannot be coherently named from the evidence, say so and set needs_human_review=true.",
        ],
        "confidence_rules": {
            "high": "coherent product-term or repeated lifted-SKU evidence, strong lift/reach, significant q-values where supplied, and ready_strong readiness",
            "medium": "some coherent purchase evidence but mixed categories, weaker reach, or partial support",
            "low": "random-looking SKU list, weak/missing metrics, no coherent purchase theme, or review readiness",
        },
        "prohibited_claims": [
            "Do not infer that customers have babies, pets, dietary restrictions, religion, income level, or health needs.",
            "Phrase claims as purchase behavior, e.g. 'baby-food buyers' not 'parents' and 'lactose-free product buyers' not 'lactose-intolerant customers'.",
        ],
        "tribe_id": tribe_id,
        "deterministic_profile_summary": index_row,
        "aggregate_evidence_rows": tribe_evidence.to_dicts(),
        "required_output_schema": {
            "tribe_id": tribe_id,
            "recommended_name": "concise purchase-evidence name",
            "confidence": "high|medium|low",
            "one_sentence_read": "plain-language segment read",
            "why_this_tribe_exists": "why the evidence supports this as a product-purchase tribe",
            "tribe_vs_rest": "how the tribe differs from the rest of customers using only given metrics",
            "activation_ideas": ["purchase-based idea", "purchase-based idea"],
            "caveats": ["evidence caveat or uncertainty"],
            "needs_human_review": False,
        },
    }
    return (
        "Analyze this already-computed Carrefour tribe evidence. Return strict JSON only.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def _index_row(final_index: pl.DataFrame, tribe_id: int) -> dict[str, Any]:
    if final_index.is_empty() or "tribe_id" not in final_index.columns:
        return {}
    row = final_index.filter(pl.col("tribe_id") == tribe_id)
    return row.to_dicts()[0] if not row.is_empty() else {}


def _gemini_system_instruction() -> str:
    return (
        "You are a careful commercial analytics assistant helping interpret a Carrefour product-first customer "
        "segmentation project. The clustering signal comes from product identity and quantity-weighted purchase "
        "behavior only; spend and visit metrics are post-clustering context. You interpret tribes after clustering "
        "using only supplied aggregate purchase evidence. You do not cluster customers, change assignments, filter "
        "clusters, or promote tribes. Your main job is to propose precise, defensible tribe names from what the "
        "customers buy more than the rest of the population. Do not infer demographics, identity, household "
        "structure, income, religion, health status, or motives. Prefer data-driven product terms and lifted SKUs "
        "over curated themes. If evidence is mixed, random-looking, or weak, say so and mark the result for human "
        "review. Return only valid JSON matching the requested schema."
    )


def _gemini_project_context() -> dict[str, Any]:
    return {
        "objective": (
            "Discover organic, product-first Carrefour customer tribes and turn retained clusters into actionable, "
            "evidence-backed purchase segments."
        ),
        "modeling_contract": [
            "Customer embeddings use products purchased and quantities purchased.",
            "Demographic attributes are not used.",
            "Spend, basket value, promo share, and visit frequency are interpretation context only.",
            "UMAP and HDBSCAN already ran before this LLM step.",
            "The LLM must not affect assignments, noise, cluster filtering, readiness, or final promotion.",
        ],
        "stage6_summary": (
            "Stage 6 discovered hard UMAP-HDBSCAN core tribes, reran stricter HDBSCAN on first-pass noise, merged "
            "the two passes, and retained clusters only when strong product-lift evidence existed."
        ),
        "stage7_summary": (
            "Stage 7 computed aggregate product, product-term, sector, theme, confidence, readiness, and commercial "
            "context evidence. This LLM step only names/interprets those frozen aggregate results."
        ),
    }


def _extract_gemini_text(response_json: dict[str, Any]) -> str:
    parts = (
        response_json.get("candidates", [{}])[0]
        .get("content", {})
        .get("parts", [])
    )
    text = "".join(str(part.get("text", "")) for part in parts)
    if not text.strip():
        raise RuntimeError("Gemini response did not contain text.")
    return text.strip()


def _parse_gemini_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise


def _analysis_row(tribe_id: int, parsed: dict[str, Any], raw_response: str, parse_status: str) -> dict[str, Any]:
    return {
        "tribe_id": int(parsed.get("tribe_id") or tribe_id),
        "recommended_name": parsed.get("recommended_name"),
        "confidence": parsed.get("confidence"),
        "one_sentence_read": parsed.get("one_sentence_read"),
        "why_this_tribe_exists": parsed.get("why_this_tribe_exists"),
        "tribe_vs_rest": parsed.get("tribe_vs_rest"),
        "activation_ideas": _json_text(parsed.get("activation_ideas")),
        "caveats": _json_text(parsed.get("caveats")),
        "needs_human_review": bool(parsed.get("needs_human_review", False)),
        "raw_response": raw_response,
        "parse_status": parse_status,
    }


def _json_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _empty_analysis_table() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "tribe_id": pl.Int64,
            "recommended_name": pl.Utf8,
            "confidence": pl.Utf8,
            "one_sentence_read": pl.Utf8,
            "why_this_tribe_exists": pl.Utf8,
            "tribe_vs_rest": pl.Utf8,
            "activation_ideas": pl.Utf8,
            "caveats": pl.Utf8,
            "needs_human_review": pl.Boolean,
            "raw_response": pl.Utf8,
            "parse_status": pl.Utf8,
        }
    )


def _analysis_markdown(rows: list[dict[str, Any]], *, cfg: PipelineConfig) -> str:
    lines = [
        "# Stage 7 Gemini Tribe Interpretation",
        "",
        "This is an optional post-hoc interpretation of frozen Stage 7 aggregate evidence. It does not affect clustering, readiness, or final tribe promotion.",
        "",
    ]
    if not rows:
        lines.append("No Gemini interpretation rows were produced.")
        return "\n".join(lines)
    for row in rows:
        lines.extend(
            [
                f"## T{int(row.get('tribe_id')):02d} - {row.get('recommended_name') or 'Review needed'}",
                "",
                f"- Confidence: {row.get('confidence') or 'n/a'}",
                f"- Read: {row.get('one_sentence_read') or 'n/a'}",
                f"- Why it exists: {row.get('why_this_tribe_exists') or 'n/a'}",
                f"- Tribe vs rest: {row.get('tribe_vs_rest') or 'n/a'}",
                f"- Activation ideas: {row.get('activation_ideas') or 'n/a'}",
                f"- Caveats: {row.get('caveats') or 'n/a'}",
                f"- Needs human review: {row.get('needs_human_review')}",
                "",
            ]
        )
    return "\n".join(lines)


def _write_gemini_manifest(
    manifest_path: Path,
    *,
    cfg: PipelineConfig,
    status: str,
    reason: str,
    **paths_or_values: Any,
) -> dict[str, Any]:
    manifest = {
        "stage": "7_optional_gemini_analysis",
        "mode": cfg.mode,
        "status": status,
        "reason": reason,
        "llm_boundary": "post_stage7_interpretation_only",
        "does_not_affect": ["customer_vectors", "umap", "hdbscan", "cluster_filtering", "final_promotion"],
    }
    for key, value in paths_or_values.items():
        manifest[key] = str(value) if isinstance(value, Path) else value
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest["manifest_json"] = manifest_path
    return manifest
