import json

import polars as pl

from src.config import PipelineConfig
from src.llm_analysis import write_stage7_gemini_analysis


def _cfg(tmp_path, *, enabled: bool) -> PipelineConfig:
    return PipelineConfig(
        values={
            "paths": {"outputs": "outputs"},
            "progress": {"enabled": False},
            "llm_analysis": {
                "enabled": enabled,
                "provider": "gemini",
                "api_key_env": "TEST_GEMINI_API_KEY",
                "model": "gemini-3.5-flash",
                "temperature": 0.2,
                "top_p": 0.9,
                "max_output_tokens": 800,
                "request_timeout_seconds": 5,
                "max_evidence_rows_per_tribe": 8,
                "write_prompt_audit": True,
                "fail_on_missing_key": False,
                "fail_on_error": False,
            },
        },
        mode="dev",
        root=tmp_path,
    )


def _write_evidence(tmp_path):
    evidence_path = tmp_path / "stage7_llm_evidence_long_dev.csv"
    index_path = tmp_path / "stage7_final_index_dev.csv"
    pl.DataFrame(
        [
            {
                "tribe_id": 2,
                "evidence_rank": 1,
                "evidence_type": "product_term",
                "proof_role": "primary_actionability_proof",
                "label": "hero baby",
                "customers": 120,
                "tribe_reach_pct": 18.5,
                "lift_vs_rest": 3.2,
                "q_value": 0.001,
                "profiling_readiness": "ready_strong",
                "actionability_proof_source": "data_driven_product_terms",
                "actionability_proof": "Product-term proof: hero baby",
            },
            {
                "tribe_id": 2,
                "evidence_rank": 1,
                "evidence_type": "sector",
                "proof_role": "supporting_category_context",
                "label": "BEBE",
                "customers": None,
                "tribe_reach_pct": None,
                "lift_vs_rest": 2.1,
                "q_value": 0.004,
                "profiling_readiness": "ready_strong",
                "actionability_proof_source": "data_driven_product_terms",
                "actionability_proof": "Product-term proof: hero baby",
            },
        ]
    ).write_csv(evidence_path)
    pl.DataFrame(
        [
            {
                "tribe_id": 2,
                "tribe_name": "Hero Baby Purchase Cluster",
                "confidence_score": 84,
                "customers": 650,
                "share_pct": 1.2,
                "actionability_proof": "Product-term proof: hero baby",
            }
        ]
    ).write_csv(index_path)
    return evidence_path, index_path


def test_gemini_analysis_skips_when_disabled(tmp_path):
    evidence_path, index_path = _write_evidence(tmp_path)

    result = write_stage7_gemini_analysis(
        evidence_path,
        final_index_path=index_path,
        output_dir=tmp_path / "llm",
        cfg=_cfg(tmp_path, enabled=False),
    )

    assert result["status"] == "disabled"
    assert result["manifest_json"].exists()
    assert json.loads(result["manifest_json"].read_text(encoding="utf-8"))["does_not_affect"] == [
        "customer_vectors",
        "umap",
        "hdbscan",
        "cluster_filtering",
        "final_promotion",
    ]


def test_gemini_analysis_writes_interpretation_with_fake_caller(tmp_path, monkeypatch):
    evidence_path, index_path = _write_evidence(tmp_path)
    monkeypatch.setenv("TEST_GEMINI_API_KEY", "not-a-real-key")
    prompts = []

    def fake_caller(prompt, system_instruction, request_config):
        prompts.append((prompt, system_instruction, request_config))
        return json.dumps(
            {
                "tribe_id": 2,
                "recommended_name": "Baby Food Over-Indexers",
                "confidence": "high",
                "one_sentence_read": "Customers in this tribe over-index on baby food evidence.",
                "why_this_tribe_exists": "The product-term and sector evidence point to baby food purchases.",
                "tribe_vs_rest": "Hero baby term is 3.2x vs rest with 18.5% reach.",
                "activation_ideas": ["baby food replenishment journey"],
                "caveats": ["Use purchase evidence only."],
                "needs_human_review": False,
            }
        )

    result = write_stage7_gemini_analysis(
        evidence_path,
        final_index_path=index_path,
        output_dir=tmp_path / "llm",
        cfg=_cfg(tmp_path, enabled=True),
        caller=fake_caller,
    )

    assert result["status"] == "complete"
    assert result["analysis_csv"].endswith("stage7_gemini_tribe_analysis_dev.csv")
    analysis = pl.read_csv(result["analysis_csv"])
    assert analysis[0, "recommended_name"] == "Baby Food Over-Indexers"
    assert analysis[0, "parse_status"] == "parsed"
    assert prompts
    assert "TEST_GEMINI_API_KEY" not in prompts[0][0]
    assert "not-a-real-key" not in prompts[0][0]
    assert "product-first Carrefour customer tribes" in prompts[0][0]
    assert "curated_theme rows only as supporting context" in prompts[0][0]
    assert "baby-food buyers" in prompts[0][0]
    assert "quantity-weighted purchase" in prompts[0][1]
    assert "what the customers buy more than the rest of the population" in prompts[0][1]
    assert "does not affect clustering" in result["reason"].lower() or result["llm_boundary"]
