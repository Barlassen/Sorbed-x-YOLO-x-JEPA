"""Explainability contract tests.

The central guarantee: every piece of evidence points at a metric the system
actually produces. This structurally forbids fabricated explanations.
"""

from __future__ import annotations

from sorbed.domain.analysis import WoundAnalysis
from sorbed.explain.resolve import metric_ref_exists, resolve_metric_ref


def test_every_evidence_metric_ref_resolves(granulating_analysis: WoundAnalysis):
    analysis = granulating_analysis
    assert analysis.decision.evidence, "expected a graded wound to carry evidence"
    for ev in analysis.decision.evidence:
        assert metric_ref_exists(analysis, ev.metric_ref), (
            f"evidence '{ev.code}' references a non-existent metric: {ev.metric_ref}"
        )


def test_resolver_reaches_enum_keyed_fraction(granulating_analysis: WoundAnalysis):
    value = resolve_metric_ref(granulating_analysis, "metrics.tissue.fractions.granulation")
    assert isinstance(value, float)


def test_resolver_reaches_property(granulating_analysis: WoundAnalysis):
    value = resolve_metric_ref(granulating_analysis, "metrics.tissue.obscured_fraction")
    assert isinstance(value, float)


def test_narrative_is_present_and_mentions_stage(granulating_analysis: WoundAnalysis):
    narrative = granulating_analysis.decision.narrative
    assert narrative
    assert "grade" in narrative.lower()


def test_missing_ref_reported_absent(granulating_analysis: WoundAnalysis):
    assert not metric_ref_exists(granulating_analysis, "metrics.tissue.nonexistent_field")
