"""Strict Clef response validation and fail-safe retention."""

from __future__ import annotations

from types import SimpleNamespace

import requests

from config import build_config
from conversation_grouper import ConversationUnit
from relevance_filter import ClefAPIError, RelevanceFilter, parse_answers


def make_units(n=3):
    return [
        ConversationUnit(
            unit_id=f"u{i}",
            kind="solo",
            messages=[{"role": "user", "content": f"msg {i}"}],
        )
        for i in range(n)
    ]


def run_filter(cfg, responses, units=None):
    """responses: list of dicts/exceptions consumed in order."""
    units = units or make_units()
    calls = {"n": 0}

    def fake(payload):
        idx = min(calls["n"], len(responses) - 1)
        calls["n"] += 1
        item = responses[idx]
        if isinstance(item, Exception):
            raise item
        return item

    engine = RelevanceFilter(cfg, call_api=fake, sleep=lambda _: None)
    decisions, outcome = engine.filter_chunk(units, "summary text")
    return {d.question_id: d for d in decisions}, outcome, calls["n"]


def answer(scores):
    return {
        "success": True,
        "result": {"answers": {qid: {"noul": v} for qid, v in scores.items()}},
    }


def test_all_messages_retained(base_config):
    ids = [f"u{i}" for i in range(3)]
    decisions, outcome, _ = run_filter(
        base_config, [answer({q: 0.9 for q in ids})]
    )
    assert all(d.retain for d in decisions.values())
    assert all(d.source == "clef" for d in decisions.values())


def test_configurable_threshold_is_respected(base_config):
    cfg = build_config({**base_config, "compress_threshold": 0.8})
    ids = [f"u{i}" for i in range(3)]
    decisions, _, _ = run_filter(
        cfg, [answer({"u0": 0.9, "u1": 0.5, "u2": 0.85})]
    )
    assert decisions["u0"].retain
    assert not decisions["u1"].retain
    assert decisions["u2"].retain


def test_missing_answer_is_fail_safe_retained(base_config):
    decisions, outcome, _ = run_filter(base_config, [answer({"u0": 0.9, "u1": 0.1})])
    assert decisions["u2"].retain
    assert decisions["u2"].source == "missing"
    assert outcome.missing_count == 1


def test_non_numeric_probability_retained(base_config):
    result = {"success": True, "result": {"answers": {
        "u0": {"noul": "high"}, "u1": {"noul": None}, "u2": {"noul": 0.9},
    }}}
    decisions, outcome, _ = run_filter(base_config, [result])
    assert decisions["u0"].retain and decisions["u0"].source == "malformed"
    assert decisions["u1"].retain and decisions["u1"].source == "malformed"
    assert decisions["u2"].source == "clef"


def test_nan_and_infinite_probabilities_retained(base_config):
    result = {"success": True, "result": {"answers": {
        "u0": {"noul": float("nan")}, "u1": {"noul": float("inf")},
        "u2": {"noul": -0.5},
    }}}
    decisions, _, _ = run_filter(base_config, [result])
    assert all(d.retain and d.source == "malformed" for d in decisions.values())


def test_out_of_range_probability_retained(base_config):
    result = {"success": True, "result": {"answers": {
        "u0": {"noul": 1.7}, "u1": {"noul": -0.2}, "u2": {"noul": 0.9},
    }}}
    decisions, _, _ = run_filter(base_config, [result])
    assert decisions["u0"].source == "malformed"
    assert decisions["u1"].source == "malformed"


def test_unexpected_aggregate_schema_is_fail_safe(base_config):
    result = {"success": True, "result": {"answers": {"keep": {"noul": 0.9}}}}
    decisions, outcome, _ = run_filter(base_config, [result])
    # Aggregate object cannot map to per-question ids -> ambiguous -> retain all.
    assert all(d.retain for d in decisions.values())
    assert all(d.source == "ambiguous" for d in decisions.values())


def test_missing_answers_container_retains_all(base_config):
    # A 200 response with no answers container is retried, then fails safe.
    decisions, outcome, _ = run_filter(base_config, [{"success": True, "result": {}}])
    assert all(d.retain for d in decisions.values())
    assert all(d.source in ("api_error", "missing") for d in decisions.values())


def test_api_timeout_retains_all_with_bounded_retries(base_config):
    decisions, outcome, calls = run_filter(
        base_config, [requests.Timeout("timed out")]
    )
    assert all(d.retain for d in decisions.values())
    assert all(d.source == "api_error" for d in decisions.values())
    assert calls <= base_config["api_max_retries"] + 1


def test_rate_limit_then_success(base_config):
    ids = [f"u{i}" for i in range(3)]
    decisions, _, calls = run_filter(
        base_config, [ClefAPIError("429", status=429), answer({q: 0.9 for q in ids})]
    )
    assert calls == 2
    assert all(d.source == "clef" for d in decisions.values())


def test_partial_response_retains_missing_only(base_config):
    decisions, outcome, _ = run_filter(base_config, [answer({"u0": 0.1, "u1": 0.2})])
    assert not decisions["u0"].retain
    assert not decisions["u1"].retain
    assert decisions["u2"].retain and decisions["u2"].source == "missing"


def test_5xx_is_retried_then_fail_safe(base_config):
    decisions, outcome, calls = run_filter(
        base_config, [ClefAPIError("500", status=500)]
    )
    assert calls == base_config["api_max_retries"] + 1
    assert all(d.retain for d in decisions.values())


def test_parse_answers_bare_number_compatibility():
    parsed = parse_answers(
        {"result": {"answers": {"u0": 0.7}}}, ["u0", "u1"]
    )
    assert parsed["u0"].status == "ok" and parsed["u0"].score == 0.7
    assert parsed["u1"].status == "missing"


def test_forced_chunk_retains_without_api_call(base_config):
    units = make_units(2)
    engine = RelevanceFilter(base_config, call_api=lambda _: (_ for _ in ()).throw(
        AssertionError("API must not be called")
    ))
    decisions, outcome = engine.filter_chunk(
        units, "s", chunk_forced_retain=True, chunk_reason="oversize"
    )
    assert all(d.source == "forced" for d in decisions)
    assert outcome.api_calls == 0
