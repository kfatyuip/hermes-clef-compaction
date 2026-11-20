"""Regressions for audit1 findings F1-F5 (agent-executable docs + dry-run gate).

No real Clef API call and no real state.db is touched: ``requests.post`` is
monkeypatched and every source/output lives under ``tmp_path``. The ``_gate``
helper is the executable form of the SKILL.md dry-run -> apply gate checklist.
"""

from __future__ import annotations

import json

from compaction_engine import HermesCompaction, build_parser, main
from config import ALGORITHM_VERSION, build_config, result_config
from hashing import compute_cache_signature, compute_config_hash

import relevance_filter


def _fake_clef(monkeypatch, calls, *, scorer=None):
    """Patch requests.post with a deterministic Clef scorer.

    ``scorer(question) -> noul``; ``question["instructions"]`` carries the
    message content, matching the real chunk payload shape used in tests.
    """
    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append(json)
        questions = (json or {}).get("questions", {})
        answers = {}
        for qid, q in questions.items():
            score = scorer(q) if scorer else 0.9
            answers[qid] = {"noul": score}
        return type("R", (), {
            "status_code": 200,
            "text": "",
            "json": lambda self: {
                "success": True,
                "result": {"answers": answers},
            },
        })()

    monkeypatch.setattr(relevance_filter.requests, "post", fake_post)


def _write_ctx(tmp_path, contents):
    ctx = tmp_path / "ctx"
    ctx.mkdir(exist_ok=True)
    ctx.joinpath("ctx.json").write_text(
        json.dumps({"messages": contents}), encoding="utf-8"
    )
    return ctx


def _cfg(tmp_path, ctx, **overrides):
    return build_config({
        "cache_dir": str(tmp_path / "cache"),
        "context_dir": str(ctx),
        "state_db": str(tmp_path / "state.db"),
        "throttle_interval": 0,
        "cloudflare_account_id": "acct",
        "cloudflare_auth_token": "token",
        "api_max_retries": 1,
        "retry_base_delay": 0,
        "hard_preserve_recent": 0,
        "chunk_size": 10,
        **overrides,
    })


def _gate(result, *, mode):
    """Executable form of the SKILL.md dry-run -> apply gate checklist."""
    m = result["metadata"]
    checks = {
        "fresh_run": m.get("cached") is not True,  # cache hits cannot be gated
        "validation_ok": m.get("validation_ok") is True,
        "no_conservative_fallback": m.get("conservative_retain_all") == 0,
        "compressed": result["compression_ratio"] < 1.0,
        "semantic_summary": "external semantic summary" in m.get("filter_method", ""),
        "no_malformed": m.get("filter_summary", {}).get("malformed_answers", 0) == 0,
        "success": m.get("success") is True,
    }
    if mode == "recreate":
        checks["reconstructed"] = m.get("database_status") == "reconstructed"
    else:
        checks["no_db_write"] = m.get("database_status") in ("dry_run", "disabled")
    return checks


# ---------------------------------------------------------------------------
# F1 -- grouping_mode must have a CLI entry (no CLI flag existed before).
# ---------------------------------------------------------------------------
def test_cli_cache_dir_reaches_config(monkeypatch):
    """--cache-dir <empty dir> is the documented way to force a fresh run."""
    captured = {}

    class FakeCompactor:
        def __init__(self, config):
            captured["config"] = build_config(config)

        def run(self):
            return {"messages": [], "metadata": {}}

    monkeypatch.setattr("compaction_engine.HermesCompaction", FakeCompactor)
    assert main(["--cache-dir", "/tmp/fresh-cache"]) == 0
    assert captured["config"]["cache_dir"] == "/tmp/fresh-cache"


def test_parser_grouping_mode_default_and_choices():
    parser = build_parser()
    assert parser.parse_args([]).grouping_mode == "group"
    assert parser.parse_args(["--grouping-mode", "individual"]).grouping_mode == "individual"
    try:
        parser.parse_args(["--grouping-mode", "bogus"])
    except SystemExit:
        pass  # argparse rejects values outside choices
    else:
        raise AssertionError("invalid grouping_mode must be rejected by argparse")


def test_cli_grouping_mode_reaches_config(monkeypatch):
    captured = {}

    class FakeCompactor:
        def __init__(self, config):
            captured["config"] = build_config(config)

        def run(self):
            return {"messages": [], "metadata": {}}

    monkeypatch.setattr("compaction_engine.HermesCompaction", FakeCompactor)
    assert main(["--grouping-mode", "individual"]) == 0
    assert captured["config"]["grouping_mode"] == "individual"
    assert main([]) == 0
    assert captured["config"]["grouping_mode"] == "group"


# ---------------------------------------------------------------------------
# F3 -- no-summary retain-all is signalled ONLY by filter_method.
# ---------------------------------------------------------------------------
def test_no_summary_retain_all_signal(monkeypatch, tmp_path):
    ctx = _write_ctx(tmp_path, [
        {"role": "user", "content": f"m{i}", "timestamp": 1000.0 + i}
        for i in range(4)
    ])
    calls = []
    _fake_clef(monkeypatch, calls)
    r = HermesCompaction(_cfg(tmp_path, ctx, dry_run=True)).run()
    m = r["metadata"]

    assert r["compression_ratio"] == 1.0
    assert calls == []  # no Clef call in retain-all mode
    assert m["conservative_retain_all"] == 0  # that flag is D2 structural only
    assert "no summary (conservative retain-all)" in m["filter_method"]
    assert m["filter_summary"]["api_calls"] == 0

    checks = _gate(r, mode="dry_run")
    assert checks["semantic_summary"] is False
    assert checks["compressed"] is False
    assert all(checks.values()) is False  # gate must reject this result


# ---------------------------------------------------------------------------
# F4 -- --output JSON exposes every gate field in one place.
# ---------------------------------------------------------------------------
def test_output_json_exposes_gate_fields(monkeypatch, tmp_path):
    ctx = _write_ctx(tmp_path, [
        {"role": "user", "content": f"m{i}", "timestamp": 1000.0 + i}
        for i in range(4)
    ])
    calls = []
    _fake_clef(monkeypatch, calls)
    out = tmp_path / "result.json"
    r = HermesCompaction(_cfg(
        tmp_path, ctx, dry_run=True,
        external_summary="semantic summary", output=str(out),
    )).run()

    for key in ("validation_ok", "success", "conservative_retain_all",
                "database_status", "filter_method", "filter_summary",
                "retention_breakdown"):
        assert key in r["metadata"], key
    assert "compression_ratio" in r
    assert "malformed_answers" in r["metadata"]["filter_summary"]

    parsed = json.loads(out.read_text(encoding="utf-8"))
    assert parsed["metadata"]["filter_summary"]["malformed_answers"] == 0
    assert parsed["metadata"]["database_status"] == "dry_run"


# ---------------------------------------------------------------------------
# F5 -- success criteria are mode-specific; gate distinguishes dry-run vs apply.
# ---------------------------------------------------------------------------
def test_gate_accepts_real_semantic_compression(monkeypatch, tmp_path):
    ctx = _write_ctx(tmp_path, [
        {"role": "user", "content": ("drop" if i % 2 else f"keep{i}"),
         "timestamp": 1000.0 + i}
        for i in range(6)
    ])
    calls = []
    _fake_clef(
        monkeypatch, calls,
        scorer=lambda q: 0.1 if "drop" in q["instructions"] else 0.9,
    )
    r = HermesCompaction(_cfg(
        tmp_path, ctx, dry_run=True,
        external_summary="semantic summary", grouping_mode="individual",
    )).run()

    assert r["compression_ratio"] < 1.0
    assert all(_gate(r, mode="dry_run").values()), _gate(r, mode="dry_run")


def test_gate_mode_distinguishes_dry_run_from_recreate(monkeypatch, tmp_path):
    ctx = _write_ctx(tmp_path, [
        {"role": "user", "content": f"m{i}", "timestamp": 1000.0 + i}
        for i in range(4)
    ])
    calls = []
    _fake_clef(monkeypatch, calls)
    r = HermesCompaction(_cfg(
        tmp_path, ctx, dry_run=True, external_summary="semantic summary",
    )).run()

    # A dry-run result is a success in its own mode but must never satisfy the
    # --recreate gate (which requires database_status == "reconstructed").
    assert r["metadata"]["database_status"] == "dry_run"
    assert _gate(r, mode="dry_run")["no_db_write"] is True
    assert _gate(r, mode="recreate")["reconstructed"] is False


# ---------------------------------------------------------------------------
# A4 -- grouping_mode must change cache identity (else modes share entries).
# ---------------------------------------------------------------------------
def test_grouping_mode_changes_cache_identity():
    cfg_group = build_config({"grouping_mode": "group"})
    cfg_individual = build_config({"grouping_mode": "individual"})
    hash_group = compute_config_hash(result_config(cfg_group))
    hash_individual = compute_config_hash(result_config(cfg_individual))
    assert hash_group != hash_individual

    common = dict(
        session_id="s", messages_hash="m", summary_hash="sm",
        threshold=0.3, algorithm_version=ALGORITHM_VERSION,
    )
    sig_group = compute_cache_signature(config_hash=hash_group, **common)
    sig_individual = compute_cache_signature(config_hash=hash_individual, **common)
    assert sig_group != sig_individual


# ---------------------------------------------------------------------------
# A1 -- a cache hit exposes the full gate field set but is never gateable.
# ---------------------------------------------------------------------------
def test_cache_hit_exposes_gate_fields_but_is_not_gateable(monkeypatch, tmp_path):
    ctx = _write_ctx(tmp_path, [
        {"role": "user", "content": ("drop" if i % 2 else f"keep{i}"),
         "timestamp": 1000.0 + i}
        for i in range(6)
    ])
    calls = []
    _fake_clef(
        monkeypatch, calls,
        scorer=lambda q: 0.1 if "drop" in q["instructions"] else 0.9,
    )
    cfg = _cfg(
        tmp_path, ctx, external_summary="semantic summary",
        grouping_mode="individual",
    )

    first = HermesCompaction(cfg).run()
    assert first["metadata"]["cached"] is False
    assert calls, "first run must call Clef"

    calls.clear()
    second = HermesCompaction(cfg).run()
    m = second["metadata"]
    assert m["cached"] is True
    assert calls == [], "cache hit must not call Clef"

    # Same field set as a fresh run: the §5 gate must not KeyError, and the
    # reconstructed values must be truthful (not .get() defaults).
    for key in ("validation_ok", "success", "conservative_retain_all",
                "database_status", "filter_method", "filter_summary",
                "retention_breakdown"):
        assert key in m, key
    assert m["filter_summary"]["malformed_answers"] == 0
    assert m["validation_ok"] is True
    assert m["success"] is True

    checks = _gate(second, mode="dry_run")
    assert checks["fresh_run"] is False        # gate refuses cached results
    assert all(v for k, v in checks.items() if k != "fresh_run"), checks


def test_cache_hit_rebuilds_filter_summary_from_flat_report_fields():
    """Older cache reports lack the nested filter_summary key."""
    legacy_report = {
        "api_calls": 2, "api_failures": ["boom"],
        "missing_answers": 1, "malformed_answers": 3,
        "validation": {"ok": True, "issues": []},
        "database_status": "disabled",
        "retention_breakdown": {"conservative_retain_all": 0},
    }
    fields = HermesCompaction._gate_fields_from_report(legacy_report)
    assert fields["filter_summary"]["malformed_answers"] == 3
    assert fields["filter_summary"]["missing_answers"] == 1
    assert fields["filter_summary"]["failures"] == ["boom"]
    assert fields["validation_ok"] is True
    assert fields["success"] is True
    assert fields["database_status"] == "disabled"
    assert fields["conservative_retain_all"] == 0
