"""Regressions for audit3 R3-1..R3-7: each blind-reader doc claim is executable.

No live Clef credentials and no real state.db are touched: ``requests.post`` is
monkeypatched and every source/output lives under ``tmp_path``.
"""

from __future__ import annotations

import json

import relevance_filter
from cache_manager import CACHE_INVALID, CacheManager
from compaction_engine import HermesCompaction, build_parser, main
from config import build_config
from context_loader import normalize_order
from relevance_filter import parse_answers


def _write_ctx(tmp_path, contents, **extra):
    ctx = tmp_path / "ctx"
    ctx.mkdir(exist_ok=True)
    ctx.joinpath("ctx.json").write_text(
        json.dumps({"messages": contents, **extra}), encoding="utf-8"
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
        "chunk_size": 10,
        **overrides,
    })


def _fake_clef(monkeypatch, calls, *, scorer=None):
    """Fake ``requests.post`` answering with the SKILL §6 documented shape."""
    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append(json)
        questions = (json or {}).get("questions", {})
        answers = {qid: {"noul": (scorer(q) if scorer else 0.9)} for qid, q in questions.items()}
        return type("R", (), {
            "status_code": 200,
            "text": "",
            "json": lambda self: {"answers": answers},
        })()

    monkeypatch.setattr(relevance_filter.requests, "post", fake_post)


def _gate_compressed(result):
    """The single §5 check this round cares about."""
    return result["compression_ratio"] < 1.0


# ---------------------------------------------------------------------------
# R3-1 -- the documented minimal ctx.json runs, batch ignores session identity.
# ---------------------------------------------------------------------------
def test_minimal_ctx_json_runs_and_reports_batch_no_session(monkeypatch, tmp_path):
    ctx = _write_ctx(
        tmp_path,
        [
            {"role": "user", "content": "q0"},
            {"role": "assistant", "content": "a0"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
        ],
        session_id="ignored-top-level",  # top-level session_id is deliberately ignored
    )
    calls = []
    _fake_clef(monkeypatch, calls)

    result = HermesCompaction(_cfg(
        tmp_path, ctx, dry_run=True, external_summary="semantic summary",
        hard_preserve_recent=0,
    )).run()

    assert "error" not in result["metadata"]
    assert result["session_id"] == "batch-no-session"
    assert calls, "a real summary must trigger a Clef call"


# ---------------------------------------------------------------------------
# R3-2 -- the documented mock answer shapes, ids, and the `noul` field name.
# ---------------------------------------------------------------------------
def test_documented_answer_shapes_parse_but_score_key_fails_safe():
    expected = ["unit_0", "unit_1"]

    bare = parse_answers({"answers": {"unit_0": 0.95, "unit_1": 0.05}}, expected)
    assert bare["unit_0"].status == "ok" and bare["unit_0"].score == 0.95
    assert bare["unit_1"].status == "ok"

    nested = parse_answers(
        {"answers": {"unit_0": {"noul": 0.95}, "unit_1": {"noul": 0.05}}}, expected
    )
    assert all(a.status == "ok" for a in nested.values())

    wrong_key = parse_answers(
        {"answers": {"unit_0": {"score": 0.95}, "unit_1": {"score": 0.05}}}, expected
    )
    assert all(a.status == "malformed" for a in wrong_key.values())


def test_question_ids_match_documented_group_and_individual_prefixes(monkeypatch, tmp_path):
    messages = [
        {"role": "user", "content": f"m{i}", "timestamp": 1000.0 + i}
        for i in range(4)
    ]
    calls = []
    _fake_clef(monkeypatch, calls)

    # Separate directories so the two modes never share a cache entry.
    g_dir = tmp_path / "g"; g_dir.mkdir(exist_ok=True)
    i_dir = tmp_path / "i"; i_dir.mkdir(exist_ok=True)
    g_dir.joinpath("ctx.json").write_text(json.dumps({"messages": messages}), encoding="utf-8")
    i_dir.joinpath("ctx.json").write_text(json.dumps({"messages": messages}), encoding="utf-8")

    HermesCompaction(_cfg(
        tmp_path, g_dir, dry_run=True, external_summary="summary",
        cache_dir=str(tmp_path / "cache-g"), grouping_mode="group",
    )).run()
    HermesCompaction(_cfg(
        tmp_path, i_dir, dry_run=True, external_summary="summary",
        cache_dir=str(tmp_path / "cache-i"), grouping_mode="individual",
    )).run()

    assert len(calls) == 2
    assert all(k.startswith("unit_") for k in calls[0]["questions"]), calls[0]["questions"]
    assert all(k.startswith("msg_") for k in calls[1]["questions"]), calls[1]["questions"]


# ---------------------------------------------------------------------------
# R3-3 -- `invalid` cache carries the documented reason.
# ---------------------------------------------------------------------------
def test_incomplete_cache_reports_documented_reason(tmp_path):
    cm = CacheManager(str(tmp_path / "cache"))
    sig = "abcdef123456"
    d = cm.root / sig[:2] / sig
    d.mkdir(parents=True)
    d.joinpath("report.json").write_text("{}", encoding="utf-8")  # no metadata/compacted

    with cm:
        lookup = cm.get(sig, {})

    assert lookup.status == CACHE_INVALID
    assert lookup.reason == "incomplete_cache_missing_files"


# ---------------------------------------------------------------------------
# R3-4 -- dry-runs never hit; only a prior non-dry run seeds a hit.
# ---------------------------------------------------------------------------
def test_dry_runs_never_hit_and_only_non_dry_run_seeds_a_hit(monkeypatch, tmp_path):
    ctx = _write_ctx(tmp_path, [
        {"role": "user", "content": ("drop" if i % 2 else f"keep{i}"),
         "timestamp": 1000.0 + i}
        for i in range(6)
    ])
    calls = []
    _fake_clef(monkeypatch, calls, scorer=lambda q: 0.05 if "drop" in q["instructions"] else 0.9)
    dry_cfg = _cfg(
        tmp_path, ctx, dry_run=True, external_summary="summary",
        grouping_mode="individual",
    )

    first = HermesCompaction(dry_cfg).run()
    second = HermesCompaction(dry_cfg).run()
    assert first["metadata"]["cached"] is False
    assert first["metadata"]["cache_status"] == "miss"
    assert second["metadata"]["cached"] is False
    assert second["metadata"]["cache_status"] == "invalid"  # never "hit"

    seed = HermesCompaction(_cfg(
        tmp_path, ctx, dry_run=False, external_summary="summary",
        grouping_mode="individual",
    )).run()
    assert seed["metadata"]["cached"] is False

    replay = HermesCompaction(dry_cfg).run()
    assert replay["metadata"]["cached"] is True
    assert replay["metadata"]["cache_status"] == "hit"


# ---------------------------------------------------------------------------
# R3-5 -- --diagnose is a documented, read-only parser mode.
# ---------------------------------------------------------------------------
def test_diagnose_flag_is_read_only_and_reports_health(tmp_path, capsys):
    parser = build_parser()
    assert parser.parse_args(["--diagnose"]).diagnose is True

    ctx = _write_ctx(tmp_path, [
        {"role": "user", "content": f"m{i}", "timestamp": 1000.0 + i}
        for i in range(4)
    ])
    cache = tmp_path / "untouched-cache"

    rc = main(["--diagnose", "--context-dir", str(ctx), "--cache-dir", str(cache)])
    assert rc == 0
    report = json.loads(capsys.readouterr().out)
    assert report["healthy"] is True
    assert report["message_count"] == 4
    assert not cache.exists(), "--diagnose must not create a cache directory"


# ---------------------------------------------------------------------------
# R3-6 -- hard_preserve_recent=2 makes <=4 messages un-compressible.
# ---------------------------------------------------------------------------
def test_four_messages_with_high_scores_still_fail_compressed_gate(monkeypatch, tmp_path):
    ctx = _write_ctx(tmp_path, [
        {"role": "user", "content": f"m{i}", "timestamp": 1000.0 + i}
        for i in range(4)
    ])
    calls = []
    _fake_clef(monkeypatch, calls, scorer=lambda q: 0.99)

    result = HermesCompaction(_cfg(
        tmp_path, ctx, dry_run=True, external_summary="semantic summary",
        hard_preserve_recent=2, grouping_mode="individual",
    )).run()

    assert calls, "Clef was called -- this is not a no-summary retain-all"
    assert "external semantic summary" in result["metadata"]["filter_method"]
    assert result["compression_ratio"] == 1.0
    assert _gate_compressed(result) is False


# ---------------------------------------------------------------------------
# R3-7 -- numeric timestamps are what preserve real chronological order.
# ---------------------------------------------------------------------------
def test_numeric_timestamps_drive_canonical_order():
    messages = [
        {"role": "user", "content": "z", "timestamp": 3},
        {"role": "user", "content": "a", "timestamp": 1},
        {"role": "user", "content": "m", "timestamp": 2},
    ]
    assert [m["content"] for m in normalize_order(messages)] == ["a", "m", "z"]
