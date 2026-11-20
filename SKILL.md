---
name: "hermes-clef-compaction"
description: "Compress the active Hermes session context using an Agent-generated semantic summary plus Cloudflare Clef-flash relevance filtering. Conservative, reversible, session-isolated."
version: 2.0
license: MIT
author: "team"
category: workflow
phase: dag-delivery
role: tracker
requested: "hermes-clef-compaction"
---

# Hermes Clef Compaction — Operational Contract

This document is a **contract for the Agent**, not merely a description. Reading it
does not by itself execute anything. When this skill is invoked in an active
Hermes session, the Agent **must perform the semantic steps below itself** before
delegating deterministic work to the Python engine.

## 0. Read this first — invocation decision table (safety)

The most dangerous mistake is pointing a **drill** at the real database.
`--session-id` and `--fallback` always read `state.db`, whose default path is the
**real** `~/.hermes/state.db`. `--recreate` rewrites that file. Never use either
for learning, testing, or an exploratory dry-run.

| Goal | Source flags (pick exactly one) | Summary | Write mode | Flags to OMIT |
|---|---|---|---|---|
| **Safe drill / verify / learn** (no real DB) | `--context-dir <scratch>/ctx` (a dir containing `ctx.json`) | `--external-summary "..."` | `--dry-run --output <scratch>/result.json` | `--session-id`, `--fallback`, `--recreate` |
| **Real compaction of a live session** | `--session-id <id>` (reads the real `~/.hermes/state.db`) | `--external-summary "..."` | `--dry-run` first, then `--recreate` | `--context-dir`, `--fallback` |
| **Batch over a prepared ctx.json** (no DB at all) | `--context-dir <dir>` containing `ctx.json` | `--external-summary "..."` | no `--recreate` | `--session-id`, `--fallback` |

For both `--context-dir` rows, `ctx.json` must hold a `messages` list of
`{role, content}` dicts (optional numeric `timestamp`); §4 has a copy-paste
example. A `session_id` at the top level of batch input is ignored.

Two hard rules:

- **No `--session-id` / `--fallback` in a drill.** Even without `--recreate`,
  those flags read the real `~/.hermes/state.db`. If your only input is a scratch
  `ctx.json`, use `--context-dir <scratch>/ctx`.
- **`--recreate` writes.** It is destructive and valid only together with a real
  `--session-id` (or `--fallback`). A verified backup is created first, but treat
  it as a real database write and pass the gate in §5 before it.

Copy-paste safe drill:

```bash
python3 scripts/compaction_engine.py \
  --context-dir <scratch>/ctx \
  --external-summary "A. Active Context: ...\nB. History: ..." \
  --dry-run --output <scratch>/result.json
```

If you see `No session context found` or `Messages have no session_id; refusing
to guess one`, you picked the wrong source (see §12). Fix the source — do not add
flags blindly.

## 1. Purpose and scope

Reduce the token footprint of the current Hermes conversation while preserving
the smallest *coherent* set of original messages needed to continue work.

Responsibilities are deliberately separated:

| Layer | Responsibility |
|---|---|
| **Agent** | Understand the conversation; author the semantic summary; verify the candidate result |
| **Cloudflare Clef-flash** | Score each conversation unit's relevance to the summary |
| **Python engine** | Deterministic grouping, chunking, caching, persistence, recovery |

The summary is a **relevance reference for selecting original messages** — never a
replacement for them. The engine always evaluates original messages against it.

## 2. When to invoke

- The current session is long enough that token cost or context limits matter.
- The user asks to compact / summarize / "free up context".
- As a node in a DAG workflow that needs a smaller context to continue.

Do **not** invoke for trivial short conversations, and never invoke as a
substitute for actually understanding the task.

## 3. Agent semantic summarization (MANDATORY in Agent mode)

When invoked in a live session, the Agent must:

1. Read and understand the current session and its recent history.
2. Identify the current objective and the overall conversation structure.
3. Produce a concise **semantic summary** covering the five dimensions below.
4. Pass it to the engine as `--external-summary`.
5. Run the relevance filtering process.
6. **Verify the output before any destructive database operation.**

The Agent must not merely summarize the last few messages. A long-running
project's earlier architectural decisions may still matter even when the latest
messages discuss an unrelated bug. Preserve the relationship between the active
task and relevant historical knowledge.

### Required summary dimensions

**A. Active Context** — current task, objectives, stage of progress, relevant
technical details, current problems and unresolved questions.

**B. Important Historical Context** — previous decisions affecting the current
task, key arguments and reasoning, rejected approaches and why, dependencies on
earlier discussion.

**C. Persistent Constraints** — user requirements, technical constraints,
explicit preferences, decisions that must not be silently reversed.

**D. Unresolved Issues** — pending tasks, known bugs, incomplete investigations,
questions awaiting confirmation.

**E. Preservation Priorities** — information that must not be lost, information
that should preferably remain, information that can safely be removed.

### Summary rules

- Prefer precise facts over vague descriptions; preserve important distinctions.
- Preserve reasons behind major decisions; distinguish confirmed facts from hypotheses.
- Do not invent conclusions; do not repeat the same information.
- Do not treat old information as invalid merely because it was not mentioned recently.
- Target ~2000 characters as an **initial target, not a hard limit**. If the context
  genuinely needs more, allow more, bounded by `--summary-max-chars` and the Clef
  token budget. Conciseness must not sacrifice important distinctions.

## 4. External summary interface

Pick the source using the §0 decision table first. For a **safe drill**, pass a
scratch `ctx.json` directory — the examples below deliberately use no
`--session-id` / `--fallback`:

**`ctx.json` structure (minimal, copy-paste).** The directory you pass to
`--context-dir` must contain a file named exactly `ctx.json`. Put the messages
in a `messages` list; each message is a dict with at least `role` and `content`:

```json
{
  "messages": [
    {"role": "user", "content": "First question"},
    {"role": "assistant", "content": "First answer"}
  ]
}
```

Add an optional numeric `timestamp` (e.g. Unix epoch seconds) to each message
to preserve real chronological order — see the caveat below. Every entry in
`messages` must be a dict; a bare string or number is rejected with
`ContextError: <source> messages[N] must be a dict, got <type>` (for example,
`context file <path>/ctx.json messages[1] must be a dict, got str`).

Batch mode ignores session identity: a `session_id` at the top level of
`ctx.json` has no effect, and with no per-message `session_id` the result
reports `session_id: "batch-no-session"`. This is normal — do not try to
"fix" it by adding `--session-id`.

Caveat — write a numeric `timestamp` on every batch message. Without timestamps
the engine orders messages by their content instead of your list order, so
user/assistant pairs can be scattered across units and the whole-unit
atomicity promised in §6 may not hold. Use the real timestamps from the
conversation you are compacting.

```bash
python3 scripts/compaction_engine.py \
  --context-dir <scratch>/ctx \
  --external-summary "A. Active Context: ...\nB. History: ...\n..." \
  --dry-run --output <scratch>/result.json
```

Or pass a file (explicit, unambiguous):

```bash
python3 scripts/compaction_engine.py \
  --context-dir <scratch>/ctx \
  --external-summary-file /path/to/summary.txt \
  --dry-run --output <scratch>/result.json
```

**For a real compaction only**, replace `--context-dir <scratch>/ctx` with
`--session-id <id>` (which reads the real `~/.hermes/state.db`) and follow §0 +
§5. Never paste a `--session-id` example into a drill.

The engine never guesses: long summary text that resembles a path is treated as
text, and a missing explicit summary file is a hard error.

## 5. Engine invocation lifecycle

The engine executes this fixed pipeline (do not skip verification steps):

```
load session (single, explicit source; chronological order)
  → resolve/validate semantic summary
  → compute content-addressed cache identity
  → check cache (validate every identity component)
  → group messages into conversation units
  → build token-safe chunks
  → call Clef (bounded retries; fail-safe)
  → validate each answer
  → apply retention (hard-preserve ∪ Clef-relevant ∪ fail-safe)
  → structural verification
  → [if --recreate] verified backup → transactional reconstruction → verify
  → save cache + report
```

### Safe execution modes (choose explicitly)

| Flag | Effect |
|---|---|
| `--analyze-only` | Load/group/report only. No API, no DB, no cache. |
| `--diagnose` | Read-only structural health check of the loaded message set; prints a JSON report. No Clef call, no DB write, no cache write. Run it before scoring to check session mixing, orphan tool links, oversized/BLOB fields, duplicate ids and order inversions. |
| `--dry-run` | Full pipeline including Clef, but no DB write and no cache *result* write. The report file is still written under the cache directory — use `--output` for a deterministic result JSON. **Use this first to verify.** |
| `--recreate` | Actually reconstruct `state.db`. Destructive; creates a verified backup first. Default is **OFF**. |

Always run `--dry-run` and pass the gate below before `--recreate`.

A dry-run never seeds a cache hit: it writes a report but no replayable cache
result, so two identical dry-runs show `cache_status: miss` then `invalid`,
never `hit`. A hit (`metadata.cached == true`) can only follow a **previous
non-dry-run** on the same input that wrote a replayable result; that is the only
way the `exit 2` branch below becomes reachable. If your goal is to gate a
**fresh** computation, pass an empty cache directory from the start
(`--cache-dir <scratch>/fresh-cache`) so no prior result can be replayed.

### Dry-run → apply gate (run before every `--recreate`)

Write the dry-run result to a file and check it with a machine gate — do not
eyeball it. All fields live in the `--output` JSON (`filter_summary` and
`retention_breakdown` are in the result JSON for exactly this gate):

```bash
python3 scripts/compaction_engine.py ... --dry-run --output <scratch>/result.json
python3 - <scratch>/result.json <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
m = r["metadata"]
# A cache hit replays an EARLIER run; it is not evidence about this input.
if m.get("cached") is True:
    print("CACHED RESULT — cannot gate. Re-run with --cache-dir <empty dir>.")
    sys.exit(2)
checks = {
    "validation_ok": m.get("validation_ok") is True,
    "no_conservative_retain_all": m.get("conservative_retain_all") == 0,
    "actually_compressed": r["compression_ratio"] < 1.0,
    "semantic_summary_used": "external semantic summary" in m.get("filter_method", ""),
    "no_malformed_answers": m.get("filter_summary", {}).get("malformed_answers", 0) == 0,
    "success": m.get("success") is True,
    "no_db_write_in_dry_run": m.get("database_status") in ("dry_run", "disabled"),
}
print(checks)
sys.exit(0 if all(checks.values()) else 1)   # non-zero = do NOT apply
PY
```

**Cache hits cannot be gated.** If `metadata.cached == true` (equivalently
`cache_status: hit`), the engine replayed a stored result instead of computing
your current input — even though the gate fields are present, that result is not
valid evidence. Re-run the **same** dry-run with an empty cache directory
(`--cache-dir <scratch>/fresh-cache`) and gate the fresh result. The `.get()`
defaults above only guard against a missing field; they never make a cached
result pass. Exit codes: `0` = apply, `1` = a check failed (fix input and rerun),
`2` = cached result (re-run with a fresh cache dir).

Take the `exit 2` branch only when `metadata.cached == true`. If it is false or
absent, do not stop here — continue through the seven checks above.

| Field | Pass value | Meaning when it fails |
|---|---|---|
| `metadata.validation_ok` | `true` | Structural verification failed; engine already retained all (see §7) |
| `metadata.conservative_retain_all` | `0` | `1` = structural retain-all fallback (D2), not a semantic result |
| `compression_ratio` (top level) | `< 1.0` | `== 1.0` means nothing was dropped |
| `metadata.filter_method` | contains `external semantic summary` | `no summary (conservative retain-all)` or `fallback truncation summary` means **no semantic filtering occurred** — add `--external-summary` and rerun (§3/§9) |
| `metadata.filter_summary.malformed_answers` | `0` | Malformed Clef answers were retained fail-safe; fix before applying |
| `metadata.success` | `true` | Overall run failed (e.g. DB write error) |
| `metadata.database_status` | `dry_run` / `disabled` in dry-run; `reconstructed` in `--recreate` | See mode rule below |

**Small-session caveat.** `hard_preserve_recent=2` (default) keeps the newest
messages, but length alone does **not** force `ratio == 1.0`. With an external
summary and low Clef scores on older messages, a 3-message `individual` run
reached `0.667` and a 4-message run reached `0.5`. `ratio == 1.0` means every
unit scored at or above the threshold, so nothing was judged droppable — not
that a hard length floor blocked compression. For a drill, prepare **at least
4–6 messages** (12 is ideal) so a low-score unit has room to be dropped.

**Mode-specific success rule (never mix these up):**

- `--recreate`: require `database_status == "reconstructed"` **and** `success ==
  true`, and confirm `backup_path` is set. `database_status == "failed"` means
  the write was rolled back — never report success.
- `--dry-run` / batch (`ctx.json`): no DB write, so `database_status` is
  `dry_run` or `disabled`. Do **not** call it a failure merely because it is not
  `reconstructed`; judge it with the gate table and the retained-set JSON
  (`messages`).

If the gate fails, fix the input / `--external-summary` and rerun the dry-run.
Only when every check passes, do the real apply from §0 (`--session-id <id>
--recreate`) and re-check with the `--recreate` rule above.

## 6. Filtering and retention rules

- Conversation units (user+assistant, tool call+result, follow-ups) are evaluated
  as wholes so a retained answer never loses its question or tool context.
- Default threshold is **0.3** and is configurable via `--threshold [0,1]`. It
  means "keep a unit whose `noul` score ≥ threshold" — **not** "retain 30% of
  messages". The score's calibration is not independently verified; treat it as
  a relative relevance signal.
- Two-level retention: **hard-preserved** (newest messages, explicit `[IMPORTANT]`
  markers, tool-link integrity) ∪ **Clef-relevant** ∪ **fail-safe retained**.
- Missing, malformed, non-finite, out-of-range or unmatched answers are **always
  retained** (fail-safe) — never silently dropped. On API failure, affected units
  are retained and the failure is recorded.

### Grouping modes and the cost of atomicity

- `grouping_mode=group` (default) scores each conversation unit **as a whole**,
  so a retained answer never loses its question or tool context. The trade-off:
  when one message in a unit is relevant, the entire unit is retained — any
  interspersed noise inside that unit is retained with it. That noise is the
  price of unit-level atomicity, not a scoring error; do not expect `group`
  mode to surgically drop noisy messages that sit inside a relevant unit.
- `hard_preserve_tool_links=true` guarantees that a tool call and its result
  enter and leave **together** (atomicity). It does **not** guarantee that a
  unit judged irrelevant must be retained: such a unit may still be dropped
  as a whole, subject only to the fail-safe and structural-verification rules
  above. The guarantee also holds for the single-message units produced by
  `individual` mode: a call and its result still leave together.
- To drop interspersed noise **message by message**, set
  `grouping_mode=individual`: each message is scored as its own unit, so an
  irrelevant message can be discarded without taking its neighbours with it.
  Tool call/result pairs are still kept together across units when
  `hard_preserve_tool_links=true` (on by default) — switching to `individual`
  never splits a call from its result.

**How to run `individual` mode (copy-paste):** it is a normal CLI flag now —

```bash
python3 scripts/compaction_engine.py \
  --context-dir <scratch>/ctx \
  --external-summary "<semantic summary>" \
  --grouping-mode individual \
  --dry-run --output <scratch>/result.json
```

There is no `--hard-preserve-tool-links` flag; for that config-only option use
Python (also the form to use inside another script):

```python
import sys; sys.path.insert(0, "scripts")   # make local modules importable from the repo root
from compaction_engine import HermesCompaction
from config import build_config

compactor = HermesCompaction(build_config({
    "cache_dir": "<scratch>/cache",        # keep outputs off the real cache
    "context_dir": "<scratch>/ctx",        # or "session_id": "<id>" for a REAL session
    "external_summary": "<semantic summary>",
    "grouping_mode": "individual",
    "hard_preserve_tool_links": True,      # keep tool call/result pairs linked
    "dry_run": True,
}))
result = compactor.run()                   # gate the result JSON exactly as in §5
```

From the repo root, `scripts/` is not on `sys.path` — the first line makes the
local imports work. Equivalent: run the script with `PYTHONPATH=scripts python3
my_script.py`. For an end-to-end test without live credentials, patch
`relevance_filter.requests.post`: the engine builds `RelevanceFilter` internally,
so there is no `call_api` argument to inject through `HermesCompaction`.

Here is a minimal fake response — exactly what your patched `post` must return:

```python
class FakeResponse:
    status_code = 200
    def json(self):
        return {"answers": {"unit_0": 0.95, "unit_1": 0.05}}
```

The keys under `answers` must be exactly the question ids from the request
payload you inspected: `unit_N` in `group` mode, `msg_N` in `individual` mode.
Each value is a relevance number in `[0, 1]`, either a bare float or an object
`{"noul": <float>}`. The field name is **`noul`**: using `score` instead
(`{"score": 0.95}`) is treated as a malformed answer, so those units are kept
fail-safe (`filter_summary.malformed_answers > 0`) and the run looks as if
nothing was compressed. Missing ids are retained fail-safe too.

## 7. Verification (before any destructive operation)

Structural verification is **mandatory** and always runs:

- retained set is an in-order subsequence of the original;
- single, consistent session identity;
- no tool result without its tool call;
- no duplicates; retained set is non-empty.

If structural verification fails, the engine falls back to retaining **all**
messages (conservative). If the Agent's `--semantic-verification` is enabled, the
Agent should additionally confirm: the current task is still understandable,
key decisions and constraints are preserved, unfinished work can continue, and
important historical dependencies did not disappear. Agent verification is
configurable; structural verification is not optional.

## 8. Database reconstruction

Only when `--recreate` is explicitly set, in fallback/`--session-id` mode, on a
real session:

1. A consistency-safe SQLite backup is created (WAL-safe, integrity-checked).
2. Only the rows that were actually read are deactivated — messages arriving
   during the Clef call remain active and visible to Hermes.
3. Reconstruction happens in one transaction; the source fingerprint is
   re-verified inside it. Concurrent edits abort the write.
4. The reconstructed session is verified before commit. Failure rolls back and
   preserves the original messages and backup.
5. The session ID is unchanged; all original columns/metadata are preserved by
   flipping `active`/`compacted` flags rather than re-inserting rows. Hot-reload
   behavior itself is a Hermes runtime property and must be confirmed in the
   target environment.

**Never report successful compaction if the database update failed.** Success is
mode-specific — apply the gate in §5:

- `--recreate`: require `metadata.database_status == "reconstructed"` **and**
  `metadata.success == true`, and confirm `backup_path` is set. `failed` means the
  write was rolled back — never call it a success.
- `--dry-run` / batch: no DB write, so `database_status` is `dry_run` or
  `disabled`. Do not report failure merely because it is not `reconstructed`;
  judge the retained set with the §5 gate instead.

## 9. Recovery and fallback

- Pre-compaction backups: `<state.db>.precompaction.<timestamp>.bak`.
- To restore, use `DatabaseManager.restore(path)` or copy a verified backup back.
- The engine refuses to restore a corrupt backup and preserves the pre-restore DB.
- **Fallback mode** (`--fallback`): reads `state.db` directly; if no semantic summary
  is provided, uses a *labelled* truncation summary and a conservative policy.
  Truncation is not semantic summarization — prefer providing a real summary.
- In Agent mode with **no** summary, the engine does not call Clef and retains all
  messages (never silently degrades truncation into a "semantic" result).

## 10. Batch mode

When no LLM-generated summary is available, provide a `ctx.json` file via
`--context-dir`. The selected source is explicit: `ctx.json` for batch,
`state.db` for Agent/fallback (`--fallback` or `--session-id`). Mixed sources or
mixed session IDs are rejected.

Give every batch message a numeric `timestamp` so the engine keeps real
chronological order; with equal or missing timestamps, ordering falls back to
message content and §6's whole-unit atomicity may not hold (see §4).

## 11. Configuration

| Parameter | Default | Meaning |
|---|---|---|
| `threshold` | 0.3 | Keep units with `noul >= threshold` |
| `chunk_size` | 20 | Hard upper bound on messages per Clef request |
| `max_messages` | 48 | Recent messages read from the session; overridable via `--max-messages N` CLI flag |
| `summary_max_chars` | 8000 | Configurable summary ceiling (target 2000) |
| `throttle_interval` | 2 | Seconds between chunks (operational only) |
| `recreate` | false | Destructive DB write; requires explicit opt-in |
| `use_external_summary` | true | Use the Agent-provided semantic summary |
| `grouping_mode` | group | `group` (conversation units) or `individual` |
| `hard_preserve_recent` | 2 | Newest N messages always retained |
| `hard_preserve_tool_links` | true | Keep tool call/result units intact |
| `api_max_retries` | 3 | Bounded retries with exponential backoff + jitter |

`hard_preserve_recent=2` only guarantees the newest messages are kept; it does
not impose a fixed length floor on `compression_ratio` (see "Small-session
caveat" in §5).

`grouping_mode` is exposed as `--grouping-mode {group,individual}` (copy-paste
recipe in §6). `hard_preserve_tool_links` has no CLI flag and is still set via
`build_config({...})` (Python recipe in §6). Both are result-relevant (part of the
cache identity). See §6 for the `group` vs `individual` trade-off and the exact
guarantee of `hard_preserve_tool_links`. The algorithm version is currently **2.1.0** and is
part of every cache identity; the 2.0.0 → 2.1.0 bump intentionally invalidates
all earlier cache entries (a recomputation after upgrade is expected, not an
error).

## 12. Troubleshooting

| Symptom | Meaning / fix |
|---|---|
| `No session context found` / `Messages have no session_id; refusing to guess one` / `Context file not found: <path>/ctx.json` / `No session id available (auto-detect failed and none supplied)` | Source-selection error (**exact strings**): no `ctx.json` under the chosen `--context-dir`, or a DB source (`--session-id` / `--fallback`) whose messages lack a usable session id. Same fix for all four — see §0: point `--context-dir` at a dir that has `ctx.json` (batch), or supply a valid `--session-id` / `--fallback` (DB), then retry |
| `external_summary_file does not exist` | Explicit file path was missing; check the path |
| `database_status: failed` | Reconstruction aborted (concurrent edit / verification); original messages preserved; inspect `database_errors` and the backup |
| `cache_status: invalid` | Cache entry incomplete, or identity/summary/version mismatched; cache was correctly ignored and recomputed. Check `cache_reason` in the report: `incomplete_cache_missing_files` means no complete replayable entry existed yet (a dry-run never writes one) — recompute is expected, not an error |
| Many `malformed_answers` | Clef response schema changed; affected units were retained fail-safe — inspect `retention_report.json` |
| `RATE_LIMITED` / HTTP 429 | Increase `throttle_interval`; retries are bounded |
| `TypeError: Object of type bytes is not JSON serializable` | state.db stores some columns as SQLite BLOBs (e.g. `display_identity`). The engine strips these before API calls and output, but if you see this error the sanitization may have missed a field — check the `_sanitized_fields` sidecar on retained messages |
| `broken_tool_dependency` with `max_messages` truncation | The default 48-message window can split a tool call from its result. Increase `--max-messages` to cover the full session, or accept the conservative retain-all fallback (all messages kept, ratio 1.0) |
| `tool_calls` field is a JSON string, not a list | state.db stores `tool_calls` as a JSON-encoded string. The validator parses this, but custom code reading state.db directly must `json.loads()` it before iterating |
| `ContextError: <source> messages[N] must be a dict, got <type>` / `malformed_input` decisions | A non-dict message was supplied. As of 2.1.0 the loader rejects non-dict input with a structured error, and the filter fails safe by retaining the affected chunk (`Decision(source="malformed_input")`) instead of crashing — fix the input; do not expect a silent drop |
| Tool call/result orphan in `individual` mode | `group` mode keeps a pair in one unit; `individual` mode makes each message its own unit, so pair preservation depends on `hard_preserve_tool_links` (default true). If it is disabled, orphans and a retain-all fallback are possible; see §6 |
| Cache reused after permuting equal/missing-timestamp messages | Canonical order is content-intrinsic (permutation-invariant) as of 2.1.0, so a permutation of the same message set maps to the same canonical order and may safely hit the cache; cache identity uses an order-sensitive fingerprint over that canonical order |

Logs: `stderr`. Machine/human reports: cache directory under
`~/.hermes/cache/hcompaction/v2/<sig>/`. No tokens or full message bodies are logged.
