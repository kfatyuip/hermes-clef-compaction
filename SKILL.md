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

## 1. Purpose and scope

Reduce the token footprint of the current Hermes conversation while preserving
the smallest _coherent_ set of original messages needed to continue work.

Responsibilities are deliberately separated:

| Layer                     | Responsibility                                                                        |
| ------------------------- | ------------------------------------------------------------------------------------- |
| **Agent**                 | Understand the conversation; author the semantic summary; verify the candidate result |
| **Cloudflare Clef-flash** | Score each conversation unit's relevance to the summary                               |
| **Python engine**         | Deterministic grouping, chunking, caching, persistence, recovery                      |

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

The cleanest interface passes raw text on stdin/argument:

```bash
python3 scripts/compaction_engine.py \
  --external-summary "A. Active Context: ...\nB. History: ...\n..." \
  --session-id <id> \
  --dry-run
```

Or pass a file (explicit, unambiguous):

```bash
python3 scripts/compaction_engine.py \
  --external-summary-file /path/to/summary.txt \
  --session-id <id> \
  --dry-run
```

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
| `--dry-run` | Full pipeline including Clef, but no DB write and no cache write. **Use this first to verify.** |
| `--recreate` | Actually reconstruct `state.db`. Destructive; creates a verified backup first. Default is **OFF**. |

Always run `--dry-run` and inspect the report before `--recreate`.

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
  above. Tool detection is content-based, so the guarantee also holds for the
  single-message units produced by `individual` mode.
- To drop interspersed noise **message by message**, set
  `grouping_mode=individual`: each message is scored as its own unit, so an
  irrelevant message can be discarded without taking its neighbours with it.
  Tool call/result pairs are still linked across units when
  `hard_preserve_tool_links=true` (content-based, fixpoint-linked).

**Clef request contract.** Workers AI uses the full model id in the URL path
and the bare model name in the JSON body. For the default model:

- URL: `POST .../accounts/$CLOUDFLARE_ACCOUNT_ID/ai/run/@cf/cloudflare/clef-flash`
- body: `{"model": "clef-flash", "state": "<summary + unit-tagged messages>", "questions": {"<unit_id>": {"type": "noul", "instructions": "..."}}}`

`state` is a **plain text field** holding the summary and unit-tagged messages.
`config["model"]` supplies the URL id; the body name is derived from it by
`clef_model_name()`, so custom `@cf/...` ids keep the same split. Only text
role/content is copied into `state`, which keeps SQLite BLOB/bytes columns out
of the JSON payload.

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

**Never report successful compaction if the database update failed** — check
`metadata.database_status == "reconstructed"` and `metadata.success == true`.

## 9. Recovery and fallback

- Pre-compaction backups: `<state.db>.precompaction.<timestamp>.bak`.
- To restore, use `DatabaseManager.restore(path)` or copy a verified backup back.
- The engine refuses to restore a corrupt backup and preserves the pre-restore DB.
- **Fallback mode** (`--fallback`): reads `state.db` directly; if no semantic summary
  is provided, uses a _labelled_ truncation summary and a conservative policy.
  Truncation is not semantic summarization — prefer providing a real summary.
- In Agent mode with **no** summary, the engine does not call Clef and retains all
  messages (never silently degrades truncation into a "semantic" result).

## 10. Batch mode

When no LLM-generated summary is available, provide a `ctx.json` file via
`--context-dir`. The selected source is explicit: `ctx.json` for batch,
`state.db` for Agent/fallback (`--fallback` or `--session-id`). Mixed sources or
mixed session IDs are rejected.

## 11. Configuration

| Parameter                  | Default | Meaning                                                                            |
| -------------------------- | ------- | ---------------------------------------------------------------------------------- |
| `threshold`                | 0.3     | Keep units with `noul >= threshold`                                                |
| `chunk_size`               | 20      | Hard upper bound on messages per Clef request                                      |
| `max_messages`             | 48      | Recent messages read from the session; overridable via `--max-messages N` CLI flag |
| `summary_max_chars`        | 8000    | Configurable summary ceiling (target 2000)                                         |
| `throttle_interval`        | 2       | Seconds between chunks (operational only)                                          |
| `recreate`                 | false   | Destructive DB write; requires explicit opt-in                                     |
| `use_external_summary`     | true    | Use the Agent-provided semantic summary                                            |
| `grouping_mode`            | group   | `group` (conversation units) or `individual`                                       |
| `hard_preserve_recent`     | 2       | Newest N messages always retained                                                  |
| `hard_preserve_tool_links` | true    | Keep tool call/result units intact                                                 |
| `api_max_retries`          | 3       | Bounded retries with exponential backoff + jitter                                  |

`grouping_mode` and `hard_preserve_tool_links` are result-relevant (part of the
cache identity) and are set in configuration, not exposed as CLI flags. See §6
for the `group` vs `individual` trade-off and the exact guarantee of
`hard_preserve_tool_links`. The algorithm version is currently **2.1.0** and is
part of every cache identity; the 2.0.0 → 2.1.0 bump intentionally invalidates
all earlier cache entries (a recomputation after upgrade is expected, not an
error).

## 12. Troubleshooting

| Symptom | Meaning / fix |
|---|---|
| `no session context found` | No `ctx.json` and no active DB session; check source/session-id |
| `external_summary_file does not exist` | Explicit file path was missing; check the path |
| `database_status: failed` | Reconstruction aborted (concurrent edit / verification); original messages preserved; inspect `database_errors` and the backup |
| `cache_status: invalid` | Cache identity/summary/version mismatch; cache was correctly ignored and recomputed |
| Many `malformed_answers` | Clef response schema changed; affected units were retained fail-safe — inspect `retention_report.json` |
| `RATE_LIMITED` / HTTP 429 | Increase `throttle_interval`; retries are bounded |
| Clef API HTTP 400 "No route for that URI" or HTTP 422 "Unsupported model" | Workers AI uses two model name formats: the URL path takes the full id (`@cf/cloudflare/clef-flash`), the JSON body's `model` field takes the bare name (`clef-flash`). See the request contract in §6 and `CLEF_MODEL_ID` / `CLEF_MODEL_NAME` / `clef_model_name()` in `config.py` |
| `TypeError: Object of type bytes is not JSON serializable` | state.db stores some columns as SQLite BLOBs (e.g. `display_identity`). The engine strips these before API calls and output, but if you see this error the sanitization may have missed a field — check the `_sanitized_fields` sidecar on retained messages |
| `broken_tool_dependency` with `max_messages` truncation | The default 48-message window can split a tool call from its result. Increase `--max-messages` to cover the full session, or accept the conservative retain-all fallback (all messages kept, ratio 1.0) |
| `tool_calls` field is a JSON string, not a list | state.db stores `tool_calls` as a JSON-encoded string. The validator parses this, but custom code reading state.db directly must `json.loads()` it before iterating |
| `ContextError: every message must be a dict` / `malformed_input` decisions | A non-dict message was supplied. As of 2.1.0 the loader rejects non-dict input with a structured error, and the filter fails safe by retaining the affected chunk (`Decision(source="malformed_input")`) instead of crashing — fix the input; do not expect a silent drop |
| Tool call/result orphan in `individual` mode | `group` mode keeps a pair in one unit; `individual` mode makes each message its own unit, so pair preservation depends on the content-based cross-unit link gated by `hard_preserve_tool_links` (default true). If it is disabled, orphans and a retain-all fallback are possible; see §6 |
| Cache reused after permuting equal/missing-timestamp messages | Canonical order is content-intrinsic (permutation-invariant) as of 2.1.0, so a permutation of the same message set maps to the same canonical order and may safely hit the cache; cache identity uses an order-sensitive fingerprint over that canonical order |

Logs: `stderr`. Machine/human reports: cache directory under
`~/.hermes/cache/hcompaction/v2/<sig>/`. No tokens or full message bodies are logged.
