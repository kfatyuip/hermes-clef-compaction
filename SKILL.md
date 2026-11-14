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
  is provided, uses a *labelled* truncation summary and a conservative policy.
  Truncation is not semantic summarization — prefer providing a real summary.
- In Agent mode with **no** summary, the engine does not call Clef and retains all
  messages (never silently degrades truncation into a "semantic" result).

## 10. Batch mode

When no LLM-generated summary is available, provide a `ctx.json` file via
`--context-dir`. The selected source is explicit: `ctx.json` for batch,
`state.db` for Agent/fallback (`--fallback` or `--session-id`). Mixed sources or
mixed session IDs are rejected.

## 11. Configuration

| Parameter | Default | Meaning |
|---|---|---|
| `threshold` | 0.3 | Keep units with `noul >= threshold` |
| `chunk_size` | 20 | Hard upper bound on messages per Clef request |
| `max_messages` | 48 | Recent messages read from the session |
| `summary_max_chars` | 8000 | Configurable summary ceiling (target 2000) |
| `throttle_interval` | 2 | Seconds between chunks (operational only) |
| `recreate` | false | Destructive DB write; requires explicit opt-in |
| `use_external_summary` | true | Use the Agent-provided semantic summary |
| `grouping_mode` | group | `group` (conversation units) or `individual` |
| `hard_preserve_recent` | 2 | Newest N messages always retained |
| `hard_preserve_tool_links` | true | Keep tool call/result units intact |
| `api_max_retries` | 3 | Bounded retries with exponential backoff + jitter |

## 12. Troubleshooting

| Symptom | Meaning / fix |
|---|---|
| `no session context found` | No `ctx.json` and no active DB session; check source/session-id |
| `external_summary_file does not exist` | Explicit file path was missing; check the path |
| `database_status: failed` | Reconstruction aborted (concurrent edit / verification); original messages preserved; inspect `database_errors` and the backup |
| `cache_status: invalid` | Cache identity/summary/version mismatch; cache was correctly ignored and recomputed |
| Many `malformed_answers` | Clef response schema changed; affected units were retained fail-safe — inspect `retention_report.json` |
| `RATE_LIMITED` / HTTP 429 | Increase `throttle_interval`; retries are bounded |

Logs: `stderr`. Machine/human reports: cache directory under
`~/.hermes/cache/hcompaction/v2/<sig>/`. No tokens or full message bodies are logged.
