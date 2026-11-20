# hermes-clef-compaction

Intelligent, **conservative and reversible** context compression for Hermes Agent
sessions. An Agent-authored semantic summary guides Cloudflare Clef-flash relevance
filtering; the Python engine handles deterministic grouping, caching and safe
persistence. The goal is not maximum compression — it is the smallest _coherent_
message set that lets Hermes keep working without losing decisions, constraints or
conversational continuity.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9+-blue.svg)

This release is a deep refactor of a monolithic 819-line script into independently
testable components, fixing critical correctness bugs:

- **Cache correctness**: cache identity now includes session + message hash +
  summary hash + threshold + config + algorithm version, all validated on read.
  A changed summary or threshold can never reuse a stale result.
- **No silent data loss**: missing/malformed/NaN/out-of-range/unmatched Clef answers
  are retained, never dropped. Bounded retries with backoff + jitter.
- **Conversation-aware filtering**: user/assistant and tool call/result units are
  kept intact; chronological order and session isolation guaranteed.
- **Database safety**: WAL-safe verified backups, transactional in-place
  reconstruction, optimistic concurrency checks, only rows actually read are
  touched, full rollback on failure.
- **Safe defaults**: destructive DB writing is OFF by default; explicit
  `--analyze-only` / `--dry-run` / `--recreate` modes; single stdout owner.

## Installation

```bash
hermes skills install https://github.com/kfatyuip/hermes-clef-compaction
# or
git clone https://github.com/kfatyuip/hermes-clef-compaction.git
```

Python 3.9+. Dependencies: `requests`. Tests need `pytest`.

```bash
export CLOUDFLARE_ACCOUNT_ID="your-account-id"
export CLOUDFLARE_AUTH_TOKEN="your-api-token"
```

## Architecture

```
scripts/
├── compaction_engine.py   # orchestration + CLI only
├── config.py              # defaults, validation, cache-relevant config
├── hashing.py             # stable, content-addressed identity
├── summary_manager.py     # summary loading/validation + conservative fallback
├── context_loader.py      # source selection, ordering, session isolation
├── conversation_grouper.py# conversation units and metadata preservation
├── chunk_manager.py       # conservative token estimation + adaptive chunking
├── cache_manager.py       # validated, atomic, versioned cache
├── relevance_filter.py    # strict Clef response validation + fail-safe
├── database_manager.py    # verified backup + transactional reconstruction
├── validator.py           # two-level retention + structural verification
└── report_generator.py    # comprehensive retention reports
tests/                     # unit + integration tests (mocked API, temp DB)
```

`compaction_engine.py` orchestrates these components; it implements none of their
logic, so each responsibility is independently testable.

## Usage

### Recommended: Agent semantic workflow

1. The Agent authors a semantic summary (see `SKILL.md` §3 — five dimensions).
2. Verify with a dry run:
   ```bash
   python3 scripts/compaction_engine.py \
     --external-summary "<semantic summary>" \
     --session-id <id> --dry-run
   ```
3. Inspect the report. If it is coherent, apply:
   ```bash
   python3 scripts/compaction_engine.py \
     --external-summary "<semantic summary>" \
     --session-id <id> --recreate
   ```

### Explicit summary file

```bash
python3 scripts/compaction_engine.py \
  --external-summary-file /path/to/summary.txt --session-id <id> --recreate
```

### Execution modes

| Flag             | Default | Effect                                                      |
| ---------------- | ------- | ----------------------------------------------------------- |
| `--analyze-only` | off     | Load/group/report only. No API, no DB, no cache.            |
| `--dry-run`      | off     | Full pipeline incl. Clef; no DB write, no cache write.      |
| `--recreate`     | **off** | Reconstruct `state.db`. Destructive; verified backup first. |
| `--no-recreate`  | —       | Explicitly disable reconstruction.                          |

### Other options

| Flag                                                   | Default                       | Meaning                                                  |
| ------------------------------------------------------ | ----------------------------- | -------------------------------------------------------- |
| `--threshold FLOAT`                                    | 0.3                           | Keep units with `noul >= threshold`, in `[0,1]`          |
| `--chunk-size INT`                                     | 20                            | Hard cap on messages per request                         |
| `--throttle INT`                                       | 2                             | Seconds between chunks                                   |
| `--session-id ID`                                      | auto                          | Explicit session (Agent/fallback mode)                   |
| `--fallback`                                           | off                           | Read `state.db`; use truncation summary if none supplied |
| `--context-dir PATH`                                   | `~/.hermes/cache/hcompaction` | Directory with `ctx.json` (batch mode)                   |
| `--output PATH`                                        | stdout                        | Write result JSON to a file                              |
| `--use-external-summary` / `--no-use-external-summary` | on                            | Use the provided semantic summary                        |

Results (JSON) are printed **once** to stdout; logs go to stderr.

## Threshold semantics

`--threshold 0.3` means "retain a unit whose Clef `noul` score is ≥ 0.3" — it does
**not** mean "retain 30% of messages". The actual ratio depends on the model's
score distribution. Treat `noul` as a relative relevance signal; its probability
calibration is not independently verified. The threshold is recorded in every
report and result (never hard-coded in reporting).

## Caching

Content-addressed and validated. A cache hit is claimed only when all of these
match:

`session_id + messages_hash + summary_hash + threshold + config_hash + algorithm_version`

- Different summaries, thresholds, messages or configs → different cache entries.
- Every cached entry is integrity- and identity-validated before use; corruption,
  version mismatch or missing files → cache ignored and recomputed.
- Writes are atomic; the previous valid result is preserved until the replacement
  is verified. A file lock serializes concurrent writers. Entries expire after
  `retention_days` (30).
- Cache location: `~/.hermes/cache/hcompaction/v2/<sig-prefix>/<signature>/`.
- The algorithm version is currently **2.1.0** and is part of every cache
  identity. The 2.0.0 → 2.1.0 bump (content-intrinsic canonical ordering,
  order-sensitive cache fingerprint, cross-unit tool-link preservation, a
  `chunk_size` ceiling on split paths, and fail-safe handling of non-dict
  input) intentionally invalidates all earlier cache entries; a recomputation
  after upgrade is expected, not a cache failure.

## Retention & filtering

- **Hard preservation** (transparent rules): newest N messages, explicit
  `[IMPORTANT]`/`[MUST KEEP]`/`[PRESERVE]`/`[CRITICAL]` markers, tool call/result
  link integrity.
- **Clef relevance**: conversation units scored against the semantic summary.
- **Fail-safe**: any missing, malformed, non-finite, out-of-range or unmatched
  answer is retained and recorded. `Retained = Hard ∪ Clef ∪ FailSafe`.
- Structural verification is mandatory; if it fails, the engine conservatively
  retains all messages.

### Grouping modes: `group` vs `individual`

- `group` (default) scores each conversation unit **as a whole**, so a retained
  answer never loses its question or tool context. The cost of that atomicity:
  when one message in a unit is relevant the whole unit is kept, and any
  interspersed noise inside it is kept along with it — that noise is the price
  of unit-level atomicity, not a mis-score. `group` mode does not surgically
  drop noisy messages that sit inside a relevant unit.
- `hard_preserve_tool_links` (default on) guarantees a tool call and its result
  are preserved **together** (atomicity). It does **not** guarantee that a unit
  judged irrelevant is retained: such a unit can still be dropped as a whole,
  within the fail-safe and structural-verification rules above. Pair detection
  is content-based, so it works in `individual` mode too.
- To drop interspersed noise **one message at a time**, use
  `grouping_mode=individual`, which scores each message as its own unit while
  still linking tool call/result pairs across units when
  `hard_preserve_tool_links` is on.

`grouping_mode` and `hard_preserve_tool_links` are configuration-level options
(result-relevant, part of the cache identity), not CLI flags — see `SKILL.md`
§6 and §11 for details and the exact guarantees.

## Database safety & recovery

- Reconstruction only with `--recreate`, only for a real explicit session.
- Backup via SQLite's online backup API (WAL-safe) with `PRAGMA integrity_check`.
- Only rows actually read are deactivated; messages arriving mid-call stay active.
- One transaction; source fingerprint re-verified inside it; full rollback on failure.
- Reconstruction flips `active`/`compacted` flags in place, preserving every column
  (e.g. metadata, custom fields) — no guessed INSERT column list.
- Session ID is preserved; empty retained sets are rejected.

Recovery: a verified backup is written before every reconstruction to
`<state.db>.precompaction.<timestamp>.bak`. Restore with
`DatabaseManager.restore(path)` (it refuses corrupt backups and keeps a
`before_restore` copy) or by copying the backup over `state.db`. A failed
reconstruction never destroys the original active messages and is reported as
`database_status: failed`, `success: false` — never as a success.

## Output files

- stdout (or `--output`): final result JSON
- `~/.hermes/cache/hcompaction/v2/<prefix>/<sig>/`:
  - `compacted.json` — validated cached result
  - `metadata.json` — full identity (signature, hashes, threshold, version)
  - `compaction_report.md` / `retention_report.json` — detailed report

Reports include session ID, input/summary fingerprints, algorithm version, model,
threshold, chunk/API counts, failures, retention breakdown by source, cache status,
database status, backup path, validation results and duration. No tokens or full
message bodies are logged.

## Testing

```bash
python3 -m pytest tests/ -q
```

132 tests cover summary edge cases (Unicode, paths, encoding, limits), cache
identity/integrity (all mismatch cases), filtering (all/missing/malformed/NaN/
schema/timeout/429/partial), conversation grouping, token-safe chunking, session
isolation/ordering, database safety (WAL, concurrency, rollback, restore, extra
columns) and full-lifecycle integration with a mocked Clef API and a temporary
Hermes-like database. Destructive tests never touch the real `~/.hermes/state.db`.

## Known limitations

- The real Hermes `state.db` schema, hot-reload mechanism and skill dispatch
  behavior could not be inspected in this repository; the reconstruction design
  is conservative (in-place flag flips preserve unknown columns) but should be
  confirmed against a live Hermes installation before production use.
- Cloudflare Clef-flash response schema and `noul` calibration could not be
  verified without live credentials; multiple documented/observed response shapes
  are supported explicitly and anything ambiguous fails safe (retains).
- Token estimation is a bounded heuristic (CJK ≈ 1 token, other ≈ 3 chars/token),
  not a real tokenizer — deliberately pessimistic.

## License

MIT
