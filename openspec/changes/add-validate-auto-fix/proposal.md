## Why

Developers frequently run `openspec validate` and encounter minor, fixable formatting or metadata issues (e.g., missing headers, trimmed whitespace, or scenario header formatting). Providing an opt-in auto-fix mode reduces friction, improves spec hygiene, and speeds up iteration.

## What Changes

- **Add** a new CLI flag/command: `openspec validate --auto-fix` (or `openspec validate --fix`) that will attempt to automatically correct common, safe validation issues.
- **Add** corresponding spec delta (if CLI capabilities are considered a capability) documenting the new behavior.

**BREAKING**: none expected. The feature is opt-in and does not change validation semantics unless the user passes `--auto-fix`.

## Impact

- Affected specs: CLI capability spec (new)
- Affected code: `openspec` CLI validation code and any formatting helper utilities
- Risks: auto-fix could change files unexpectedly if run in CI without review. Recommend `--dry-run` output and explicit confirmation in interactive mode.
