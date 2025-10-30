## ADDED Requirements

### Requirement: CLI Auto-Fix Validation
The CLI SHALL provide an opt-in auto-fix mode for validation that applies a safe, deterministic set of fixes to spec files when invoked with `--auto-fix`.

#### Scenario: Dry-run lists proposed fixes without writing files
- **WHEN** the user runs `openspec validate --auto-fix --dry-run`
- **THEN** the CLI SHALL output a concise list of proposed fixes and exit with code `0` if only fixable issues were found

#### Scenario: Auto-fix applies changes and reports summary
- **WHEN** the user runs `openspec validate --auto-fix`
- **THEN** the CLI SHALL apply safe fixes to the file system and print a summary of changed files and a report of remaining validation errors (if any)

#### Scenario: No-op when nothing to fix
- **WHEN** the user runs `openspec validate --auto-fix` on already-valid specs
- **THEN** the CLI SHALL report "No fixes necessary" and perform no file writes
