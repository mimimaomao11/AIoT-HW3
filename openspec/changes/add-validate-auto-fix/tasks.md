## 1. Implementation
- [ ] 1.1 Add CLI parsing for `--auto-fix` and `--dry-run` flags
- [ ] 1.2 Implement a modular fixer pipeline (whitelist of safe fixes)
- [ ] 1.3 Add unit tests for each fixer
- [ ] 1.4 Add integration test(s) for `openspec validate --auto-fix` flow
- [ ] 1.5 Update `openspec` help text and docs
- [ ] 1.6 Add a spec delta under `changes/add-validate-auto-fix/specs/cli/spec.md`

## 2. QA / Validation
- [ ] 2.1 Run `openspec validate add-validate-auto-fix --strict` on the change directory
- [ ] 2.2 Add tests demonstrating no-op behavior when no fixes are necessary

## 3. Rollout
- [ ] 3.1 Feature flag behind CLI opt-in only
- [ ] 3.2 Announce in changelog / release notes
