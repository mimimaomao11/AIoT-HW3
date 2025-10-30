# Project Context

## Purpose
This repository holds the OpenSpec-driven specification and implementation for the project you are developing in this workspace. The repo's goals are:
- Capture behavioral requirements as machine-parseable specs
- Drive development via reviewable change proposals
- Keep specs and implementation in sync using the OpenSpec workflow

## Tech Stack (assumptions)
The project appears to be a Node.js-based developer tooling / spec project. I've populated reasonable defaults below — tell me if these should change.
- Runtime: Node.js 18+ (LTS)
- Language: TypeScript
- CLI tooling: npm / pnpm (npm by default)
- Testing: Jest
- Linting & Formatting: ESLint + Prettier
- CI: GitHub Actions

If your project uses a different stack (Python, Go, plain JS, etc.) let me know and I'll update this section.

## Project Conventions

### Code Style
- Use TypeScript with strict mode enabled where possible
- Files: kebab-case for CLI and folder names, PascalCase for React components, camelCase for variables/functions
- Formatting: Prettier defaults, 2-space indent
- Linting: ESLint with recommended rules and TypeScript plugin

### Architecture Patterns
- Single-repository focused on an opinionated CLI + specs
- Keep capabilities small and single-purpose (one capability per folder under `openspec/specs/`)
- Prefer small, well-tested modules over large frameworks

### Testing Strategy
- Unit tests for business logic (Jest)
- Integration tests for CLI flows (spawn the CLI in test harness)
- Specs are the source of truth: write tests that assert implementation satisfies spec scenarios

### Git Workflow
- Branch per change: `change/<change-id>` or `feat/<short-desc>`
- Commit messages: Conventional-ish — short imperative summary; reference change-id when relevant
- PRs should reference the OpenSpec `changes/<change-id>/proposal.md` and include test results

## Domain Context
Provide here any domain-specific knowledge the assistant should know (authentication model, data retention rules, expected scale). Examples:
- This is a developer tooling project for spec management and validation
- Not user-facing; data sensitivity is low

## Important Constraints
- Backwards compatibility for existing spec formats is important
- CLI changes must be discoverable via `--help` and not break existing scripts

## External Dependencies
- npm registry packages (openspec CLI and other libs)
- CI (GitHub Actions) for automated validation

---

Notes / Assumptions:
- I assumed a Node + TypeScript toolchain. If that is incorrect, tell me the correct tech stack and I'll update `project.md` accordingly.
- I left domain-specific fields intentionally high-level — if you provide more context (auth, external APIs, scale), I will incorporate it.
