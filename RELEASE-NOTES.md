# Release Notes

## 0.1.2 — 2026-10-25

Split belief persistence into atomic per-game checkpoints, migrate and verify legacy state with a protected backup, and clean up server-confirmed completed games. Active runner memory and checkpoint work no longer grow with lifetime game history.

These notes summarize the packaged bot runtime release history reconstructed
from the current repository state. Add a new section at the top when the
package version changes or when runtime, deployment-facing, or user-visible bot
behavior changes. Test-only and docs-only changes do not need entries unless
they affect operator workflow.

## bot-stockfish v. 0.1.1

- **Bot Identity**: `stockfishwild`, the listed T1 Stockfish-backed Wild 16 bot.
- **Rulesets**: supports `wild16` only and uses the ordinary public Kriegspiel
  API without hidden-board access.
- **Runtime Shape**: runs one process with one lightweight runner thread per
  active game, defaults to one active game because Stockfish analysis is
  CPU-bound, and persists local belief matrices in `.bot-state.json`.
- **Lobby Policy**: defaults to listed registration, can keep one Wild 16
  human-joinable lobby game open, and can sample compatible bot-created Wild 16
  waiting games with low probability.
- **Policy**: samples plausible full-board hypotheses from public Wild 16
  material information, scores root moves with local Stockfish MultiPV, averages
  the results, and falls back to deterministic public-state heuristics when
  Stockfish is unavailable or analysis fails.
- **Host Dependency**: expects a local Stockfish binary, with production using
  `/usr/local/bin/stockfish`.
