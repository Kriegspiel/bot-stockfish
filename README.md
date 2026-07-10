# bot-stockfish

Experimental Stockfish-backed Wild 16 Kriegspiel bot.

This is not a privileged Stockfish player and it does not read hidden board
state. The bot uses the ordinary Kriegspiel API, builds sampled full-board
hypotheses from public Wild 16 information, asks Stockfish to score legal root
moves on those hypotheses, averages the scores, and submits public API move
attempts until one succeeds.

## Current Scope

- supports only `wild16`
- defaults to unlisted registration for testing
- intended lobby catalogue tier is T1 under username `stockfishwild`
- runs one bot process with one lightweight runner thread per active game
- defaults to one active game because Stockfish analysis is CPU-bound
- auto-creates one open Wild 16 lobby game when below the active-game cap
- samples compatible bot-created Wild 16 waiting games with low probability
- persists local per-game belief matrices in `.bot-state.json`
- falls back to deterministic public-state heuristics when Stockfish is absent
  or analysis fails

The first policy is intentionally modest. It samples opponent king, pawn, and
piece locations from Darkboard-style public material priors, then uses Stockfish
root analysis on each sampled board. It should be treated as an experiment, not
as a strength claim.

## Setup

Install Stockfish on the host. On Ubuntu, package `stockfish 16-1build1`
installs the binary at `/usr/games/stockfish`; the production-friendly path in
`.env.example` assumes a symlink at `/usr/local/bin/stockfish`.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
python bot.py --register
python bot.py --poll-seconds 2
```

Multiple instances should use separate env and state files:

```bash
python bot.py \
  --env-file instances/stockfish.env \
  --state-file instances/stockfish-state.json
```

Key runtime controls:

- `STOCKFISH_PATH=/usr/local/bin/stockfish`
- `STOCKFISH_ENABLED=true|false`
- `STOCKFISH_SAMPLE_COUNT=4`
- `STOCKFISH_DEPTH=6`
- `STOCKFISH_TIME_LIMIT_SECONDS=0.05`
- `STOCKFISH_MULTIPV=8`
- `STOCKFISH_RANDOM_SEED=` for reproducible sampling
- `KRIEGSPIEL_MAX_ACTIVE_GAMES=1`
- `KRIEGSPIEL_BOT_LISTED=false` while testing
- keep `KRIEGSPIEL_BOT_USERNAME=stockfishwild` unless the web app tier mapping
  is updated with the replacement username

## Development

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## systemd

A production host can run the bot as a service with
`deploy/kriegspiel-stockfish-bot.service` once credentials have been created on
that host.
