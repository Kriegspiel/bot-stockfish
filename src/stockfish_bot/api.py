"""Kriegspiel API runtime for the Stockfish Wild 16 bot."""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import threading
import time
from pathlib import Path
from typing import Any

import requests

from stockfish_bot.checkpoints import CheckpointStore
from stockfish_bot.belief import BeliefState
from stockfish_bot.belief import WILD16_RULESET
from stockfish_bot.belief import apply_move_result_evidence
from stockfish_bot.belief import restore_belief_snapshot
from stockfish_bot.belief import serialize_belief
from stockfish_bot.policy import ranked_actions


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_PATH = Path(os.environ.get("STOCKFISH_BOT_ENV_PATH", PACKAGE_ROOT / ".env"))
DEFAULT_STATE_PATH = Path(os.environ.get("STOCKFISH_BOT_STATE_PATH", PACKAGE_ROOT / ".bot-state.json"))
ENV_PATH = DEFAULT_ENV_PATH
STATE_PATH = DEFAULT_STATE_PATH
DEFAULT_TIMEOUT_SECONDS = 20
BOT_JOIN_COOLDOWN_SECONDS = 60
FAILED_MOVE_RETRY_DELAY_SECONDS = 1
DEFAULT_ACTIVE_GAME_DISCOVERY_LIMIT = 100
SUPPORTED_RULE_VARIANTS = (WILD16_RULESET,)
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO), format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)
_STATE_LOCK = threading.RLock()
_CHECKPOINT_STORE: CheckpointStore | None = None
_BOT_PROFILE_SYNCED = False


def configure_runtime_paths(*, env_path: str | Path | None = None, state_path: str | Path | None = None) -> None:
    global ENV_PATH, STATE_PATH
    ENV_PATH = Path(env_path).expanduser().resolve() if env_path else DEFAULT_ENV_PATH
    STATE_PATH = Path(state_path).expanduser().resolve() if state_path else DEFAULT_STATE_PATH


def load_env_file(path: str | Path | None = None) -> None:
    env_path = Path(path) if path is not None else ENV_PATH
    if not env_path.exists():
        return

    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def base_url() -> str:
    return os.environ.get("KRIEGSPIEL_API_BASE", "http://localhost:8000").rstrip("/")


def auth_headers() -> dict[str, str]:
    token = os.environ.get("KRIEGSPIEL_BOT_TOKEN", "").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def bot_username() -> str:
    return os.environ.get("KRIEGSPIEL_BOT_USERNAME", "").strip().lower()


def bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def int_env(name: str, default: int, *, minimum: int = 0, maximum: int | None = None) -> int:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = max(minimum, int(raw))
    except ValueError:
        return default
    return min(maximum, value) if maximum is not None else value


def float_env(name: str, default: float, *, minimum: float = 0.0, maximum: float | None = None) -> float:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = max(minimum, float(raw))
    except ValueError:
        return default
    return min(maximum, value) if maximum is not None else value


def max_active_games() -> int:
    return int_env("KRIEGSPIEL_MAX_ACTIVE_GAMES", 1, minimum=1)


def active_game_discovery_limit() -> int:
    return int_env("KRIEGSPIEL_ACTIVE_GAME_DISCOVERY_LIMIT", DEFAULT_ACTIVE_GAME_DISCOVERY_LIMIT, minimum=1, maximum=100)


def bot_game_pick_probability() -> float:
    return float_env("BOT_GAME_PICK_PROBABILITY", 0.05, minimum=0.0, maximum=1.0)


def checkpoint_store() -> CheckpointStore:
    global _CHECKPOINT_STORE
    with _STATE_LOCK:
        if _CHECKPOINT_STORE is None or _CHECKPOINT_STORE.state_path != STATE_PATH:
            _CHECKPOINT_STORE = CheckpointStore(STATE_PATH)
        return _CHECKPOINT_STORE


def _load_state_unlocked() -> dict[str, Any]:
    return checkpoint_store().load_metadata()


def _save_state_unlocked(state: dict[str, Any]) -> None:
    checkpoint_store().save_metadata(state)


def load_state() -> dict[str, Any]:
    with _STATE_LOCK:
        return _load_state_unlocked()


def save_state(state: dict[str, Any]) -> None:
    with _STATE_LOCK:
        _save_state_unlocked(state)


def save_token(token: str) -> None:
    with _STATE_LOCK:
        state = _load_state_unlocked()
        state["token"] = token
        _save_state_unlocked(state)


def maybe_restore_token() -> None:
    if os.environ.get("KRIEGSPIEL_BOT_TOKEN"):
        return
    token = load_state().get("token")
    if isinstance(token, str) and token:
        os.environ["KRIEGSPIEL_BOT_TOKEN"] = token


def supported_rule_variants() -> list[str]:
    raw = os.environ.get("KRIEGSPIEL_SUPPORTED_RULE_VARIANTS", WILD16_RULESET)
    variants: list[str] = []
    for item in raw.split(","):
        value = item.strip()
        if value in SUPPORTED_RULE_VARIANTS and value not in variants:
            variants.append(value)
    return variants or [WILD16_RULESET]


def register_bot() -> None:
    response = requests.post(
        f"{base_url()}/auth/bots/register",
        json={
            "username": os.environ.get("KRIEGSPIEL_BOT_USERNAME", "stockfishwild"),
            "display_name": os.environ.get("KRIEGSPIEL_BOT_DISPLAY_NAME", "Stockfish Wild 16"),
            "owner_email": os.environ["KRIEGSPIEL_BOT_OWNER_EMAIL"],
            "description": os.environ.get(
                "KRIEGSPIEL_BOT_DESCRIPTION",
                "Experimental Stockfish-backed Wild 16 bot using public-state board hypotheses.",
            ),
            "listed": bool_env("KRIEGSPIEL_BOT_LISTED", False),
            "supported_rule_variants": supported_rule_variants(),
        },
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    save_token(payload["api_token"])
    logger.info("registered bot %s", os.environ.get("KRIEGSPIEL_BOT_USERNAME", "stockfishwild"))


def sync_bot_profile() -> dict[str, Any]:
    return post_json(
        "/bots/profile",
        {
            "display_name": os.environ.get("KRIEGSPIEL_BOT_DISPLAY_NAME", "Stockfish Wild 16"),
            "description": os.environ.get(
                "KRIEGSPIEL_BOT_DESCRIPTION",
                "Experimental Stockfish-backed Wild 16 bot using public-state board hypotheses.",
            ),
            "supported_rule_variants": supported_rule_variants(),
        },
    )


def maybe_sync_bot_profile() -> bool:
    global _BOT_PROFILE_SYNCED
    if _BOT_PROFILE_SYNCED:
        return False
    payload = sync_bot_profile()
    _BOT_PROFILE_SYNCED = True
    logger.info("synced bot profile: %s", payload.get("supported_rule_variants"))
    return True


def get_json(path: str) -> dict[str, Any]:
    response = requests.get(f"{base_url()}{path}", headers=auth_headers(), timeout=DEFAULT_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def get_public_user(username: str) -> dict[str, Any]:
    response = requests.get(f"{base_url()}/user/{username}", headers=auth_headers(), timeout=DEFAULT_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def post_json(path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    response = requests.post(
        f"{base_url()}{path}",
        headers=auth_headers(),
        json=payload or {},
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def active_games(games: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [game for game in games if game.get("state") == "active"]


def waiting_games(games: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [game for game in games if game.get("state") == "waiting"]


def under_active_game_limit(games: list[dict[str, Any]]) -> bool:
    return len(active_games(games)) < max_active_games()


def auto_create_enabled() -> bool:
    return bool_env("KRIEGSPIEL_AUTO_CREATE_LOBBY_GAME", True)


def create_payload() -> dict[str, str]:
    return {
        "rule_variant": WILD16_RULESET,
        "play_as": os.environ.get("KRIEGSPIEL_AUTO_CREATE_PLAY_AS", "random").strip() or "random",
        "time_control": "rapid",
        "opponent_type": "human",
    }


def has_own_waiting_game(open_games: list[dict[str, Any]]) -> bool:
    own_username = bot_username()
    for game in open_games:
        created_by = str(game.get("created_by") or "").strip().lower()
        if created_by and created_by == own_username:
            return True
    return False


def maybe_create_lobby_game(games: list[dict[str, Any]]) -> bool:
    if not auto_create_enabled() or not under_active_game_limit(games) or waiting_games(games):
        return False
    open_games = get_json("/game/open").get("games", [])
    if not isinstance(open_games, list) or has_own_waiting_game(open_games):
        return False
    created = post_json("/game/create", create_payload())
    logger.info("created Wild 16 lobby game %s (%s)", created.get("game_id"), created.get("game_code"))
    return True


def open_bot_lobby_candidates(open_games: list[dict[str, Any]], *, profile_lookup=None) -> list[dict[str, Any]]:
    profile_lookup = profile_lookup or get_public_user
    own_username = bot_username()
    candidates: list[dict[str, Any]] = []
    for game in open_games:
        if str(game.get("rule_variant") or "").strip() != WILD16_RULESET:
            continue
        creator_username = str(game.get("created_by") or "").strip()
        if not creator_username or creator_username.lower() == own_username:
            continue
        try:
            profile = profile_lookup(creator_username)
        except requests.RequestException:
            continue
        is_bot = bool(profile.get("is_bot")) or str(profile.get("role") or "").strip().lower() == "bot"
        if is_bot:
            candidates.append(game)
    return candidates


def can_attempt_bot_join(now: float | None = None) -> bool:
    current = time.time() if now is None else now
    last_attempt = load_state().get("last_bot_game_join_attempt_at", 0)
    try:
        last_attempt = float(last_attempt)
    except (TypeError, ValueError):
        last_attempt = 0
    return current - last_attempt >= BOT_JOIN_COOLDOWN_SECONDS


def record_bot_join_attempt(now: float | None = None) -> None:
    state = load_state()
    state["last_bot_game_join_attempt_at"] = time.time() if now is None else now
    save_state(state)


def choose_bot_game_to_join(open_games: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = open_bot_lobby_candidates(open_games)
    return random.choice(candidates) if candidates else None


def maybe_join_bot_lobby_game(games: list[dict[str, Any]]) -> bool:
    if not under_active_game_limit(games) or not can_attempt_bot_join():
        return False
    record_bot_join_attempt()
    open_games = get_json("/game/open").get("games", [])
    if not isinstance(open_games, list):
        return False
    candidate = choose_bot_game_to_join(open_games)
    if candidate is None or random.random() >= bot_game_pick_probability():
        return False
    game_code = str(candidate.get("game_code") or "").strip()
    if not game_code:
        return False
    joined = post_json(f"/game/join/{game_code}")
    logger.info("joined Wild 16 bot lobby game %s (%s)", joined.get("game_id"), joined.get("game_code"))
    return True


def game_ref(game: dict[str, Any]) -> str | None:
    for key in ("game_id", "game_code"):
        value = game.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def game_ruleset(game: dict[str, Any]) -> str:
    return str(game.get("rule_variant") or WILD16_RULESET)


def restore_belief(game_id: str, belief: BeliefState) -> BeliefState:
    snapshot = checkpoint_store().load_belief(game_id)
    return restore_belief_snapshot(belief, snapshot)


def save_belief(game_id: str, belief: BeliefState) -> None:
    checkpoint_store().save_belief(game_id, serialize_belief(belief))


def prune_one_checkpoint() -> None:
    store = checkpoint_store()
    ref = store.next_cleanup_ref()
    if ref is not None:
        state = get_json(f"/game/{ref}/public-status")
        if state.get("state") == "completed":
            store.release(ref, completed=True)


def maybe_play_game(game: dict[str, Any] | str) -> bool:
    if isinstance(game, str):
        ref = game
        ruleset = WILD16_RULESET
    else:
        ref = game_ref(game)
        ruleset = game_ruleset(game)
    if not ref or ruleset != WILD16_RULESET:
        return False

    state = get_json(f"/game/{ref}/state")
    if state.get("state") != "active" or state.get("turn") != state.get("your_color"):
        return False
    if "move" not in (state.get("possible_actions") if isinstance(state.get("possible_actions"), list) else []):
        return False

    belief = restore_belief(ref, BeliefState.from_api_state(state, ruleset=ruleset))
    save_belief(ref, belief)
    actions = ranked_actions(belief)
    if not actions:
        return False

    for index, uci in enumerate(actions):
        result = post_json(f"/game/{ref}/move", {"uci": uci})
        logger.debug("%s: tried %s -> %s", ref, uci, result.get("announcement"))
        belief = apply_move_result_evidence(belief, uci=uci, result=result)
        save_belief(ref, belief)
        if result.get("move_done"):
            return True
        if index < len(actions) - 1:
            time.sleep(FAILED_MOVE_RETRY_DELAY_SECONDS)
    return False


def http_status_code(exc: requests.RequestException) -> int | None:
    response = getattr(exc, "response", None)
    if response is None:
        return None
    status_code = getattr(response, "status_code", None)
    return int(status_code) if isinstance(status_code, int) else None


class GameRunner:
    def __init__(self, game_ref_value: str, *, poll_seconds: float) -> None:
        self.game_ref = game_ref_value
        self.poll_seconds = max(0.5, float(poll_seconds))
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name=f"stockfish-bot-game-{game_ref_value}", daemon=True)
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        logger.info("%s: starting game runner", self.game_ref)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()

    def join(self, timeout: float | None = None) -> None:
        if self._started:
            self.thread.join(timeout=timeout)

    def is_alive(self) -> bool:
        return self._started and self.thread.is_alive()

    def _wait(self) -> None:
        self.stop_event.wait(self.poll_seconds)

    def _run(self) -> None:
        stop_reason = "stopped"
        try:
            while not self.stop_event.is_set():
                try:
                    state = get_json(f"/game/{self.game_ref}/state")
                except requests.RequestException as exc:
                    status_code = http_status_code(exc)
                    if status_code in {400, 403, 404, 409}:
                        stop_reason = f"state unavailable http_{status_code}"
                        break
                    logger.warning("%s: runner state poll failed: %s", self.game_ref, exc)
                    self._wait()
                    continue

                state_value = state.get("state")
                if state_value != "active":
                    if state_value == "completed":
                        checkpoint_store().release(self.game_ref, completed=True)
                    stop_reason = f"state={state_value}"
                    break

                if state.get("turn") == state.get("your_color"):
                    try:
                        maybe_play_game(self.game_ref)
                    except requests.RequestException as exc:
                        status_code = http_status_code(exc)
                        if status_code in {400, 403, 404, 409}:
                            stop_reason = f"play stopped http_{status_code}"
                            break
                        logger.warning("%s: runner play failed: %s", self.game_ref, exc)

                self._wait()
        finally:
            checkpoint_store().release(self.game_ref)
            logger.info("%s: stopped game runner (%s)", self.game_ref, stop_reason)


class GameRunnerScheduler:
    def __init__(self, *, poll_seconds: float, runner_factory: Any | None = None) -> None:
        self.poll_seconds = poll_seconds
        self.runner_factory = runner_factory or (lambda ref: GameRunner(ref, poll_seconds=poll_seconds))
        self.runners: dict[str, Any] = {}

    @staticmethod
    def game_id_for(game: dict[str, Any]) -> str:
        return game_ref(game) or ""

    def reconcile(self, games: list[dict[str, Any]]) -> None:
        active_ids: set[str] = set()
        for game in active_games(games):
            if game_ruleset(game) != WILD16_RULESET:
                continue
            ref = self.game_id_for(game)
            if not ref:
                continue
            active_ids.add(ref)
            runner = self.runners.get(ref)
            if runner is not None and runner.is_alive():
                continue
            if runner is not None:
                runner.join(timeout=0)
            runner = self.runner_factory(ref)
            self.runners[ref] = runner
            runner.start()

        for ref, runner in list(self.runners.items()):
            if ref in active_ids or runner.is_alive():
                continue
            runner.join(timeout=0)
            self.runners.pop(ref, None)

        self.prune_finished()

    def prune_finished(self) -> None:
        for ref, runner in list(self.runners.items()):
            if runner.is_alive():
                continue
            runner.join(timeout=0)
            self.runners.pop(ref, None)

    def stop_all(self) -> None:
        for runner in list(self.runners.values()):
            runner.stop()
        for runner in list(self.runners.values()):
            runner.join(timeout=2.0)
        self.runners.clear()


def run_loop(poll_seconds: float) -> None:
    discovery_limit = active_game_discovery_limit()
    logger.info("active-game discovery limit configured: max=%s", discovery_limit)
    scheduler = GameRunnerScheduler(poll_seconds=poll_seconds)
    try:
        while True:
            try:
                maybe_sync_bot_profile()
                mine = get_json(f"/game/mine/active?limit={discovery_limit}")
                games = mine.get("games", [])
                if not isinstance(games, list):
                    games = []
                maybe_create_lobby_game(games)
                maybe_join_bot_lobby_game(games)
                scheduler.reconcile(games)
                prune_one_checkpoint()
            except requests.RequestException as exc:
                logger.warning("poll failed: %s", exc)
            time.sleep(poll_seconds)
    finally:
        scheduler.stop_all()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Stockfish-backed Wild 16 bot.")
    parser.add_argument(
        "--env-file",
        default=os.environ.get("STOCKFISH_BOT_ENV_PATH", str(DEFAULT_ENV_PATH)),
        help="path to the bot instance env file",
    )
    parser.add_argument(
        "--state-file",
        default=os.environ.get("STOCKFISH_BOT_STATE_PATH", str(DEFAULT_STATE_PATH)),
        help="path to the bot instance state file",
    )
    parser.add_argument("--register", action="store_true", help="register the bot and store its bearer token")
    parser.add_argument("--poll-seconds", type=float, default=2.0, help="poll interval between API rounds")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    configure_runtime_paths(env_path=args.env_file, state_path=args.state_file)
    load_env_file()
    checkpoint_store().prepare()
    maybe_restore_token()

    if args.register:
        register_bot()
        return 0

    if not os.environ.get("KRIEGSPIEL_BOT_TOKEN"):
        logger.error("missing KRIEGSPIEL_BOT_TOKEN; run with --register first or set it in the environment")
        return 1

    run_loop(args.poll_seconds)
    return 0
