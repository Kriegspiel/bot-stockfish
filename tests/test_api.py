from stockfish_bot import api


VISIBLE_WHITE_OPENING = "8/8/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1"


def test_supported_rule_variants_are_wild16_only(monkeypatch) -> None:
    monkeypatch.setenv("KRIEGSPIEL_SUPPORTED_RULE_VARIANTS", "berkeley,wild16,crazykrieg")

    assert api.supported_rule_variants() == ["wild16"]


def test_maybe_play_game_retries_ranked_attempts_and_persists_belief(tmp_path, monkeypatch) -> None:
    api.configure_runtime_paths(state_path=tmp_path / "state.json")
    state_payload = {
        "game_id": "game-1",
        "state": "active",
        "turn": "white",
        "your_color": "white",
        "your_fen": VISIBLE_WHITE_OPENING,
        "allowed_moves": ["e2e4", "d2d4"],
        "possible_actions": ["move"],
        "move_number": 1,
        "material_summary": {
            "white": {"pieces_remaining": 16, "pawns_captured": 0},
            "black": {"pieces_remaining": 16, "pawns_captured": 0},
        },
        "referee_log": [],
    }
    posts: list[tuple[str, dict]] = []
    responses = [
        {"announcement": "ILLEGAL_MOVE", "move_done": False},
        {"announcement": "REGULAR_MOVE", "move_done": True},
    ]

    monkeypatch.setattr(api, "get_json", lambda path: state_payload)
    monkeypatch.setattr(api, "ranked_actions", lambda belief: ("e2e4", "d2d4"))
    monkeypatch.setattr(api.time, "sleep", lambda _seconds: None)

    def fake_post(path, payload=None):
        posts.append((path, payload or {}))
        return responses.pop(0)

    monkeypatch.setattr(api, "post_json", fake_post)

    assert api.maybe_play_game({"game_id": "game-1", "rule_variant": "wild16"}) is True
    assert posts == [
        ("/game/game-1/move", {"uci": "e2e4"}),
        ("/game/game-1/move", {"uci": "d2d4"}),
    ]
    state = api.load_state()
    assert "game-1" in state["beliefs"]
    assert state["beliefs"]["game-1"]["observed_referee_log_size"] == 1


def test_open_bot_lobby_candidates_filter_to_wild16_bots(monkeypatch) -> None:
    monkeypatch.setenv("KRIEGSPIEL_BOT_USERNAME", "stockfishwild")
    games = [
        {"game_code": "WLD123", "created_by": "randobot", "rule_variant": "wild16"},
        {"game_code": "BER123", "created_by": "randobot", "rule_variant": "berkeley"},
        {"game_code": "SELF12", "created_by": "stockfishwild", "rule_variant": "wild16"},
        {"game_code": "HUM123", "created_by": "fil", "rule_variant": "wild16"},
    ]

    def profile(username):
        return {"role": "bot" if username == "randobot" else "user"}

    assert api.open_bot_lobby_candidates(games, profile_lookup=profile) == [games[0]]
