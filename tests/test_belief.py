import math

import chess

from stockfish_bot.belief import BeliefState
from stockfish_bot.belief import apply_move_result_evidence
from stockfish_bot.belief import restore_belief_snapshot


VISIBLE_WHITE_OPENING = "8/8/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1"


def api_state(**overrides):
    payload = {
        "game_id": "game-1",
        "state": "active",
        "turn": "white",
        "your_color": "white",
        "your_fen": VISIBLE_WHITE_OPENING,
        "allowed_moves": ["e2e4", "d2d4", "g1f3"],
        "possible_actions": ["move"],
        "move_number": 1,
        "material_summary": {
            "white": {"pieces_remaining": 16, "pawns_captured": 0},
            "black": {"pieces_remaining": 16, "pawns_captured": 0},
        },
        "referee_log": [],
    }
    payload.update(overrides)
    return payload


def test_belief_from_api_state_builds_public_material_priors() -> None:
    belief = BeliefState.from_api_state(api_state(), ruleset="wild16")

    assert belief.color == chess.WHITE
    assert belief.legal_actions == ("e2e4", "d2d4", "g1f3")
    assert math.isclose(sum(belief.opponent_king), 1.0)
    assert math.isclose(sum(belief.opponent_pawns), 8.0)
    assert math.isclose(sum(belief.opponent_pieces), 7.0)
    assert belief.opponent_king[chess.E1] == 0.0
    assert belief.opponent_pawns[chess.A2] == 0.0
    assert belief.opponent_pieces[chess.D1] == 0.0


def test_restore_belief_snapshot_ignores_incompatible_game() -> None:
    belief = BeliefState.from_api_state(api_state(), ruleset="wild16")
    snapshot = {
        "game_id": "other-game",
        "ruleset": "wild16",
        "color": "white",
        "observed_referee_log_size": 5,
        "opponent_king": [0.0] * 64,
        "opponent_pawns": [0.0] * 64,
        "opponent_pieces": [0.0] * 64,
    }

    restored = restore_belief_snapshot(belief, snapshot)

    assert math.isclose(sum(restored.opponent_king), 1.0)
    assert restored.observed_referee_log_size == 0


def test_failed_pawn_capture_suppresses_target_occupancy() -> None:
    state = api_state(
        your_fen="8/8/8/8/4P3/8/PPPP1PPP/RNBQKBNR w - - 0 1",
        allowed_moves=["e4d5"],
    )
    belief = BeliefState.from_api_state(state, ruleset="wild16")
    before = belief.opponent_pawns[chess.D5] + belief.opponent_pieces[chess.D5]

    updated = apply_move_result_evidence(belief, uci="e4d5", result={"move_done": False, "announcement": "ILLEGAL_MOVE"})
    after = updated.opponent_pawns[chess.D5] + updated.opponent_pieces[chess.D5]

    assert after < before
    assert updated.observed_referee_log_size == 1


def test_capture_result_decrements_public_pawn_total() -> None:
    state = api_state(
        your_fen="8/8/8/3P4/8/8/PPP1PPPP/RNBQKBNR w - - 0 1",
        allowed_moves=["d5e6"],
    )
    belief = BeliefState.from_api_state(state, ruleset="wild16")

    updated = apply_move_result_evidence(
        belief,
        uci="d5e6",
        result={
            "move_done": True,
            "announcement": "CAPTURE_DONE",
            "capture_square": "e6",
            "captured_piece_announcement": "PAWN",
        },
    )

    assert math.isclose(sum(updated.opponent_pawns), 7.0)
    assert updated.opponent_pawns[chess.E6] == 0.0
