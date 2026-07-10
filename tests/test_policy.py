import random

import chess
import chess.engine

from stockfish_bot.belief import BeliefState
from stockfish_bot.policy import StockfishConfig
from stockfish_bot.policy import fallback_ranked_actions
from stockfish_bot.policy import ranked_actions
from stockfish_bot.policy import sample_hypothesis_boards


VISIBLE_WHITE_OPENING = "8/8/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1"


def belief_with_only_opponent_king() -> BeliefState:
    return BeliefState.from_api_state(
        {
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
                "black": {"pieces_remaining": 1, "pawns_captured": 8},
            },
            "referee_log": [],
        },
        ruleset="wild16",
    )


class FakeEngine:
    def __init__(self, scores: dict[str, int]) -> None:
        self.scores = scores
        self.closed = False

    def analyse(self, board, limit, *, root_moves, multipv):  # noqa: ANN001, ANN002, ARG002
        infos = []
        for move in sorted(root_moves, key=lambda item: self.scores.get(item.uci(), 0), reverse=True)[:multipv]:
            infos.append(
                {
                    "pv": [move],
                    "score": chess.engine.PovScore(chess.engine.Cp(self.scores.get(move.uci(), 0)), chess.WHITE),
                }
            )
        return infos

    def quit(self) -> None:
        self.closed = True


def test_sample_hypothesis_boards_places_public_opponent_material() -> None:
    belief = belief_with_only_opponent_king()

    boards = sample_hypothesis_boards(belief, count=2, rng=random.Random(7))

    assert len(boards) == 2
    for board in boards:
        pieces = board.piece_map()
        assert sum(1 for piece in pieces.values() if piece.color == chess.BLACK) == 1
        assert board.king(chess.WHITE) == chess.E1
        assert board.king(chess.BLACK) is not None
        assert board.piece_at(chess.E2) == chess.Piece(chess.PAWN, chess.WHITE)


def test_ranked_actions_uses_stockfish_root_scores() -> None:
    belief = belief_with_only_opponent_king()
    fake = FakeEngine({"e2e4": 120, "d2d4": 20})
    config = StockfishConfig(sample_count=1, multipv=2, random_seed=3, depth=1, time_limit_seconds=0.0)

    ranked = ranked_actions(belief, config=config, engine_factory=lambda _path: fake)

    assert ranked[0] == "e2e4"
    assert fake.closed is True


def test_ranked_actions_falls_back_when_engine_missing() -> None:
    belief = BeliefState.from_api_state(
        {
            "game_id": "game-1",
            "state": "active",
            "turn": "white",
            "your_color": "white",
            "your_fen": "8/P7/8/8/8/8/8/4K3 w - - 0 1",
            "allowed_moves": ["a7a8n", "a7a8q"],
            "possible_actions": ["move"],
            "move_number": 1,
            "material_summary": {
                "white": {"pieces_remaining": 2, "pawns_captured": 7},
                "black": {"pieces_remaining": 1, "pawns_captured": 8},
            },
            "referee_log": [],
        },
        ruleset="wild16",
    )
    config = StockfishConfig(sample_count=1, random_seed=1)

    ranked = ranked_actions(belief, config=config, engine_factory=lambda _path: (_ for _ in ()).throw(FileNotFoundError()))

    assert ranked == fallback_ranked_actions(belief)
    assert ranked[0] == "a7a8q"
