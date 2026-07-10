"""Move ranking for the Stockfish Wild 16 bot."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import random
from typing import Any, Callable, Mapping, Sequence

import chess
import chess.engine

from stockfish_bot.belief import BeliefState
from stockfish_bot.belief import opponent_material_counts
from stockfish_bot.belief import parse_move
from stockfish_bot.belief import visible_board


logger = logging.getLogger(__name__)
EngineFactory = Callable[[str], chess.engine.SimpleEngine]
PIECE_VALUES = {
    chess.PAWN: 100.0,
    chess.KNIGHT: 320.0,
    chess.BISHOP: 330.0,
    chess.ROOK: 500.0,
    chess.QUEEN: 900.0,
    chess.KING: 0.0,
}


@dataclass(frozen=True)
class StockfishConfig:
    stockfish_path: str = "/usr/local/bin/stockfish"
    enabled: bool = True
    sample_count: int = 4
    depth: int = 6
    time_limit_seconds: float = 0.05
    multipv: int = 8
    random_seed: int | None = None
    illegal_sample_score: float = -350.0
    missing_multipv_margin: float = 50.0

    @classmethod
    def from_env(cls) -> "StockfishConfig":
        return cls(
            stockfish_path=os.environ.get("STOCKFISH_PATH", cls.stockfish_path).strip() or cls.stockfish_path,
            enabled=_bool_env("STOCKFISH_ENABLED", cls.enabled),
            sample_count=_int_env("STOCKFISH_SAMPLE_COUNT", cls.sample_count, minimum=1, maximum=32),
            depth=_int_env("STOCKFISH_DEPTH", cls.depth, minimum=1, maximum=30),
            time_limit_seconds=_float_env("STOCKFISH_TIME_LIMIT_SECONDS", cls.time_limit_seconds, minimum=0.0, maximum=10.0),
            multipv=_int_env("STOCKFISH_MULTIPV", cls.multipv, minimum=1, maximum=128),
            random_seed=_optional_int_env("STOCKFISH_RANDOM_SEED"),
            illegal_sample_score=_float_env("STOCKFISH_ILLEGAL_SAMPLE_SCORE", cls.illegal_sample_score),
            missing_multipv_margin=_float_env("STOCKFISH_MISSING_MULTIPV_MARGIN", cls.missing_multipv_margin, minimum=0.0),
        )


def ranked_actions(
    belief: BeliefState,
    *,
    config: StockfishConfig | None = None,
    engine_factory: EngineFactory | None = None,
) -> tuple[str, ...]:
    """Return candidate UCI moves ranked by sampled Stockfish analysis."""

    actions = tuple(dict.fromkeys(move for move in belief.legal_actions if isinstance(move, str)))
    if not actions:
        return ()

    fallback = fallback_ranked_actions(belief)
    config = config or StockfishConfig.from_env()
    if not config.enabled:
        return fallback

    engine_factory = engine_factory or _open_engine
    try:
        engine = engine_factory(config.stockfish_path)
    except (FileNotFoundError, PermissionError, OSError, chess.engine.EngineError) as exc:
        logger.warning("Stockfish unavailable; using fallback ranking: %s", exc)
        return fallback

    try:
        ranked = ranked_actions_with_engine(belief, engine=engine, config=config, fallback=fallback)
    except (OSError, chess.engine.EngineError, chess.engine.EngineTerminatedError) as exc:
        logger.warning("Stockfish analysis failed; using fallback ranking: %s", exc)
        ranked = fallback
    finally:
        try:
            engine.quit()
        except (OSError, chess.engine.EngineError):
            pass
    return ranked


def ranked_actions_with_engine(
    belief: BeliefState,
    *,
    engine: Any,
    config: StockfishConfig,
    fallback: Sequence[str] | None = None,
) -> tuple[str, ...]:
    actions = tuple(dict.fromkeys(move for move in belief.legal_actions if isinstance(move, str)))
    fallback_order = {uci: index for index, uci in enumerate(fallback or fallback_ranked_actions(belief))}
    seed = config.random_seed
    if seed is None:
        seed = hash((belief.game_id, belief.ply, belief.visible_fen, belief.legal_actions)) & 0xFFFFFFFF
    rng = random.Random(seed)
    boards = sample_hypothesis_boards(belief, count=config.sample_count, rng=rng)
    if not boards:
        return tuple(fallback_order)

    totals = {uci: 0.0 for uci in actions}
    observations = {uci: 0 for uci in actions}
    limit = chess.engine.Limit(
        depth=config.depth,
        time=config.time_limit_seconds if config.time_limit_seconds > 0 else None,
    )

    for board in boards:
        sample_scores = _sample_scores(
            belief,
            board=board,
            actions=actions,
            engine=engine,
            limit=limit,
            multipv=config.multipv,
            illegal_sample_score=config.illegal_sample_score,
            missing_multipv_margin=config.missing_multipv_margin,
        )
        if not sample_scores:
            continue
        for uci, score in sample_scores.items():
            totals[uci] += score
            observations[uci] += 1

    if not any(observations.values()):
        return tuple(fallback_order)

    def sort_key(uci: str) -> tuple[float, int, str]:
        count = observations[uci] or 1
        average = totals[uci] / count
        return (-average, fallback_order.get(uci, 100000), uci)

    return tuple(sorted(actions, key=sort_key))


def sample_hypothesis_boards(
    belief: BeliefState,
    *,
    count: int,
    rng: random.Random,
) -> tuple[chess.Board, ...]:
    boards: list[chess.Board] = []
    seen: set[str] = set()
    attempts = max(count * 8, count)
    for _ in range(attempts):
        board = build_hypothesis_board(belief, rng=rng)
        if board is None:
            continue
        key = board.board_fen()
        if key in seen:
            continue
        seen.add(key)
        boards.append(board)
        if len(boards) >= count:
            break
    return tuple(boards)


def build_hypothesis_board(belief: BeliefState, *, rng: random.Random) -> chess.Board | None:
    board = visible_board(belief.visible_fen)
    if board is None:
        return None

    board.turn = belief.color
    board.castling_rights = 0
    board.ep_square = None
    for square, piece in list(board.piece_map().items()):
        if piece.color != belief.color:
            board.remove_piece_at(square)

    opponent = not belief.color
    remaining, pawns = opponent_material_counts(opponent=opponent, material_summary=belief.material_summary)
    if remaining <= 0:
        return board

    occupied = set(board.piece_map())
    king_square = _sample_square(
        belief.opponent_king,
        candidates=_king_candidates(board=board, color=belief.color, occupied=occupied),
        rng=rng,
    )
    if king_square is None:
        return None
    board.set_piece_at(king_square, chess.Piece(chess.KING, opponent))
    occupied.add(king_square)

    pawn_squares = _sample_squares(
        belief.opponent_pawns,
        count=pawns,
        candidates=_pawn_candidates(opponent=opponent, occupied=occupied),
        rng=rng,
    )
    for square in pawn_squares:
        board.set_piece_at(square, chess.Piece(chess.PAWN, opponent))
        occupied.add(square)

    non_pawn_count = max(0, remaining - pawns - 1)
    piece_squares = _sample_squares(
        belief.opponent_pieces,
        count=non_pawn_count,
        candidates=[square for square in chess.SQUARES if square not in occupied],
        rng=rng,
    )
    for square, piece_type in zip(piece_squares, _opponent_piece_inventory(non_pawn_count), strict=False):
        board.set_piece_at(square, chess.Piece(piece_type, opponent))

    if board.king(belief.color) is None or board.king(opponent) is None:
        return None
    return board


def fallback_ranked_actions(belief: BeliefState) -> tuple[str, ...]:
    actions = tuple(dict.fromkeys(move for move in belief.legal_actions if isinstance(move, str)))
    board = visible_board(belief.visible_fen)
    latest_capture = _latest_capture_square(belief.referee_log)
    scores = [(uci, _fallback_score(belief, board=board, uci=uci, latest_capture=latest_capture)) for uci in actions]
    return tuple(uci for uci, _score in sorted(scores, key=lambda item: (-item[1], item[0])))


def _sample_scores(
    belief: BeliefState,
    *,
    board: chess.Board,
    actions: Sequence[str],
    engine: Any,
    limit: chess.engine.Limit,
    multipv: int,
    illegal_sample_score: float,
    missing_multipv_margin: float,
) -> dict[str, float]:
    root_moves: list[chess.Move] = []
    scores = {uci: illegal_sample_score for uci in actions}
    legal_uci: set[str] = set()
    legal_moves = set(board.legal_moves)
    for uci in actions:
        move = parse_move(uci)
        if move is not None and move in legal_moves:
            root_moves.append(move)
            legal_uci.add(uci)

    if not root_moves:
        return scores

    result = engine.analyse(
        board,
        limit,
        root_moves=root_moves,
        multipv=min(max(1, multipv), len(root_moves)),
    )
    infos = result if isinstance(result, list) else [result]
    returned: set[str] = set()
    returned_scores: list[float] = []
    for info in infos:
        move = _pv_move(info)
        score = _engine_score(info, color=belief.color)
        if move is None or score is None:
            continue
        uci = move.uci()
        if uci not in legal_uci:
            continue
        scores[uci] = score
        returned.add(uci)
        returned_scores.append(score)

    floor = (min(returned_scores) - missing_multipv_margin) if returned_scores else 0.0
    for uci in legal_uci - returned:
        scores[uci] = floor
    return scores


def _pv_move(info: Mapping[str, Any]) -> chess.Move | None:
    pv = info.get("pv")
    if not isinstance(pv, list) or not pv:
        return None
    move = pv[0]
    return move if isinstance(move, chess.Move) else None


def _engine_score(info: Mapping[str, Any], *, color: chess.Color) -> float | None:
    score = info.get("score")
    if not isinstance(score, chess.engine.PovScore):
        return None
    value = score.pov(color).score(mate_score=100000)
    return float(value) if value is not None else None


def _fallback_score(
    belief: BeliefState,
    *,
    board: chess.Board | None,
    uci: str,
    latest_capture: chess.Square | None,
) -> float:
    move = parse_move(uci)
    if move is None:
        return -10000.0
    score = 0.0
    piece = board.piece_at(move.from_square) if board is not None else None
    if piece is None or piece.color != belief.color:
        score -= 250.0
    else:
        score += _development_score(board=board, color=belief.color, move=move, piece=piece)
        if move.promotion:
            score += PIECE_VALUES.get(move.promotion, 0.0)
        if piece.piece_type == chess.PAWN and chess.square_file(move.from_square) != chess.square_file(move.to_square):
            score += 90.0
        if latest_capture is not None and move.to_square == latest_capture:
            score += 260.0
    score += _centrality(move.to_square) * 4.0
    score += _move_distance(uci) * 2.0
    return score


def _development_score(*, board: chess.Board, color: chess.Color, move: chess.Move, piece: chess.Piece) -> float:
    if piece.piece_type == chess.PAWN:
        direction = 1 if color == chess.WHITE else -1
        rank_gain = (chess.square_rank(move.to_square) - chess.square_rank(move.from_square)) * direction
        return (rank_gain * 12.0) + _centrality(move.to_square)
    if piece.piece_type in {chess.KNIGHT, chess.BISHOP} and _is_home_minor(color=color, square=move.from_square):
        return 28.0 + _centrality(move.to_square)
    if piece.piece_type == chess.ROOK and _file_has_no_own_pawns(board=board, color=color, file=chess.square_file(move.to_square)):
        return 16.0
    return _centrality(move.to_square) - _centrality(move.from_square)


def _king_candidates(*, board: chess.Board, color: chess.Color, occupied: set[chess.Square]) -> list[chess.Square]:
    own_king = board.king(color)
    candidates: list[chess.Square] = []
    for square in chess.SQUARES:
        if square in occupied:
            continue
        if own_king is not None and chess.square_distance(square, own_king) <= 1:
            continue
        if board.is_attacked_by(color, square):
            continue
        candidates.append(square)
    return candidates


def _pawn_candidates(*, opponent: chess.Color, occupied: set[chess.Square]) -> list[chess.Square]:
    del opponent
    return [square for square in chess.SQUARES if square not in occupied and chess.square_rank(square) not in {0, 7}]


def _sample_squares(
    weights: Sequence[float],
    *,
    count: int,
    candidates: Sequence[chess.Square],
    rng: random.Random,
) -> list[chess.Square]:
    selected: list[chess.Square] = []
    remaining = list(candidates)
    for _ in range(max(0, min(count, len(remaining)))):
        square = _sample_square(weights, candidates=remaining, rng=rng)
        if square is None:
            break
        selected.append(square)
        remaining.remove(square)
    return selected


def _sample_square(
    weights: Sequence[float],
    *,
    candidates: Sequence[chess.Square],
    rng: random.Random,
) -> chess.Square | None:
    if not candidates:
        return None
    weighted = [(square, max(0.0, float(weights[square]) if square < len(weights) else 0.0)) for square in candidates]
    total = sum(weight for _square, weight in weighted)
    if total <= 0:
        return rng.choice(list(candidates))
    roll = rng.random() * total
    cumulative = 0.0
    for square, weight in weighted:
        cumulative += weight
        if roll <= cumulative:
            return square
    return weighted[-1][0]


def _opponent_piece_inventory(count: int) -> tuple[chess.PieceType, ...]:
    standard = (chess.QUEEN, chess.ROOK, chess.ROOK, chess.BISHOP, chess.BISHOP, chess.KNIGHT, chess.KNIGHT)
    return standard[: max(0, min(count, len(standard)))]


def _latest_capture_square(referee_log: Sequence[Mapping[str, Any]]) -> chess.Square | None:
    for item in reversed(referee_log):
        value = item.get("capture_square")
        if isinstance(value, str):
            try:
                return chess.parse_square(value.lower())
            except ValueError:
                pass
    return None


def _is_home_minor(*, color: chess.Color, square: chess.Square) -> bool:
    home_rank = 0 if color == chess.WHITE else 7
    return chess.square_rank(square) == home_rank and chess.square_file(square) in {1, 2, 5, 6}


def _file_has_no_own_pawns(*, board: chess.Board, color: chess.Color, file: int) -> bool:
    for rank in range(8):
        if board.piece_at(chess.square(file, rank)) == chess.Piece(chess.PAWN, color):
            return False
    return True


def _centrality(square: chess.Square) -> float:
    file = chess.square_file(square)
    rank = chess.square_rank(square)
    return 7.0 - (abs(file - 3.5) + abs(rank - 3.5))


def _move_distance(uci: str) -> int:
    if len(uci) < 4:
        return -1
    try:
        start = chess.parse_square(uci[:2])
        end = chess.parse_square(uci[2:4])
    except ValueError:
        return -1
    return max(abs(chess.square_file(end) - chess.square_file(start)), abs(chess.square_rank(end) - chess.square_rank(start)))


def _open_engine(path: str) -> chess.engine.SimpleEngine:
    return chess.engine.SimpleEngine.popen_uci(path)


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _int_env(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name, str(default)).strip()
    try:
        parsed = int(raw)
    except ValueError:
        return default
    return min(maximum, max(minimum, parsed))


def _optional_int_env(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _float_env(
    name: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    raw = os.environ.get(name, str(default)).strip()
    try:
        parsed = float(raw)
    except ValueError:
        return default
    if minimum is not None:
        parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed
