"""Public-state belief helpers for the Stockfish Wild 16 bot."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import re
from typing import Any, Mapping, Sequence

import chess


WILD16_RULESET = "wild16"
CAPTURE_SQUARE_PATTERN = re.compile(r"\b([A-H][1-8])\b", re.IGNORECASE)
CHECK_CODES = {
    "CHECK_RANK",
    "CHECK_FILE",
    "CHECK_LONG_DIAGONAL",
    "CHECK_SHORT_DIAGONAL",
    "CHECK_KNIGHT",
    "CHECK_DOUBLE",
}


@dataclass(frozen=True)
class OpponentPriors:
    king: tuple[float, ...]
    pawns: tuple[float, ...]
    pieces: tuple[float, ...]


@dataclass(frozen=True)
class BeliefState:
    """Player-visible state plus coarse opponent probability matrices."""

    color: chess.Color
    visible_fen: str
    legal_actions: tuple[str, ...]
    ruleset: str = WILD16_RULESET
    ply: int = 1
    game_id: str = ""
    game_state: str = ""
    turn: str | None = None
    possible_actions: tuple[str, ...] = ()
    material_summary: dict[str, Any] = field(default_factory=dict)
    referee_log: tuple[dict[str, Any], ...] = ()
    observed_referee_log_size: int = 0
    opponent_king: tuple[float, ...] = field(default_factory=lambda: (0.0,) * 64)
    opponent_pawns: tuple[float, ...] = field(default_factory=lambda: (0.0,) * 64)
    opponent_pieces: tuple[float, ...] = field(default_factory=lambda: (0.0,) * 64)

    def __post_init__(self) -> None:
        if self.ruleset != WILD16_RULESET:
            raise ValueError(f"only {WILD16_RULESET!r} is supported")
        if len(self.opponent_king) != 64:
            raise ValueError("opponent_king must contain 64 probabilities")
        if len(self.opponent_pawns) != 64:
            raise ValueError("opponent_pawns must contain 64 probabilities")
        if len(self.opponent_pieces) != 64:
            raise ValueError("opponent_pieces must contain 64 probabilities")
        if self.observed_referee_log_size < 0:
            raise ValueError("observed_referee_log_size must be non-negative")

    @property
    def your_fen(self) -> str:
        return self.visible_fen

    @classmethod
    def from_api_state(cls, state: Mapping[str, Any], *, ruleset: str | None = None) -> "BeliefState":
        color_name = str(state.get("your_color") or "").strip().lower()
        if color_name not in {"white", "black"}:
            raise ValueError("state must include your_color as white or black")

        color = chess.WHITE if color_name == "white" else chess.BLACK
        visible_fen = str(state.get("your_fen") or state.get("visible_fen") or "")
        allowed_moves = state.get("allowed_moves") if isinstance(state.get("allowed_moves"), list) else []
        possible_actions = state.get("possible_actions") if isinstance(state.get("possible_actions"), list) else []
        material_summary = state.get("material_summary") if isinstance(state.get("material_summary"), dict) else {}
        referee_log = state.get("referee_log") if isinstance(state.get("referee_log"), list) else []
        priors = opponent_priors(visible_fen=visible_fen, color=color, material_summary=material_summary)

        return cls(
            color=color,
            visible_fen=visible_fen,
            legal_actions=tuple(move for move in allowed_moves if isinstance(move, str)),
            ruleset=str(ruleset or state.get("rule_variant") or WILD16_RULESET),
            ply=_bounded_int(state.get("ply_count") or state.get("move_number"), default=1, minimum=1, maximum=10000),
            game_id=str(state.get("game_id") or ""),
            game_state=str(state.get("state") or ""),
            turn=str(state.get("turn")) if state.get("turn") is not None else None,
            possible_actions=tuple(action for action in possible_actions if isinstance(action, str)),
            material_summary=material_summary,
            referee_log=tuple(item for item in referee_log if isinstance(item, dict)),
            opponent_king=priors.king,
            opponent_pawns=priors.pawns,
            opponent_pieces=priors.pieces,
        )


def serialize_belief(belief: BeliefState) -> dict[str, Any]:
    return {
        "game_id": belief.game_id,
        "ruleset": belief.ruleset,
        "color": "white" if belief.color == chess.WHITE else "black",
        "ply": belief.ply,
        "your_fen": belief.your_fen,
        "legal_actions": list(belief.legal_actions),
        "possible_actions": list(belief.possible_actions),
        "observed_referee_log_size": belief.observed_referee_log_size,
        "opponent_king": list(belief.opponent_king),
        "opponent_pawns": list(belief.opponent_pawns),
        "opponent_pieces": list(belief.opponent_pieces),
    }


def restore_belief_snapshot(belief: BeliefState, snapshot: Mapping[str, Any] | None) -> BeliefState:
    """Carry compatible persisted matrices into the current API state."""

    if not isinstance(snapshot, Mapping) or not _snapshot_is_compatible(belief, snapshot):
        return apply_referee_log_evidence(belief)

    restored = replace(
        belief,
        observed_referee_log_size=_bounded_log_size(snapshot.get("observed_referee_log_size")),
        opponent_king=_matrix_from_snapshot(snapshot.get("opponent_king"), fallback=belief.opponent_king),
        opponent_pawns=_matrix_from_snapshot(snapshot.get("opponent_pawns"), fallback=belief.opponent_pawns),
        opponent_pieces=_matrix_from_snapshot(snapshot.get("opponent_pieces"), fallback=belief.opponent_pieces),
    )
    return apply_referee_log_evidence(_renormalize_to_public_state(restored))


def apply_referee_log_evidence(belief: BeliefState) -> BeliefState:
    """Apply newly visible public capture evidence once."""

    start = min(belief.observed_referee_log_size, len(belief.referee_log))
    updated = belief
    for item in belief.referee_log[start:]:
        square = _capture_square(item)
        if square is not None:
            updated = _apply_capture(updated, square=square, captured_kind=_capture_kind(item), decrement=False)

    if updated.observed_referee_log_size != len(belief.referee_log):
        updated = replace(updated, observed_referee_log_size=len(belief.referee_log))
    return updated


def apply_move_result_evidence(belief: BeliefState, *, uci: str, result: Mapping[str, Any]) -> BeliefState:
    """Update belief from the immediate response to one of our attempts."""

    board = visible_board(belief.visible_fen)
    move = parse_move(uci)
    piece = board.piece_at(move.from_square) if board is not None and move is not None else None
    if board is None or move is None or piece is None:
        return _mark_attempt_observed(belief)

    if not bool(result.get("move_done")):
        if piece.piece_type == chess.PAWN:
            if chess.square_file(move.from_square) != chess.square_file(move.to_square):
                return _mark_attempt_observed(_suppress_occupancy(belief, [move.to_square]))
            return _mark_attempt_observed(_boost_occupancy(belief, _path_and_target(move=move, piece=piece), amount=0.7))
        if piece.piece_type in {chess.BISHOP, chess.ROOK, chess.QUEEN}:
            path = list(chess.SquareSet(chess.between(move.from_square, move.to_square)))
            if path:
                return _mark_attempt_observed(_boost_occupancy(belief, path, amount=0.75))
        return _mark_attempt_observed(_suppress_occupancy(belief, [move.to_square], factor=0.45))

    updated = belief
    capture_square = _capture_square(result)
    if capture_square is not None:
        updated = _apply_capture(updated, square=capture_square, captured_kind=_capture_kind(result), decrement=True)
    else:
        updated = _suppress_occupancy(updated, _path_and_target(move=move, piece=piece))

    updated = _apply_check_evidence(updated, board=board, move=move, piece=piece, gave_check=_has_check_announcement(result))
    return _mark_attempt_observed(updated)


def opponent_priors(*, visible_fen: str, color: chess.Color, material_summary: dict[str, Any]) -> OpponentPriors:
    board = visible_board(visible_fen)
    occupied = set(board.piece_map()) if board is not None else set()
    opponent = not color
    remaining, pawns = opponent_material_counts(opponent=opponent, material_summary=material_summary)
    non_pawn_pieces = max(0, remaining - pawns - (1 if remaining > 0 else 0))

    return OpponentPriors(
        king=_normalize(_king_weights(opponent=opponent, occupied=occupied), 1.0 if remaining > 0 else 0.0),
        pawns=_normalize(_pawn_weights(opponent=opponent, occupied=occupied), float(pawns)),
        pieces=_normalize(_piece_weights(opponent=opponent, occupied=occupied), float(non_pawn_pieces)),
    )


def opponent_material_counts(*, opponent: chess.Color, material_summary: Mapping[str, Any]) -> tuple[int, int]:
    side_name = "white" if opponent == chess.WHITE else "black"
    side = material_summary.get(side_name) if isinstance(material_summary, Mapping) else None
    if not isinstance(side, Mapping):
        return 16, 8

    remaining = _bounded_int(side.get("pieces_remaining"), default=16, minimum=0, maximum=16)
    pawns_captured = side.get("pawns_captured")
    if pawns_captured is None:
        pawns = min(8, remaining)
    else:
        pawns = 8 - _bounded_int(pawns_captured, default=0, minimum=0, maximum=8)
        pawns = min(max(0, pawns), remaining)
    return remaining, pawns


def visible_board(fen: str) -> chess.Board | None:
    try:
        board = chess.Board(fen)
    except ValueError:
        return None
    board.clear_stack()
    return board


def parse_move(uci: str) -> chess.Move | None:
    try:
        return chess.Move.from_uci(uci)
    except ValueError:
        return None


def _snapshot_is_compatible(belief: BeliefState, snapshot: Mapping[str, Any]) -> bool:
    ruleset = snapshot.get("ruleset")
    if isinstance(ruleset, str) and ruleset and ruleset != belief.ruleset:
        return False

    color = snapshot.get("color")
    if isinstance(color, str) and color in {"white", "black"}:
        expected = "white" if belief.color == chess.WHITE else "black"
        if color != expected:
            return False

    game_id = snapshot.get("game_id")
    return not (isinstance(game_id, str) and game_id and belief.game_id and game_id != belief.game_id)


def _matrix_from_snapshot(value: Any, *, fallback: tuple[float, ...]) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 64:
        return fallback
    try:
        return tuple(max(0.0, float(item)) for item in value)
    except (TypeError, ValueError):
        return fallback


def _mark_attempt_observed(belief: BeliefState) -> BeliefState:
    return replace(belief, observed_referee_log_size=max(belief.observed_referee_log_size, len(belief.referee_log) + 1))


def _apply_capture(
    belief: BeliefState,
    *,
    square: chess.Square,
    captured_kind: str | None,
    decrement: bool,
) -> BeliefState:
    king = list(belief.opponent_king)
    pawns = list(belief.opponent_pawns)
    pieces = list(belief.opponent_pieces)
    totals = _effective_totals(belief)
    captured_kind = (captured_kind or "").upper()

    if captured_kind == "PAWN":
        pawns[square] = 0.0
        if decrement:
            totals["pawns"] = max(0.0, totals["pawns"] - 1.0)
    elif captured_kind in {"PIECE", "KNIGHT", "BISHOP", "ROOK", "QUEEN"}:
        pieces[square] = 0.0
        if decrement:
            totals["pieces"] = max(0.0, totals["pieces"] - 1.0)
    elif pawns[square] >= pieces[square]:
        pawns[square] = 0.0
        pieces[square] *= 0.25
        if decrement:
            totals["pawns"] = max(0.0, totals["pawns"] - 1.0)
    else:
        pieces[square] = 0.0
        pawns[square] *= 0.25
        if decrement:
            totals["pieces"] = max(0.0, totals["pieces"] - 1.0)

    king[square] *= 0.08
    return _renormalize_to_public_state(
        replace(belief, opponent_king=tuple(king), opponent_pawns=tuple(pawns), opponent_pieces=tuple(pieces)),
        totals=totals,
    )


def _suppress_occupancy(
    belief: BeliefState,
    squares: Sequence[chess.Square],
    *,
    factor: float = 0.08,
) -> BeliefState:
    return _renormalize_to_public_state(
        replace(
            belief,
            opponent_king=_multiply_squares(belief.opponent_king, squares, factor=factor),
            opponent_pawns=_multiply_squares(belief.opponent_pawns, squares, factor=factor),
            opponent_pieces=_multiply_squares(belief.opponent_pieces, squares, factor=factor),
        ),
        totals=_effective_totals(belief),
    )


def _boost_occupancy(belief: BeliefState, squares: Sequence[chess.Square], *, amount: float) -> BeliefState:
    return _renormalize_to_public_state(
        replace(
            belief,
            opponent_pawns=_add_to_squares(belief.opponent_pawns, squares, amount=amount),
            opponent_pieces=_add_to_squares(belief.opponent_pieces, squares, amount=amount),
        ),
        totals=_effective_totals(belief),
    )


def _apply_check_evidence(
    belief: BeliefState,
    *,
    board: chess.Board,
    move: chess.Move,
    piece: chess.Piece,
    gave_check: bool,
) -> BeliefState:
    attacks = tuple(_attacks_after_move(board=board, move=move, piece=piece))
    if not attacks:
        return belief
    king = _add_to_squares(belief.opponent_king, attacks, amount=0.45) if gave_check else _multiply_squares(
        belief.opponent_king,
        attacks,
        factor=0.72,
    )
    return _renormalize_to_public_state(replace(belief, opponent_king=king), totals=_effective_totals(belief))


def _renormalize_to_public_state(
    belief: BeliefState,
    *,
    totals: Mapping[str, float] | None = None,
) -> BeliefState:
    public_totals = dict(totals or _public_totals(belief))
    occupied = _visible_occupied_squares(belief.visible_fen)
    fallback = opponent_priors(visible_fen=belief.visible_fen, color=belief.color, material_summary=belief.material_summary)
    king = _normalize(_mask_occupied(list(belief.opponent_king), occupied), public_totals.get("king", 0.0))
    pawns = _normalize(_mask_occupied(list(belief.opponent_pawns), occupied), public_totals.get("pawns", 0.0))
    pieces = _normalize(_mask_occupied(list(belief.opponent_pieces), occupied), public_totals.get("pieces", 0.0))
    if public_totals.get("king", 0.0) > 0 and sum(king) <= 0:
        king = fallback.king
    if public_totals.get("pawns", 0.0) > 0 and sum(pawns) <= 0:
        pawns = fallback.pawns
    if public_totals.get("pieces", 0.0) > 0 and sum(pieces) <= 0:
        pieces = fallback.pieces
    return replace(belief, opponent_king=king, opponent_pawns=pawns, opponent_pieces=pieces)


def _public_totals(belief: BeliefState) -> dict[str, float]:
    remaining, pawns = opponent_material_counts(opponent=not belief.color, material_summary=belief.material_summary)
    king = 1.0 if remaining > 0 else 0.0
    return {"king": king, "pawns": float(pawns), "pieces": float(max(0, remaining - pawns - int(king)))}


def _effective_totals(belief: BeliefState) -> dict[str, float]:
    totals = _public_totals(belief)
    return {
        "king": totals["king"],
        "pawns": min(totals["pawns"], max(0.0, sum(belief.opponent_pawns))),
        "pieces": min(totals["pieces"], max(0.0, sum(belief.opponent_pieces))),
    }


def _visible_occupied_squares(fen: str) -> set[chess.Square]:
    board = visible_board(fen)
    return set(board.piece_map()) if board is not None else set()


def _mask_occupied(weights: list[float], occupied: set[chess.Square]) -> list[float]:
    return [0.0 if square in occupied else max(0.0, weights[square]) for square in chess.SQUARES]


def _king_weights(*, opponent: chess.Color, occupied: set[chess.Square]) -> list[float]:
    home_rank = 0 if opponent == chess.WHITE else 7
    weights: list[float] = []
    for square in chess.SQUARES:
        if square in occupied:
            weights.append(0.0)
            continue
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        home_distance = abs(rank - home_rank)
        edge_pressure = 1.3 if file in {0, 7} else 1.0
        weights.append(max(0.2, 3.5 - home_distance) * edge_pressure)
    return weights


def _pawn_weights(*, opponent: chess.Color, occupied: set[chess.Square]) -> list[float]:
    start_rank = 1 if opponent == chess.WHITE else 6
    promotion_rank = 7 if opponent == chess.WHITE else 0
    weights: list[float] = []
    for square in chess.SQUARES:
        if square in occupied:
            weights.append(0.0)
            continue
        rank = chess.square_rank(square)
        if rank in {0, 7}:
            weights.append(0.0)
            continue
        progress = abs(rank - start_rank)
        promotion_distance = abs(rank - promotion_rank)
        weights.append(max(0.2, 3.0 - (0.55 * progress)) + (0.15 / max(1, promotion_distance)))
    return weights


def _piece_weights(*, opponent: chess.Color, occupied: set[chess.Square]) -> list[float]:
    home_rank = 0 if opponent == chess.WHITE else 7
    weights: list[float] = []
    for square in chess.SQUARES:
        if square in occupied:
            weights.append(0.0)
            continue
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        home_bias = max(0.3, 3.0 - (0.7 * abs(rank - home_rank)))
        centrality = 1.0 + (0.2 * (3.5 - abs(file - 3.5)))
        weights.append(home_bias * centrality)
    return weights


def _normalize(weights: list[float], total: float) -> tuple[float, ...]:
    if total <= 0:
        return (0.0,) * 64
    weight_sum = sum(max(0.0, weight) for weight in weights)
    if weight_sum <= 0:
        return (0.0,) * 64
    scale = total / weight_sum
    return tuple(max(0.0, weight) * scale for weight in weights)


def _capture_square(item: Mapping[str, Any]) -> chess.Square | None:
    value = item.get("capture_square")
    if isinstance(value, str):
        parsed = _parse_square(value)
        if parsed is not None:
            return parsed
    announcement = item.get("announcement")
    if isinstance(announcement, str) and "captur" in announcement.lower():
        match = CAPTURE_SQUARE_PATTERN.search(announcement)
        if match:
            return _parse_square(match.group(1))
    return None


def _capture_kind(item: Mapping[str, Any]) -> str | None:
    for key in ("captured_piece_announcement", "captured_kind", "capture_kind"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().upper()
    return None


def _has_check_announcement(item: Mapping[str, Any]) -> bool:
    special = item.get("special_announcement")
    if isinstance(special, str) and special.upper() in CHECK_CODES:
        return True
    checks = item.get("checks")
    if isinstance(checks, list) and checks:
        return True
    announcement = item.get("announcement")
    return isinstance(announcement, str) and "check" in announcement.lower()


def _parse_square(value: str) -> chess.Square | None:
    try:
        return chess.parse_square(value.lower())
    except ValueError:
        return None


def _path_and_target(*, move: chess.Move, piece: chess.Piece) -> list[chess.Square]:
    squares = list(chess.SquareSet(chess.between(move.from_square, move.to_square)))
    squares.append(move.to_square)
    if piece.piece_type == chess.PAWN and abs(chess.square_rank(move.to_square) - chess.square_rank(move.from_square)) == 2:
        direction = 1 if piece.color == chess.WHITE else -1
        squares.insert(0, chess.square(chess.square_file(move.from_square), chess.square_rank(move.from_square) + direction))
    return list(dict.fromkeys(squares))


def _attacks_after_move(*, board: chess.Board, move: chess.Move, piece: chess.Piece) -> chess.SquareSet:
    if piece.piece_type == chess.PAWN:
        return chess.SquareSet(chess.BB_PAWN_ATTACKS[piece.color][move.to_square])

    projected = board.copy(stack=False)
    projected.remove_piece_at(move.from_square)
    projected.set_piece_at(move.to_square, chess.Piece(move.promotion or piece.piece_type, piece.color))
    return projected.attacks(move.to_square)


def _multiply_squares(values: tuple[float, ...], squares: Sequence[chess.Square], *, factor: float) -> tuple[float, ...]:
    updated = list(values)
    for square in squares:
        updated[square] *= factor
    return tuple(updated)


def _add_to_squares(values: tuple[float, ...], squares: Sequence[chess.Square], *, amount: float) -> tuple[float, ...]:
    updated = list(values)
    for square in squares:
        updated[square] += amount
    return tuple(updated)


def _bounded_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return min(maximum, max(minimum, parsed))


def _bounded_log_size(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0
