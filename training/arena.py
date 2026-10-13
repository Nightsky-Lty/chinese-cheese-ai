"""按固定开局换先评估 NNUE，并用配对结果控制冠军晋升。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import queue
import random
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OPENINGS = PROJECT_ROOT / "training/examples/arena_openings.json"
DEFAULT_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"
REPORT_FORMAT = "xiangqi-arena-v4"
PROMOTION_POLICY = "same-color-win-improvement-v1"
BASELINE_PROMOTION_POLICY = "color-balanced-paired-development-v1"
DEFAULT_REQUIRED_WIN_IMPROVEMENT = 0.10
DEFAULT_REQUIRED_SCORE = 0.55
DEFAULT_MAX_UNFINISHED = 10
PROMOTION_MAX_UNFINISHED_FRACTION = 0.10
PROMOTION_MIN_COMPLETE_PAIR_FRACTION = 0.80
PROMOTION_MIN_GAMES = 100
PROMOTION_MIN_INDEPENDENT_PAIRS = 50
DEFAULT_DEPTH_SEARCH_DEPTH = 4
DEFAULT_TIME_SEARCH_DEPTH_CAP = 64


def resolve_search_depth(search_mode: str, explicit_depth: int | None) -> int:
    """Resolve omitted CLI depth without constraining timed searches by accident."""

    if search_mode not in ("depth", "time"):
        raise ValueError("search mode must be 'depth' or 'time'")
    if explicit_depth is not None:
        return explicit_depth
    return (
        DEFAULT_TIME_SEARCH_DEPTH_CAP
        if search_mode == "time" else DEFAULT_DEPTH_SEARCH_DEPTH
    )


@dataclass(frozen=True)
class Opponent:
    """一个固定评估对手；network=None 表示 C++ 手工评估。"""

    name: str
    network: Path | None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ProtocolEngine:
    """通过 protocol.cpp 的逐行协议控制引擎，并对每条命令设置期限。"""

    def __init__(
        self, executable: Path, network: Path | None, timeout_seconds: float,
    ) -> None:
        command = [str(executable.resolve())]
        if network is not None:
            command.extend(["--nnue", str(network.resolve())])
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
            start_new_session=True,
        )
        self.timeout_seconds = timeout_seconds
        self.responses: queue.Queue[str | None] = queue.Queue()
        self.stderr_tail: list[str] = []
        self._stdout_thread = threading.Thread(
            target=self._read_stdout, name="xiangqi-protocol-stdout", daemon=True
        )
        self._stderr_thread = threading.Thread(
            target=self._read_stderr, name="xiangqi-protocol-stderr", daemon=True
        )
        self._stdout_thread.start()
        self._stderr_thread.start()
        try:
            greeting = self._next_response(max(timeout_seconds, 5.0))
        except BaseException:
            self._close_streams()
            raise
        if greeting != "ready xiangqi-protocol 1":
            self.terminate()
            self._close_streams()
            raise RuntimeError(f"engine did not start: {greeting!r}")
        self.transcript: list[dict[str, str]] = []

    def _read_stdout(self) -> None:
        assert self.process.stdout is not None
        try:
            for line in self.process.stdout:
                self.responses.put(line.rstrip("\r\n"))
        finally:
            self.responses.put(None)

    def _read_stderr(self) -> None:
        assert self.process.stderr is not None
        for line in self.process.stderr:
            self.stderr_tail.append(line.rstrip("\r\n"))
            if len(self.stderr_tail) > 100:
                del self.stderr_tail[: len(self.stderr_tail) - 100]

    def _next_response(self, timeout_seconds: float | None = None) -> str:
        deadline = self.timeout_seconds if timeout_seconds is None else timeout_seconds
        try:
            response = self.responses.get(timeout=deadline)
        except queue.Empty as error:
            self.terminate()
            raise TimeoutError(
                f"protocol response timed out after {deadline:g}s"
            ) from error
        if response is None:
            returncode = self.process.poll()
            raise RuntimeError(
                f"engine closed protocol output (exit={returncode}); "
                f"stderr={self.stderr_tail[-8:]}"
            )
        return response

    def ask(self, command: str) -> str:
        if self.process.poll() is not None:
            raise RuntimeError(f"engine exited with status {self.process.returncode}")
        assert self.process.stdin is not None
        started = time.monotonic()
        try:
            self.process.stdin.write(command + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            raise RuntimeError(f"failed to send protocol command {command!r}") from error
        response = self._next_response()
        self.transcript.append({"command": command, "response": response})
        if response.startswith("error "):
            raise RuntimeError(f"protocol command {command!r} failed: {response}")
        if time.monotonic() - started > self.timeout_seconds:
            raise TimeoutError(f"protocol command {command!r} exceeded its deadline")
        return response

    def terminate(self) -> None:
        if self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except (ProcessLookupError, AttributeError):
                self.process.kill()
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()

    def _close_streams(self) -> None:
        self._stdout_thread.join(timeout=1)
        self._stderr_thread.join(timeout=1)
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            if stream is not None and not stream.closed:
                stream.close()

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self.ask("quit")
                self.process.wait(timeout=2)
            except (OSError, RuntimeError, TimeoutError, subprocess.TimeoutExpired):
                self.terminate()
        if self.process.poll() is None:
            self.terminate()
        self._close_streams()


def load_openings(path: Path) -> tuple[str, list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("opening file must be a JSON object")
    initial_fen = payload.get("initial_fen", DEFAULT_FEN)
    raw_openings = payload.get("openings")
    if not isinstance(initial_fen, str) or not isinstance(raw_openings, list) or not raw_openings:
        raise ValueError("opening file requires initial_fen and a non-empty openings list")
    openings: list[dict[str, Any]] = []
    names: set[str] = set()
    for item in raw_openings:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise ValueError("each opening needs a string name and a move list")
        name, moves = item["name"], item.get("moves")
        if not name or name in names or not isinstance(moves, list):
            raise ValueError("opening names must be unique and moves must be a list")
        if any(not isinstance(move, str) or len(move) != 4 for move in moves):
            raise ValueError(f"opening {name!r} contains an invalid coordinate move")
        names.add(name)
        openings.append({"name": name, "moves": moves})
    return initial_fen, openings


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    location = (len(ordered) - 1) * fraction
    lower = int(location)
    upper = min(lower + 1, len(ordered) - 1)
    weight = location - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def paired_opening_interval(
    games: Sequence[dict[str, Any]], confidence: float = 0.95,
    bootstrap_samples: int = 10_000, seed: int = 0x5849414E,
) -> dict[str, Any]:
    """以同一开局红黑换先的一对为簇，报告保守界和 bootstrap 诊断。"""

    paired: dict[str, dict[int, dict[str, float]]] = {}
    for index, game in enumerate(games):
        if game.get("outcome") == "unfinished":
            continue
        side = game["candidate_side"]
        outcome = game["outcome"]
        score = 0.5 if outcome == "draw" else float(outcome == f"{side}_win")
        opening = str(game.get("opening", f"legacy-game-{index}"))
        pair_id = int(game.get("pair_id", index))
        paired.setdefault(opening, {}).setdefault(pair_id, {})[side] = score
    clusters = []
    for opening_pairs in paired.values():
        scores = [
            (pair["red"] + pair["black"]) / 2.0
            for pair in opening_pairs.values()
            if "red" in pair and "black" in pair
        ]
        if scores:
            clusters.append(sum(scores) / len(scores))
    alpha = 1.0 - confidence
    bootstrap_interval: dict[str, float] | None = None
    if len(clusters) >= 2:
        rng = random.Random(seed)
        means = [
            sum(rng.choice(clusters) for _ in clusters) / len(clusters)
            for _ in range(bootstrap_samples)
        ]
        bootstrap_interval = {
            "lower": percentile(means, alpha / 2.0),
            "upper": percentile(means, 1.0 - alpha / 2.0),
        }
    mean = sum(clusters) / len(clusters) if clusters else 0.0
    # Hoeffding is conservative for bounded paired-opening means and remains
    # non-degenerate when every observed pair has the same outcome.
    half_width = (
        math.sqrt(math.log(2.0 / alpha) / (2.0 * len(clusters)))
        if clusters else 1.0
    )
    return {
        "confidence": confidence,
        "method": "paired-opening-cluster-hoeffding-bound",
        "independent_openings": len(clusters),
        "mean": mean,
        "lower": max(0.0, mean - half_width),
        "upper": min(1.0, mean + half_width),
        "half_width": half_width,
        "assumption": (
            "nominal coverage treats distinct configured opening clusters as independent; "
            "shared root positions can still induce dependence"
        ),
        "paired_bootstrap_percentile": bootstrap_interval,
    }


def color_statistics(games: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """按候选执红/执黑统计；未完成局不进入胜率和得分率的分母。

    games 为带候选颜色和赛果的对局记录。win_fraction 为胜局/完成局，
    score_fraction 为（胜局 + 半数和局）/完成局；无完成局时返回 None。
    """

    result = {
        side: {"wins": 0, "draws": 0, "losses": 0, "unfinished": 0}
        for side in ("red", "black")
    }
    for game in games:
        side = game["candidate_side"]
        if side not in result:
            raise ValueError(f"unknown candidate side: {side}")
        outcome = game["outcome"]
        if outcome == "unfinished":
            key = "unfinished"
        elif outcome == "draw":
            key = "draws"
        elif outcome == f"{side}_win":
            key = "wins"
        elif outcome in ("red_win", "black_win"):
            key = "losses"
        else:
            raise ValueError(f"unknown game outcome: {outcome}")
        result[side][key] += 1
    for counts in result.values():
        decided = counts["wins"] + counts["draws"] + counts["losses"]
        counts.update(
            games=decided + counts["unfinished"], decided_games=decided,
            win_fraction=counts["wins"] / decided if decided else None,
            score_fraction=(counts["wins"] + 0.5 * counts["draws"]) / decided if decided else None,
        )
    return result


def format_color_statistics(counts: dict[str, Any]) -> str:
    """将一种颜色的统计格式化为可读日志；空分母用 n/a 表示。"""

    win = f"{counts['win_fraction']:.2%}" if counts["win_fraction"] is not None else "n/a"
    score = f"{counts['score_fraction']:.2%}" if counts["score_fraction"] is not None else "n/a"
    return (f"W/D/L={counts['wins']}/{counts['draws']}/{counts['losses']} "
            f"unfinished={counts['unfinished']} win={win} score={score}")


def summarize_games(
    games: list[dict[str, Any]], required_score: float, max_unfinished: int,
    min_games: int, min_independent_pairs: int = 50,
    *, required_win_improvement: float | None = None,
) -> dict[str, Any]:
    """按颜色报告成绩，并裁决同色胜率提升或初始基线门禁。

    required_win_improvement 是候选对对手同色胜率的绝对差（0.10 为
    10 个百分点），红黑都必须达到。为 None 时用 required_score 裁决
    初始基线的配对得分；有冠军时配对得分和区间只作诊断。
    max_unfinished 是步数上限未完成局的绝对额度，另有总局数 10% 上限。
    min_games/min_independent_pairs 不能低于正式样本下限；实际完整配对
    可随未完成额度降低，但至少保留开局要求的 80%。协议故障直接拒绝。
    """

    by_color = color_statistics(games)
    wins, losses, draws, unfinished = (
        sum(counts[key] for counts in by_color.values())
        for key in ("wins", "losses", "draws", "unfinished")
    )
    # 对手执红的对局就是候选执黑的对局，交换胜负即可得到同色对照。
    opponent_by_color = {
        side: {**by_color[other], "wins": by_color[other]["losses"],
               "losses": by_color[other]["wins"],
               "win_fraction": (by_color[other]["losses"] / by_color[other]["decided_games"]
                                if by_color[other]["decided_games"] else None),
               "score_fraction": (1.0 - by_color[other]["score_fraction"]
                                  if by_color[other]["score_fraction"] is not None else None)}
        for side, other in (("red", "black"), ("black", "red"))
    }
    win_improvement = {
        side: (by_color[side]["win_fraction"] - opponent_by_color[side]["win_fraction"]
               if by_color[side]["win_fraction"] is not None
               and opponent_by_color[side]["win_fraction"] is not None else None)
        for side in ("red", "black")
    }
    decided = wins + losses + draws
    score_fraction = (wins + 0.5 * draws) / decided if decided else 0.0
    interval = paired_opening_interval(games)
    complete_pairs = int(interval["independent_openings"])
    # Only a ply-cap truncation is tolerated; protocol errors invalidate the match.
    failed_games = sum(game["outcome"] == "unfinished" and game.get("reason") != "ply_cap"
                       for game in games)
    paired_score = interval["mean"] if complete_pairs else None
    red_score, black_score = (by_color[side]["score_fraction"] for side in ("red", "black"))
    balanced_score = ((red_score + black_score) / 2.0
                      if red_score is not None and black_score is not None else None)
    effective_max_unfinished = min(
        max_unfinished, math.floor(len(games) * PROMOTION_MAX_UNFINISHED_FRACTION)
    )
    effective_min_games = max(min_games, PROMOTION_MIN_GAMES)
    effective_min_independent_pairs = max(
        min_independent_pairs, PROMOTION_MIN_INDEPENDENT_PAIRS
    )
    # Each tolerated unfinished game can invalidate at most one paired opening.
    # 未完成局不能用作和棋；完整配对数仍是样本质量底线。
    required_complete_pairs = max(
        math.ceil(effective_min_independent_pairs * PROMOTION_MIN_COMPLETE_PAIR_FRACTION),
        effective_min_independent_pairs - effective_max_unfinished,
    )
    checks = {
        "insufficient_games": len(games) >= effective_min_games,
        "unbalanced_scheduled_colors": by_color["red"]["games"] == by_color["black"]["games"],
        "unfinished_limit_exceeded": unfinished <= effective_max_unfinished,
        "protocol_failure": failed_games == 0,
        "insufficient_complete_opening_pairs": complete_pairs >= required_complete_pairs,
    }
    if required_win_improvement is None:
        checks["paired_score_below_threshold"] = paired_score is not None and paired_score >= required_score
    else:
        if not 0.0 <= required_win_improvement <= 1.0:
            raise ValueError("required win improvement must be between 0 and 1")
        for side in ("red", "black"):
            improvement = win_improvement[side]
            # 只消除比例相减的浮点舍入误差，不放宽用户要求的提升幅度。
            checks[f"{side}_win_improvement_below_threshold"] = (
                improvement is not None and improvement + 1e-12 >= required_win_improvement
            )
    rejection_reasons = [reason for reason, satisfied in checks.items() if not satisfied]
    return {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "unfinished": unfinished,
        "games": len(games),
        "decided_games": decided,
        "complete_paired_openings": complete_pairs,
        "required_complete_paired_openings": required_complete_pairs,
        "failed_games": failed_games,
        "score_fraction": score_fraction,
        "win_fraction": wins / decided if decided else None,
        "by_color": by_color,
        "opponent_by_color": opponent_by_color,
        "color_balanced_score_fraction": balanced_score,
        "paired_score_fraction": paired_score,
        "gate_score_fraction": paired_score if required_win_improvement is None else None,
        "gate_metric": ("paired_score_fraction" if required_win_improvement is None
                        else "same_color_win_fraction_difference"),
        "gate_policy": BASELINE_PROMOTION_POLICY if required_win_improvement is None else PROMOTION_POLICY,
        "win_improvement_by_color": win_improvement,
        "required_win_improvement": required_win_improvement,
        "rejection_reasons": rejection_reasons,
        "score_fraction_ci95": interval,
        "required_score": required_score,
        "min_games": min_games,
        "min_independent_pairs": min_independent_pairs,
        "effective_min_games": effective_min_games,
        "effective_min_independent_pairs": effective_min_independent_pairs,
        "max_unfinished": max_unfinished,
        "effective_max_unfinished": effective_max_unfinished,
        "max_unfinished_fraction": PROMOTION_MAX_UNFINISHED_FRACTION,
        "passed": not rejection_reasons,
    }


def _result(engine: ProtocolEngine) -> tuple[str, str]:
    response = engine.ask("result")
    parts = response.split(" reason ", 1)
    if len(parts) != 2 or not parts[0].startswith("result "):
        raise RuntimeError(f"unexpected result response: {response!r}")
    outcome = parts[0].split(" ", 1)[1]
    if outcome not in ("ongoing", "red_win", "black_win", "draw"):
        raise RuntimeError(f"unknown protocol outcome: {response!r}")
    return outcome, parts[1]


def _side_to_move(fen: str) -> str:
    fields = fen.split()
    if len(fields) < 2 or fields[1] not in ("w", "r", "b"):
        raise ValueError("initial FEN must include side to move as w, r or b")
    return "red" if fields[1] in ("w", "r") else "black"


def _play_game(
    engines: dict[str, ProtocolEngine], initial_fen: str,
    candidate_side: str, opening: dict[str, Any], search_mode: str,
    depth: int, time_limit_ms: int | None, max_plies: int, pair_id: int,
) -> dict[str, Any]:
    name = opening["name"]
    opening_moves = list(opening["moves"])
    starting_side = _side_to_move(initial_fen)
    for engine in engines.values():
        engine.ask("position fen " + initial_fen)
    moves: list[str] = []
    for move in opening_moves:
        states = [engine.ask("move " + move) for engine in engines.values()]
        if states[0] != states[1]:
            raise RuntimeError("engine states diverged during opening")
        moves.append(move)
        verdicts = [_result(engine) for engine in engines.values()]
        if verdicts[0] != verdicts[1]:
            raise RuntimeError("engine verdicts diverged during opening")
        outcome, reason = verdicts[0]
        if outcome != "ongoing":
            return {
                "opening": name, "candidate_side": candidate_side,
                "pair_id": pair_id,
                "outcome": outcome, "reason": reason,
                "plies": len(moves), "moves": moves, "opening_moves": opening_moves,
            }

    outcome, reason = _result(engines["candidate"])
    other_result = _result(engines["opponent"])
    if (outcome, reason) != other_result:
        raise RuntimeError("engine verdicts diverged after opening")
    if outcome != "ongoing":
        return {
            "opening": name, "candidate_side": candidate_side,
            "pair_id": pair_id,
            "outcome": outcome, "reason": reason,
            "plies": len(moves), "moves": moves, "opening_moves": opening_moves,
        }

    for ply in range(len(opening_moves), max_plies):
        side = starting_side if ply % 2 == 0 else ("black" if starting_side == "red" else "red")
        player = "candidate" if side == candidate_side else "opponent"
        command = f"go depth {depth}"
        if search_mode == "time":
            assert time_limit_ms is not None
            command += f" movetime {time_limit_ms}"
        answer = engines[player].ask(command)
        parts = answer.split()
        if len(parts) < 2 or parts[0] != "bestmove":
            raise RuntimeError(f"unexpected search response: {answer}")
        if parts[1] == "none":
            outcome, reason = _result(engines[player])
            if outcome == "ongoing":
                raise RuntimeError("engine returned no move in an ongoing position")
            return {
                "opening": name, "candidate_side": candidate_side,
                "pair_id": pair_id,
                "outcome": outcome, "reason": reason,
                "plies": len(moves), "moves": moves, "opening_moves": opening_moves,
            }
        move = parts[1]
        states = [engine.ask("move " + move) for engine in engines.values()]
        if states[0] != states[1]:
            raise RuntimeError("engine states diverged after move")
        moves.append(move)
        verdicts = [_result(engine) for engine in engines.values()]
        if verdicts[0] != verdicts[1]:
            raise RuntimeError("engine verdicts diverged")
        outcome, reason = verdicts[0]
        # Query after every move, including the final permitted ply, so a mate
        # or adjudication at the cap is recorded as decided rather than unfinished.
        if outcome != "ongoing":
            break
    if outcome == "ongoing":
        outcome, reason = "unfinished", "ply_cap"
    return {
        "opening": name, "candidate_side": candidate_side,
        "pair_id": pair_id,
        "outcome": outcome, "reason": reason,
        "plies": len(moves), "moves": moves, "opening_moves": opening_moves,
    }


def _atomic_report(output: Path, report: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output)


def run_arena(
    engine: Path,
    candidate: Path,
    opponents: Sequence[Opponent],
    output: Path,
    *,
    search_mode: str = "time",
    depth: int = 4,
    time_limit_ms: int | None = 1_000,
    max_plies: int = 300,
    required_score: float = DEFAULT_REQUIRED_SCORE,
    required_win_improvement: float = DEFAULT_REQUIRED_WIN_IMPROVEMENT,
    max_unfinished: int = DEFAULT_MAX_UNFINISHED,
    min_games: int = 100,
    min_independent_pairs: int = 50,
    openings_path: Path = DEFAULT_OPENINGS,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    """候选与每个对手在相同深度/时间、固定开局换先对弈并持久化报告。"""

    if search_mode not in ("depth", "time"):
        raise ValueError("search mode must be 'depth' or 'time'")
    if depth <= 0 or max_plies <= 0:
        raise ValueError("arena depth and max plies must be positive")
    if search_mode == "time" and (time_limit_ms is None or time_limit_ms <= 0):
        raise ValueError("time search mode requires a positive time limit")
    if min_games <= 0 or min_independent_pairs <= 0 or timeout_seconds <= 0:
        raise ValueError("arena sample and timeout limits must be positive")
    if min_games % 2:
        raise ValueError("minimum game count must be even for candidate-color pairs")
    if not 0.0 <= required_score <= 1.0 or max_unfinished < 0:
        raise ValueError("invalid arena promotion threshold")
    if not 0.0 <= required_win_improvement <= 1.0:
        raise ValueError("required win improvement must be between 0 and 1")
    if not opponents:
        raise ValueError("at least one arena opponent is required")
    if output.exists() or output.with_suffix(output.suffix + ".tmp").exists():
        raise FileExistsError(f"arena report already exists; refusing to overwrite: {output}")
    if not engine.is_file() or not candidate.is_file():
        raise FileNotFoundError("arena engine and candidate model must exist")
    initial_fen, openings = load_openings(openings_path)
    if any(len(opening["moves"]) > max_plies for opening in openings):
        raise ValueError("an opening is longer than --max-plies")
    if len({item.name for item in opponents}) != len(opponents):
        raise ValueError("opponent names must be unique")
    for item in opponents:
        if item.network is not None and not item.network.is_file():
            raise FileNotFoundError(item.network)

    pair_count = min_games // 2
    schedule = [
        (pair_id, openings[pair_id % len(openings)])
        for pair_id in range(pair_count)
    ]
    # 有现任冠军时只对冠军要求棋力提升；其他对手是诊断对照。
    # 没有冠军时保留对配置对手的基线要求，不能无条件晋升第一代。
    promotion_opponents = (["champion"] if any(item.name == "champion" for item in opponents)
                           else [item.name for item in opponents])
    config = {
        "promotion_policy": PROMOTION_POLICY if promotion_opponents == ["champion"] else BASELINE_PROMOTION_POLICY,
        "promotion_opponents": promotion_opponents,
        "search_mode": search_mode,
        "depth": depth,
        "time_limit_ms": time_limit_ms if search_mode == "time" else None,
        "time_depth_cap": depth if search_mode == "time" else None,
        "max_plies": max_plies,
        "required_score": required_score,
        "required_win_improvement": required_win_improvement,
        "max_unfinished": max_unfinished,
        "max_unfinished_fraction": PROMOTION_MAX_UNFINISHED_FRACTION,
        "min_complete_pair_fraction": PROMOTION_MIN_COMPLETE_PAIR_FRACTION,
        "min_games": min_games,
        "min_independent_pairs": min_independent_pairs,
        "promotion_floor": {
            "games": PROMOTION_MIN_GAMES,
            "independent_opening_pairs": PROMOTION_MIN_INDEPENDENT_PAIRS,
        },
        "uncertainty": {
            "confidence": 0.95,
            "diagnostic_interval": "paired-opening-cluster-hoeffding-bound",
            "used_for_promotion": False,
            "diagnostic_bootstrap_samples": 10_000,
            "diagnostic_bootstrap_seed": 0x5849414E,
        },
        "timeout_seconds": timeout_seconds,
        "opening_file": str(openings_path.resolve()),
        "opening_file_sha256": sha256_file(openings_path),
        "initial_fen": initial_fen,
        "opening_names": [opening["name"] for opening in openings],
        "schedule_openings": [opening["name"] for _, opening in schedule],
        "candidate_sha256": sha256_file(candidate),
        "engine_sha256": sha256_file(engine),
        "opponents": [
            {
                "name": item.name,
                "network": str(item.network.resolve()) if item.network else None,
                "network_sha256": sha256_file(item.network) if item.network else None,
            }
            for item in opponents
        ],
    }
    report: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "config": config,
        "opponents": {},
        "passed": False,
    }
    try:
        for opponent in opponents:
            games: list[dict[str, Any]] = []
            engines: dict[str, ProtocolEngine] = {}
            failure_seen = False
            try:
                engines["candidate"] = ProtocolEngine(engine, candidate, timeout_seconds)
                engines["opponent"] = ProtocolEngine(
                    engine, opponent.network, timeout_seconds
                )
                for pair_id, opening in schedule:
                    for side in ("red", "black"):
                        if failure_seen:
                            games.append({
                                "opening": opening["name"],
                                "candidate_side": side,
                                "pair_id": pair_id,
                                "outcome": "unfinished",
                                "reason": "aborted_after_protocol_failure",
                                "plies": 0,
                                "moves": [],
                                "opening_moves": list(opening["moves"]),
                            })
                            continue
                        try:
                            game = _play_game(
                                engines, initial_fen, side, opening,
                                search_mode, depth,
                                time_limit_ms if search_mode == "time" else None,
                                max_plies, pair_id,
                            )
                        except (OSError, RuntimeError, TimeoutError, ValueError) as error:
                            failure_seen = True
                            game = {
                                "opening": opening["name"],
                                "candidate_side": side,
                                "pair_id": pair_id,
                                "outcome": "unfinished",
                                "reason": "protocol_failure",
                                "failure": f"{type(error).__name__}: {error}",
                                "plies": 0,
                                "moves": [],
                                "opening_moves": list(opening["moves"]),
                            }
                        games.append(game)
                        print(
                            f"{opponent.name} {opening['name']} candidate={side}: "
                            f"{game['outcome']} ({game['reason']})",
                            flush=True,
                        )
            except (OSError, RuntimeError, TimeoutError, ValueError) as error:
                failure_seen = True
                for pair_id, opening in schedule:
                    for side in ("red", "black"):
                        games.append({
                            "opening": opening["name"],
                            "candidate_side": side,
                            "pair_id": pair_id,
                            "outcome": "unfinished",
                            "reason": "protocol_failure" if not games else "not_started_after_engine_failure",
                            "failure": f"{type(error).__name__}: {error}" if not games else None,
                            "plies": 0,
                            "moves": [],
                            "opening_moves": list(opening["moves"]),
                        })
            finally:
                for protocol in engines.values():
                    protocol.close()
            summary = summarize_games(
                games, required_score, max_unfinished, min_games,
                min_independent_pairs,
                required_win_improvement=(required_win_improvement if opponent.name == "champion" else None),
            )
            report["opponents"][opponent.name] = {
                "summary": summary,
                "games": games,
                "transcripts": {
                    role: protocol.transcript for role, protocol in engines.items()
                },
            }
            _atomic_report(output, report)
            for side in ("red", "black"):
                own = summary["by_color"][side]
                other = summary["opponent_by_color"][side]
                print(
                    f"arena {opponent.name} {side}: candidate {format_color_statistics(own)}; "
                    f"opponent {format_color_statistics(other)}", flush=True,
                )
                improvement = summary["win_improvement_by_color"][side]
                if opponent.name == "champion":
                    display = f"{improvement * 100:+.2f} pp" if improvement is not None else "n/a"
                    print(f"arena champion {side}: win improvement={display}; "
                          f"required={required_win_improvement * 100:.2f} pp", flush=True)
            print(f"arena {opponent.name}: {summary}; report: {output}", flush=True)
        report["passed"] = all(
            report["opponents"][name]["summary"]["passed"] for name in promotion_opponents
        ) and all(result["summary"]["failed_games"] == 0 for result in report["opponents"].values())
        report["rejection_reasons"] = [
            f"{name}: {reason}" for name, result in report["opponents"].items()
            for reason in result["summary"]["rejection_reasons"]
            if name in promotion_opponents or reason == "protocol_failure"
        ]
        _atomic_report(output, report)
        return report
    except BaseException:
        # Preserve already completed/failed game evidence before propagating.
        _atomic_report(output, report)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, default=Path("build/make/xiangqi_protocol"))
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--hand", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--champion", type=Path)
    parser.add_argument("--history", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--openings", type=Path, default=DEFAULT_OPENINGS)
    parser.add_argument(
        "--search-mode", choices=("depth", "time"), default="time",
        help="depth uses only the depth cap; time uses depth as a cap plus equal movetime",
    )
    parser.add_argument(
        "--depth", type=int, default=None,
        help="search depth/cap (default: 64 in time mode, 4 in depth mode)",
    )
    parser.add_argument("--time-limit-ms", type=int, default=1_000)
    parser.add_argument("--protocol-timeout", type=float, default=10.0)
    parser.add_argument("--max-plies", type=int, default=300)
    parser.add_argument("--required-score", type=float, default=DEFAULT_REQUIRED_SCORE,
                        help="baseline paired score without a champion (default: 0.55)")
    parser.add_argument("--required-win-improvement", type=float, default=DEFAULT_REQUIRED_WIN_IMPROVEMENT,
                        help="minimum same-color win-rate difference for BOTH colors vs champion (0.10 = 10 percentage points)")
    parser.add_argument("--max-unfinished", type=int, default=DEFAULT_MAX_UNFINISHED,
                        help="ply-cap allowance (default: 10), also capped at 10%%; protocol failures remain fatal")
    parser.add_argument("--min-games", type=int, default=100)
    parser.add_argument("--min-independent-pairs", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    opponents: list[Opponent] = []
    if args.hand:
        opponents.append(Opponent("hand", None))
    if args.champion:
        opponents.append(Opponent("champion", args.champion))
    for index, network in enumerate(args.history, start=1):
        opponents.append(Opponent(f"history-{index}", network))
    run_arena(
        args.engine,
        args.candidate,
        opponents,
        args.output,
        search_mode=args.search_mode,
        depth=resolve_search_depth(args.search_mode, args.depth),
        time_limit_ms=args.time_limit_ms,
        max_plies=args.max_plies,
        required_score=args.required_score,
        required_win_improvement=args.required_win_improvement,
        max_unfinished=args.max_unfinished,
        min_games=args.min_games,
        min_independent_pairs=args.min_independent_pairs,
        openings_path=args.openings,
        timeout_seconds=args.protocol_timeout,
    )


if __name__ == "__main__":
    main()
