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
REPORT_FORMAT = "xiangqi-arena-v2"
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


def summarize_games(
    games: list[dict[str, Any]], required_score: float, max_unfinished: int,
    min_games: int, min_independent_pairs: int = 50,
) -> dict[str, Any]:
    wins = losses = draws = unfinished = 0
    for game in games:
        outcome = game["outcome"]
        if outcome == "unfinished":
            unfinished += 1
        elif outcome == "draw":
            draws += 1
        elif outcome == f"{game['candidate_side']}_win":
            wins += 1
        elif outcome in ("red_win", "black_win"):
            losses += 1
        else:
            raise ValueError(f"unknown game outcome: {outcome}")
    decided = wins + losses + draws
    score_fraction = (wins + 0.5 * draws) / decided if decided else 0.0
    interval = paired_opening_interval(games)
    complete_pairs = int(interval["independent_openings"])
    lower = interval["lower"]
    effective_min_games = max(min_games, PROMOTION_MIN_GAMES)
    effective_min_independent_pairs = max(
        min_independent_pairs, PROMOTION_MIN_INDEPENDENT_PAIRS
    )
    passed = (
        len(games) >= effective_min_games
        and unfinished <= max_unfinished
        and decided > 0
        and complete_pairs >= effective_min_independent_pairs
        and score_fraction >= required_score
        and lower is not None
        and lower >= required_score
    )
    return {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "unfinished": unfinished,
        "games": len(games),
        "decided_games": decided,
        "complete_paired_openings": complete_pairs,
        "score_fraction": score_fraction,
        "score_fraction_ci95": interval,
        "required_score": required_score,
        "min_games": min_games,
        "min_independent_pairs": min_independent_pairs,
        "effective_min_games": effective_min_games,
        "effective_min_independent_pairs": effective_min_independent_pairs,
        "max_unfinished": max_unfinished,
        "passed": passed,
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
    required_score: float = 0.70,
    max_unfinished: int = 0,
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
    config = {
        "search_mode": search_mode,
        "depth": depth,
        "time_limit_ms": time_limit_ms if search_mode == "time" else None,
        "time_depth_cap": depth if search_mode == "time" else None,
        "max_plies": max_plies,
        "required_score": required_score,
        "max_unfinished": max_unfinished,
        "min_games": min_games,
        "min_independent_pairs": min_independent_pairs,
        "promotion_floor": {
            "games": PROMOTION_MIN_GAMES,
            "independent_opening_pairs": PROMOTION_MIN_INDEPENDENT_PAIRS,
        },
        "uncertainty": {
            "confidence": 0.95,
            "gate_interval": "paired-opening-cluster-hoeffding-bound",
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
            )
            report["opponents"][opponent.name] = {
                "summary": summary,
                "games": games,
                "transcripts": {
                    role: protocol.transcript for role, protocol in engines.items()
                },
            }
            _atomic_report(output, report)
            print(f"arena {opponent.name}: {summary}; report: {output}", flush=True)
        report["passed"] = all(
            result["summary"]["passed"]
            for result in report["opponents"].values()
        )
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
    parser.add_argument("--required-score", type=float, default=0.70)
    parser.add_argument("--max-unfinished", type=int, default=0)
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
        max_unfinished=args.max_unfinished,
        min_games=args.min_games,
        min_independent_pairs=args.min_independent_pairs,
        openings_path=args.openings,
        timeout_seconds=args.protocol_timeout,
    )


if __name__ == "__main__":
    main()
