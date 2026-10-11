"""检查并抽样 nowcan 的 SQLite 棋谱；只读取原库，不执行其中的文本。"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sqlite3
import statistics
import time
from typing import Any

from gui.protocol_client import ProtocolEngineClient, ProtocolError
from .dataset import JsonlPositionDataset, phase_for_fen
from .diagnostics import write_json
from .halfka import encode_fen
from .partition import canonical_position_key, file_fingerprint, load_test_manifest, source_id, write_test_manifest
from .run_lock import run_lock
from .uci_teacher import UciTeacher

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "training/data/modelscope-nowcan"
SOURCES = (
    ("depth11-mg", "XQDB.db", "v.db", 0.5),
    ("depth11-eg", "XQDB.db", "XQDB_v.db", 0.25),
    ("depth16-eg", "XQDB.db", "XQDB.db", 0.25),
)


def source_paths(directory: Path, split: str) -> list[tuple[str, Path, float]]:
    """列出中盘/两类残局原库；split 只能为 train 或 val。"""

    return [(name, directory / "unpacked" / split / name / (train if split == "train" else val), ratio)
            for name, train, val, ratio in SOURCES]


def open_database(path: Path) -> tuple[sqlite3.Connection, int]:
    """只读打开固定的 XQPGN(PGN) 表，并通过索引读取最大 rowid。"""

    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")
    columns = connection.execute("PRAGMA table_info(XQPGN)").fetchall()
    if not columns or "PGN" not in {column[1] for column in columns}:
        connection.close()
        raise ValueError(f"unsupported SQLite schema: {path}")
    maximum = connection.execute("SELECT MAX(rowid) FROM XQPGN").fetchone()[0]
    if maximum is None:
        connection.close()
        raise ValueError(f"empty database: {path}")
    return connection, int(maximum)


def decode_move(value: int) -> str:
    """将低字节起点、高字节终点的 16×16 邮箱坐标转换为 ICCS。"""

    if type(value) is not int or not 0 <= value <= 65535:
        raise ValueError("move must be an unsigned 16-bit integer")
    squares = []
    for square in (value & 255, value >> 8):
        row, column = divmod(square, 16)
        if not (3 <= row <= 12 and 3 <= column <= 11):
            raise ValueError(f"mailbox square outside board: {square}")
        squares.append(chr(ord("a") + column - 3) + str(12 - row))
    return "".join(squares)


def mirror_fen(fen: str) -> str:
    """左右翻转棋盘，保留行棋方；用于整盘棋的镜像身份。"""

    board, side = fen.split()[:2]
    ranks = []
    for rank in board.split("/"):
        expanded = "".join("." * int(char) if char.isdigit() else char for char in rank)
        ranks.append(expanded[::-1])
    return "/".join(ranks) + " " + side


def replay_game(client: ProtocolEngineClient, text: str, max_score: float) -> tuple[str, list[dict[str, Any]], Counter[str]]:
    """完整合法重放后返回可用落子后局面；任一步非法则拒绝整局。

    参数:
        client: 不执行搜索的本地规则引擎。
        text: SQLite 内的 JSON 棋局字符串，不是可执行代码。
        max_score: 保留的普通评分绝对值上限，排除将杀及极端标签。
    """

    game = json.loads(text)
    if not isinstance(game, dict) or not isinstance(game.get("start_fen"), str):
        raise ValueError("game requires start_fen")
    moves = game.get("mvs")
    skip = game.get("skip_step", 0)
    result = game.get("result", 2)
    if not isinstance(moves, list) or not moves or len(moves) > 2000:
        raise ValueError("invalid move list")
    if type(skip) is not int or not 0 <= skip <= len(moves):
        raise ValueError("invalid skip_step")
    if type(result) is not int or result not in (-1, 0, 1, 2):
        raise ValueError("invalid result")
    initial = client.set_position(game["start_fen"]).fen
    iccs = [decode_move(item["mv"]) for item in moves]
    mirror_moves = [chr(ord("i") - (ord(move[0]) - ord("a"))) + move[1]
                    + chr(ord("i") - (ord(move[2]) - ord("a"))) + move[3] for move in iccs]
    # Expand both original/mirror identically; canonical ID does not depend on DB path/rowid.
    original_board = mirror_fen(mirror_fen(initial))
    identity = min(original_board + "|" + "".join(iccs), mirror_fen(initial) + "|" + "".join(mirror_moves))
    game_id = "nowcan-" + hashlib.sha256(identity.encode()).hexdigest()
    previous = initial
    positions: list[dict[str, Any]] = []
    filters: Counter[str] = Counter()
    for ply, (item, move) in enumerate(zip(moves, iccs), 1):
        fen = client.play(move).fen
        score = item.get("score")
        if type(score) not in (int, float) or not math.isfinite(score):
            raise ValueError(f"invalid score at ply {ply}")
        if ply <= skip:
            filters["random_prefix"] += 1
        elif abs(score) > max_score:
            filters["mate_or_extreme"] += 1
        else:
            encode_fen(fen)
            positions.append({"fen": fen, "before_fen": previous, "raw_score": score,
                              "ply": ply, "move": move, "site_result": result, "skip_step": skip})
        previous = fen
    return game_id, positions, filters


def score_for_position(position: dict[str, Any], convention: str) -> float:
    """把原标签换成落子后当前行棋方视角；约定必须先经过审计。"""

    raw = float(position["raw_score"])
    if convention == "post-mover":
        return -raw
    if convention == "post-stm":
        return raw
    if convention == "post-red":
        return raw if position["fen"].split()[1] == "w" else -raw
    raise ValueError(f"unsupported score convention: {convention}")


def audit(args: argparse.Namespace) -> None:
    """随机抽局，比较三种评分视角和落子前/后的教师评分，保存证据。"""

    rng = random.Random(args.seed)
    records: list[dict[str, Any]] = []
    errors: Counter[str] = Counter()
    splits = ("train", "val") if args.source_split == "all" else (args.source_split,)
    paths = [(split + "/" + name, path, ratio) for split in splits
             for name, path, ratio in source_paths(args.directory, split)]
    with ProtocolEngineClient(args.rules_engine) as client, UciTeacher(
        args.teacher, args.teacher_network, timeout=60.0,
    ) as teacher:
        identity = teacher.identity
        for name, path, _ in paths:
            if not path.exists():
                continue
            connection, maximum = open_database(path)
            accepted = 0
            try:
                for _ in range(args.samples * 100):
                    rowid = rng.randint(1, maximum)
                    row = connection.execute("SELECT rowid,PGN FROM XQPGN WHERE rowid=?", (rowid,)).fetchone()
                    if row is None:
                        continue
                    try:
                        _, positions, _ = replay_game(client, row[1], args.max_score)
                    except (ValueError, ProtocolError, KeyError, TypeError) as error:
                        errors[type(error).__name__] += 1
                        continue
                    pool = [p for p in positions if 100 <= abs(p["raw_score"]) <= 1500]
                    rng.shuffle(pool)
                    for position in pool[:2]:
                        analysis = teacher.analyze(position["fen"], nodes=args.nodes)
                        if analysis is None or analysis.score_kind != "cp" or analysis.bound is not None:
                            continue
                        before = teacher.analyze(position["before_fen"], nodes=args.nodes)
                        records.append({**position, "source": name, "rowid": rowid,
                                        "teacher_post_stm": analysis.score,
                                        "teacher_before_stm": before.score if before and before.score_kind == "cp"
                                        and before.bound is None else None})
                        accepted += 1
                        if accepted >= args.samples:
                            break
                    if accepted >= args.samples:
                        break
                print(f"audit {name}: {accepted} positions", flush=True)
            finally:
                connection.close()
    if not records:
        raise ValueError("no audit samples")
    comparison = {}
    for convention in ("post-mover", "post-stm", "post-red"):
        differences = [abs(score_for_position(p, convention) - p["teacher_post_stm"]) for p in records]
        decisive = [p for p in records if abs(p["teacher_post_stm"]) >= 100]
        comparison[convention] = {
            "mae": statistics.mean(differences), "median_absolute_error": statistics.median(differences),
            "sign_agreement": sum(score_for_position(p, convention) * p["teacher_post_stm"] > 0
                                  for p in decisive) / len(decisive) if decisive else None,
            "decisive_count": len(decisive),
        }
    before_errors = [abs(p["raw_score"] - p["teacher_before_stm"]) for p in records
                     if p["teacher_before_stm"] is not None]
    per_source = {}
    for source in sorted({p["source"] for p in records}):
        pool = [p for p in records if p["source"] == source]
        decisive = [p for p in pool if abs(p["teacher_post_stm"]) >= 100]
        per_source[source] = {"samples": len(pool), "decisive_count": len(decisive),
                              "post_mover_mae": statistics.mean(abs(-p["raw_score"] - p["teacher_post_stm"]) for p in pool),
                              "post_mover_sign_agreement": sum(-p["raw_score"] * p["teacher_post_stm"] > 0 for p in decisive)
                              / len(decisive) if decisive else None}
    report = {"format": "nowcan-score-audit-v1", "seed": args.seed, "nodes": args.nodes,
              "teacher_identity": identity, "teacher_sha256": file_fingerprint(args.teacher),
              "network_sha256": file_fingerprint(args.teacher_network), "comparison": comparison,
              "before_mover_mae": statistics.mean(before_errors) if before_errors else None,
              "per_source": per_source, "rejected_games": dict(errors), "samples": records}
    write_json(args.output, report)
    print(json.dumps({"comparison": comparison, "before_mover_mae": report["before_mover_mae"]}, indent=2), flush=True)


def export_split(connection: sqlite3.Connection, split: str, output: Path) -> int:
    """按稳定顺序原子导出指定分区，避免中断留下可误用的半文件。"""

    temporary = output.with_suffix(output.suffix + ".partial")
    count = 0
    with temporary.open("w", encoding="utf-8") as stream:
        for (payload,) in connection.execute("SELECT payload FROM positions WHERE split=? ORDER BY source,rowid,ply", (split,)):
            stream.write(payload + "\n")
            count += 1
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output)
    return count


def prepare(args: argparse.Namespace) -> None:
    """按文件配额和随机整局抽样；先验证后训练，全局排除重复与镜像。

    SQLite 每盘提交并保存随机状态；相同配置重跑可以断点继续。
    只抽样少量局面，不全量载入几十 GB 的数据库。
    """

    directory, output = args.directory.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    all_paths = [(split, *item) for split in ("val", "train") for item in source_paths(directory, split)]
    for _, _, path, _ in all_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    audit_report = json.loads(args.audit.read_text())
    result = audit_report["comparison"][args.convention]
    if result["decisive_count"] < 20 or result["sign_agreement"] is None or result["sign_agreement"] < 0.9:
        raise ValueError("score audit does not support the selected convention")
    if args.convention == "post-mover":
        expected_sources = {split + "/" + name for split, name, _, _ in all_paths}
        if expected_sources - set(audit_report.get("per_source", {})):
            raise ValueError("audit must cover all six databases before sampling")
        for source, metrics in audit_report["per_source"].items():
            if metrics["decisive_count"] < 10 or metrics["post_mover_sign_agreement"] is None or metrics["post_mover_sign_agreement"] < 0.9:
                raise ValueError(f"ambiguous score convention in {source}; do not train yet")
    protected = load_test_manifest(args.test_manifest)
    plan = {"format": "nowcan-sampling-plan-v1", "train_count": args.train_count,
            "validation_count": args.validation_count, "seed": args.seed, "per_game": args.per_game,
            "max_score": args.max_score, "convention": args.convention, "score_multiplier": args.score_multiplier,
            "rules_sha256": file_fingerprint(args.rules_engine), "audit_sha256": file_fingerprint(args.audit),
            "protected_test_sha256": file_fingerprint(args.test_manifest),
            "inputs": {str(path): {"size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
                       for _, _, path, _ in all_paths}}
    # Optional fields keep existing 100k plans byte-for-byte compatible.
    if args.fixed_validation:
        plan["fixed_validation"] = {"path": str(args.fixed_validation.resolve()),
                                    "sha256": file_fingerprint(args.fixed_validation)}
    if args.test_count:
        plan["test_count"] = args.test_count
    plan_path = output / "sampling-plan.json"
    state_path = output / "sampling.sqlite3"
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != plan or not state_path.exists():
            raise ValueError("resume input/config changed or database missing; use a new output directory")
        report_path = output / "sampling-report.json"
        if report_path.exists():
            completed = json.loads(report_path.read_text())
            if completed.get("status") == "complete":
                for filename, field in (("train.jsonl", "train_sha256"), ("validation.jsonl", "validation_sha256")):
                    if file_fingerprint(output / filename) != completed[field]:
                        raise ValueError(f"completed sample file changed: {filename}")
                if args.test_count:
                    if file_fingerprint(output / "test.jsonl") != completed["test_sha256"]:
                        raise ValueError("completed test samples changed")
                    load_test_manifest(output / "test-manifest.json")
                print(f"already complete: train={completed['train_count']} validation={completed['validation_count']}", flush=True)
                return
    else:
        if state_path.exists() or (output / "train.jsonl").exists():
            raise FileExistsError("sampling output exists without its plan")
        write_json(plan_path, plan)
    state = sqlite3.connect(state_path)
    state.execute("PRAGMA journal_mode=WAL")
    state.execute("PRAGMA synchronous=FULL")
    state.executescript("""
        CREATE TABLE IF NOT EXISTS positions (key TEXT PRIMARY KEY,split TEXT,source TEXT,rowid INTEGER,game_id TEXT,ply INTEGER,payload TEXT);
        CREATE TABLE IF NOT EXISTS games (source TEXT,rowid INTEGER,game_id TEXT,status TEXT,PRIMARY KEY(source,rowid));
        CREATE TABLE IF NOT EXISTS identities (game_id TEXT PRIMARY KEY,split TEXT);
        CREATE TABLE IF NOT EXISTS rng (source TEXT PRIMARY KEY,state TEXT);
        CREATE TABLE IF NOT EXISTS blocked (key TEXT PRIMARY KEY);
    """)
    state.executemany("INSERT OR IGNORE INTO blocked VALUES (?)", ((key,) for key in protected["position_keys"]))
    state.commit()
    if args.fixed_validation:
        # Freeze the original validation games before selecting test/train games.
        # One transaction makes interrupted imports safe to repeat.
        existing = state.execute("SELECT COUNT(*) FROM positions WHERE split='val'").fetchone()[0]
        protected_keys = set(protected["position_keys"])
        if not existing:
            with args.fixed_validation.open(encoding="utf-8") as stream, state:
                for line in stream:
                    record = json.loads(line)
                    encode_fen(record["fen"])
                    key = canonical_position_key(record["fen"])
                    if key in protected_keys:
                        raise ValueError("fixed validation overlaps protected test")
                    if not math.isfinite(float(record["score"])):
                        raise ValueError("invalid fixed validation score")
                    origin = record["provenance"]
                    identifier = "val/" + origin["database"].split("/")[-1]
                    state.execute("INSERT INTO positions VALUES (?,?,?,?,?,?,?)",
                                  (key, "val", identifier, origin["rowid"], record["game_id"], record["ply"],
                                   json.dumps(record, ensure_ascii=False, allow_nan=False)))
                    state.execute("INSERT OR IGNORE INTO identities VALUES (?,?)", (record["game_id"], "val"))
            existing = state.execute("SELECT COUNT(*) FROM positions WHERE split='val'").fetchone()[0]
        if existing != args.validation_count:
            raise ValueError("fixed validation count does not match --validation-count")
    sample_paths = [item for item in all_paths if not (args.fixed_validation and item[0] == "val")]
    if args.test_count:
        sample_paths = [("test", *item) for item in source_paths(directory, "val")] + sample_paths
    started = time.monotonic()
    filters: Counter[str] = Counter()
    try:
        with ProtocolEngineClient(args.rules_engine) as client:
            for split, name, path, ratio in sample_paths:
                total = {"train": args.train_count, "val": args.validation_count, "test": args.test_count}[split]
                # Exact allocation also works when a requested count is not divisible by four.
                quotas = [total // 2, total // 4, total - total // 2 - total // 4]
                quota = quotas[[item[0] for item in SOURCES].index(name)]
                identifier = split + "/" + name
                count = state.execute("SELECT COUNT(*) FROM positions WHERE source=?", (identifier,)).fetchone()[0]
                rng = random.Random(f"{args.seed}:{identifier}")
                saved = state.execute("SELECT state FROM rng WHERE source=?", (identifier,)).fetchone()
                if saved:
                    version, internal, gaussian = json.loads(saved[0])
                    rng.setstate((version, tuple(internal), gaussian))
                connection, maximum = open_database(path)
                attempts = 0
                print(f"sampling {identifier}: {count}/{quota}; max_rowid={maximum}", flush=True)
                try:
                    while count < quota:
                        attempts += 1
                        if attempts > max(10000, quota * 30):
                            raise ValueError(f"too few eligible positions in {identifier}")
                        rowid = rng.randint(1, maximum)
                        if state.execute("SELECT 1 FROM games WHERE source=? AND rowid=?", (identifier, rowid)).fetchone():
                            continue
                        row = connection.execute("SELECT PGN FROM XQPGN WHERE rowid=?", (rowid,)).fetchone()
                        if row is None:
                            continue
                        game_id = None
                        status = "accepted"
                        try:
                            game_id, positions, rejected = replay_game(client, row[0], args.max_score)
                            filters.update(rejected)
                            if state.execute("SELECT 1 FROM identities WHERE game_id=?", (game_id,)).fetchone():
                                status = "duplicate_game"
                                positions = []
                            else:
                                state.execute("INSERT INTO identities VALUES (?,?)", (game_id, split))
                            rng.shuffle(positions)
                            added = 0
                            for position in positions:
                                if added >= args.per_game or count >= quota:
                                    break
                                key = canonical_position_key(position["fen"])
                                if state.execute("SELECT 1 FROM blocked WHERE key=?", (key,)).fetchone():
                                    filters["protected_test"] += 1
                                    continue
                                if state.execute("SELECT 1 FROM positions WHERE key=?", (key,)).fetchone():
                                    filters["duplicate_or_mirror"] += 1
                                    continue
                                score = score_for_position(position, args.convention) * args.score_multiplier
                                side = position["fen"].split()[1]
                                result = position["site_result"]
                                record = {"fen": position["fen"], "score": score, "game_id": game_id,
                                          "ply": position["ply"], "phase": phase_for_fen(position["fen"]),
                                          "result": None if result == 2 else result * (1 if side == "w" else -1),
                                          "source": "modelscope/nowcan/xiangqi_train_data",
                                          "provenance": {"database": identifier, "rowid": rowid,
                                              "move": position["move"], "raw_score": position["raw_score"],
                                              "skip_step": position["skip_step"], "raw_result": result,
                                              "score_convention": args.convention, "score_multiplier": args.score_multiplier}}
                                state.execute("INSERT INTO positions VALUES (?,?,?,?,?,?,?)",
                                              (key, split, identifier, rowid, game_id, position["ply"],
                                               json.dumps(record, ensure_ascii=False, allow_nan=False)))
                                count += 1
                                added += 1
                        except (ValueError, ProtocolError, KeyError, TypeError) as error:
                            status = f"rejected:{type(error).__name__}:{str(error)[:200]}"
                        state.execute("INSERT INTO games VALUES (?,?,?,?)", (identifier, rowid, game_id, status))
                        state.execute("INSERT OR REPLACE INTO rng VALUES (?,?)", (identifier, json.dumps(rng.getstate())))
                        state.commit()
                        if attempts % 250 == 0 or count >= quota:
                            print(f"{identifier}: {count}/{quota} elapsed={time.monotonic()-started:.1f}s", flush=True)
                finally:
                    connection.close()
        train = export_split(state, "train", output / "train.jsonl")
        validation = export_split(state, "val", output / "validation.jsonl")
        test = export_split(state, "test", output / "test.jsonl") if args.test_count else 0
        if args.fixed_validation and file_fingerprint(output / "validation.jsonl") != file_fingerprint(args.fixed_validation):
            raise ValueError("fixed validation export changed order or content")
        if args.test_count:
            test_path = output / "test.jsonl"
            samples = JsonlPositionDataset(test_path).samples
            manifest_payload = {
                "format": "xiangqi-fixed-test-v1", "seed": args.seed,
                "fraction": test / (train + validation + test),
                "groups": sorted({sample.group_key for sample in samples}),
                "position_keys": sorted({sample.position_key for sample in samples}),
                "source_fingerprints": {source_id(test_path): file_fingerprint(test_path)},
            }
            manifest_path = output / "test-manifest.json"
            if manifest_path.exists():
                manifest = load_test_manifest(manifest_path)
                if any(manifest[key] != value for key, value in manifest_payload.items()):
                    raise ValueError("fixed test manifest changed")
            else:
                write_test_manifest(manifest_path, manifest_payload)
        phases: Counter[str] = Counter()
        scores: dict[str, list[float]] = {"train": [], "val": [], "test": []} if args.test_count else {"train": [], "val": []}
        for split, payload in state.execute("SELECT split,payload FROM positions"):
            record = json.loads(payload)
            phases[split + "/" + record["phase"]] += 1
            scores[split].append(record["score"])
        report = {"format": "nowcan-sampling-report-v1", "status": "complete", "plan": plan,
                  "train_count": train, "validation_count": validation,
                  "source_counts": dict(state.execute("SELECT source,COUNT(*) FROM positions GROUP BY source")),
                  "game_counts": dict(state.execute("SELECT split,COUNT(DISTINCT game_id) FROM positions GROUP BY split")),
                  "game_status_counts": dict(state.execute("SELECT status,COUNT(*) FROM games GROUP BY status")),
                  "phases": dict(phases), "filters_this_invocation": dict(filters),
                  "score_ranges": {split: {"min": min(values), "max": max(values), "median": statistics.median(values)}
                                   for split, values in scores.items()},
                  "elapsed_seconds": time.monotonic() - started,
                  "train_sha256": file_fingerprint(output / "train.jsonl"),
                  "validation_sha256": file_fingerprint(output / "validation.jsonl")}
        if args.test_count:
            report.update(test_count=test, test_sha256=file_fingerprint(output / "test.jsonl"),
                          test_manifest_sha256=file_fingerprint(output / "test-manifest.json"))
        write_json(output / "sampling-report.json", report)
        print(f"complete train={train} validation={validation}", flush=True)
    finally:
        state.close()


def parse_args() -> argparse.Namespace:
    """解析审计或抽样选项，数据量均以局面而不是棋局计数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inventory = sub.add_parser("inventory")
    inventory.add_argument("--directory", type=Path, default=DATA_ROOT)
    for name in ("audit", "prepare"):
        command = sub.add_parser(name)
        command.add_argument("--directory", type=Path, default=DATA_ROOT)
        command.add_argument("--rules-engine", type=Path, default=ROOT / "build/make/xiangqi_protocol")
        command.add_argument("--seed", type=int, default=2026)
        command.add_argument("--max-score", type=float, default=3000.0)
        command.add_argument("--output", required=True, type=Path)
        if name == "audit":
            command.add_argument("--source-split", choices=("train", "val", "all"), default="val")
            command.add_argument("--samples", type=int, default=32, help="positions per available source")
            command.add_argument("--nodes", type=int, default=20000)
            command.add_argument("--teacher", type=Path, default=ROOT / "training/tools/pikafish-2026-09-06/Pikafish-MacOS-universal")
            command.add_argument("--teacher-network", type=Path, default=ROOT / "training/tools/pikafish-2026-09-06/pikafish.nnue")
        else:
            command.add_argument("--audit", required=True, type=Path)
            command.add_argument("--convention", required=True, choices=("post-mover", "post-stm", "post-red"))
            command.add_argument("--score-multiplier", type=float, default=1.0)
            command.add_argument("--train-count", type=int, default=100000)
            command.add_argument("--validation-count", type=int, default=10000)
            command.add_argument("--fixed-validation", type=Path,
                                 help="reuse an existing validation JSONL unchanged; reserve all its games")
            command.add_argument("--test-count", type=int, default=0,
                                 help="additional final-test positions from original val databases")
            command.add_argument("--per-game", type=int, default=8)
            command.add_argument("--test-manifest", type=Path, default=ROOT / "training/runs/pikafish-full-20261011/teacher-test-manifest.json")
    args = parser.parse_args()
    if args.command != "inventory" and (args.max_score <= 0 or not math.isfinite(args.max_score)):
        parser.error("max-score must be finite and positive")
    if args.command == "prepare" and (min(args.train_count, args.validation_count, args.per_game) <= 0
                                     or args.test_count < 0 or not math.isfinite(args.score_multiplier) or args.score_multiplier <= 0):
        parser.error("counts, per-game and multiplier must be positive")
    return args


def inventory(args: argparse.Namespace) -> None:
    """核对六个原包的发布者 SHA-256，并记录本地大小和目录，不修改原包。"""

    expected = {
        "train_dat_pkf_depth_11_12m_mg_pgn_20231023.7z": "d5028fe984004a7ff65b54c4c1df5c6ed31f3388a64a9448bba1da44243d880d",
        "train_dat_pkf_depth_11_7m_eg_pgn_20240119.7z": "5a269aaddc466c6188a696726228cf27e38076324a76a6072be6e18aea3b4d5a",
        "train_dat_pkf_depth_16_4m_eg_pgn_20240319.7z": "f268454cb0d97973de233c82dd238341e1e5fd968819b16aaf9dded25b195dee",
        "val_dat_pkf_depth_11_eg_pgn_20240119.7z": "d7cf96f21c2ab08434fdb51a8361aaf875abc544cce0b684ab86c17fca92e707",
        "val_dat_pkf_depth_11_mg_pgn_20231010.7z": "1ba7793f421fb064bd2b4306dd0340502699d0457d1a8d389364d2394af55d5f",
        "val_dat_pkf_depth_16_eg_pgn_20240304.7z": "afaa13473a9c5f0b1212547521c582c4aea71f53c3004e72bebb4c764284e308",
    }
    files = []
    for name, digest in expected.items():
        path = args.directory / "raw" / ("train" if name.startswith("train_") else "val") / name
        actual = file_fingerprint(path)
        if actual != digest:
            raise ValueError(f"archive checksum mismatch: {path}")
        files.append({"file": str(path.resolve()), "bytes": path.stat().st_size, "sha256": actual})
        print(f"verified {name}", flush=True)
    write_json(args.directory / "inventory.json", {
        "source": "https://modelscope.cn/datasets/nowcan/xiangqi_train_data",
        "license_declared_by_publisher": "Apache License 2.0",
        "archives": files,
        "total_compressed_bytes": sum(item["bytes"] for item in files),
        "storage": "raw/train and raw/val retain the originals; unpacked stores SQLite; processed stores sampled JSONL",
    })


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.command == "inventory":
        inventory(arguments)
    elif arguments.command == "audit":
        audit(arguments)
    else:
        with run_lock(arguments.output.resolve() / "sampling.lock"):
            prepare(arguments)
