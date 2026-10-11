"""用 Pikafish 重标 JSONL，隔离固定测试局，SQLite 逐条续标，不覆盖原数据。"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sqlite3
import time
from typing import Any

from .dataset import phase_for_fen
from .diagnostics import write_json
from .halfka import encode_fen
from .partition import (
    DEFAULT_TEST_MANIFEST, canonical_position_key, file_fingerprint,
    game_group_key, load_test_manifest, source_id,
)
from .uci_teacher import UciTeacher
from .run_lock import run_lock


def select_positions(paths: list[Path], manifest: dict[str, Any], partition: str,
                     limit: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """按原对局及镜像隔离，再按阶段均衡抽样；limit=0 表示全部。"""

    groups, protected = set(manifest["groups"]), set(manifest["position_keys"])
    seen: set[str] = set()
    pools: dict[str, list[dict[str, Any]]] = {name: [] for name in ("opening", "middlegame", "endgame")}
    counts: Counter[str] = Counter()
    for path in sorted(paths):
        with path.open(encoding="utf-8") as stream:
            records = [(number, json.loads(line)) for number, line in enumerate(stream, 1)
                       if line.strip() and not line.lstrip().startswith("#")]
        # 与原 dataset 的保守规则一致：编号缺失和有编号混杂时，整文件成组。
        original_groups = {game_group_key(path, record) for _, record in records}
        mixed = any(group.endswith("#ungrouped") for group in original_groups) and any(
            "#game=" in group for group in original_groups)
        for number, record in records:
            fen = str(record["fen"])
            encode_fen(fen)
            if not math.isfinite(float(record["score"])):
                raise ValueError(f"non-finite original score at {path}:{number}")
            group = f"{source_id(path)}#file" if mixed else game_group_key(path, record)
            key = canonical_position_key(fen)
            counts["input"] += 1
            in_test = group in groups
            if partition == "development" and (in_test or key in protected):
                counts["protected_test"] += 1
                continue
            if partition == "fixed-test" and (not in_test or key not in protected):
                counts["not_fixed_test"] += 1
                continue
            if key in seen:
                counts["duplicate_or_mirror"] += 1
                continue
            seen.add(key)
            phase = phase_for_fen(fen)
            # 原批次+对局作为新 game_id，同一原对局的所有样本仍成组。
            pools[phase].append({
                "fen": fen, "phase": phase, "game_id": group,
                "original": {"source": source_id(path), "line": number,
                             "game_group": group, "score": record["score"],
                             "result": record.get("result"), "ply": record.get("ply")},
            })
    rng = random.Random(seed)
    for pool in pools.values():
        rng.shuffle(pool)
    selected: list[dict[str, Any]] = []
    maximum = limit or sum(map(len, pools.values()))
    while len(selected) < maximum and any(pools.values()):
        for pool in pools.values():
            if pool and len(selected) < maximum:
                selected.append(pool.pop())
    counts["selected"] = len(selected)
    return selected, dict(counts)


def export_records(connection: sqlite3.Connection, output: Path) -> int:
    """从已提交数据库原子导出有效样本；中断恢复以数据库而非半写 JSONL 为准。"""

    temporary = output.with_suffix(output.suffix + ".partial")
    count = 0
    with temporary.open("w", encoding="utf-8") as stream:
        for (payload,) in connection.execute("SELECT payload FROM records WHERE status='accepted' ORDER BY ordinal"):
            stream.write(payload + "\n")
            count += 1
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output)
    return count


def label(args: argparse.Namespace) -> dict[str, Any]:
    """执行一次固定配置的标注；同命令重跑只处理未提交样本。"""

    paths = [path.resolve() for path in args.datasets]
    output = args.output.resolve()
    if output in paths:
        raise ValueError("never overwrite an input dataset")
    if args.limit < 0 or args.nodes <= 0 or (args.depth is not None and args.depth <= 0):
        raise ValueError("limit must be nonnegative; search limits must be positive")
    if args.threads <= 0 or args.hash_mb <= 0 or not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError("threads, hash and timeout must be positive")
    if not math.isfinite(args.cp_multiplier) or args.cp_multiplier <= 0 or not 0 < args.max_score < 28000:
        raise ValueError("invalid score conversion or maximum")
    if len(set(paths)) != len(paths):
        raise ValueError("duplicate input paths")
    manifest = load_test_manifest(args.test_manifest)
    selected, selection = select_positions(paths, manifest, args.partition, args.limit, args.seed)
    if not selected:
        raise ValueError("no usable positions selected")
    output.parent.mkdir(parents=True, exist_ok=True)
    state_path = output.with_suffix(".state.sqlite3")
    plan_path = output.with_suffix(".plan.json")
    report_path = output.with_suffix(".report.json")
    plan = {
        "format": "xiangqi-pikafish-label-plan-v1",
        "engine": str(args.engine.resolve()), "engine_sha256": file_fingerprint(args.engine),
        "network": str(args.network.resolve()), "network_sha256": file_fingerprint(args.network),
        "inputs": {source_id(path): file_fingerprint(path) for path in paths},
        "protected_manifest": str(args.test_manifest.resolve()),
        "protected_manifest_sha256": file_fingerprint(args.test_manifest),
        "protected_identity": manifest.get("identity_sha256"),
        "partition": args.partition, "limit": args.limit, "seed": args.seed,
        "nodes": args.nodes if args.depth is None else None, "depth": args.depth,
        "threads": args.threads, "hash_mb": args.hash_mb,
        "timeout": args.timeout, "cp_multiplier": args.cp_multiplier,
        "max_score": args.max_score, "selection": selection,
        "selection_sha256": hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest(),
        "score_convention": "side-to-move Pikafish UCI cp multiplied by cp_multiplier; not assumed equal to hand material units",
        "history_convention": "standalone FEN, fresh hash, no invented repetition history",
        "result_convention": "unknown; old weak-player outcome preserved only in original metadata",
    }
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != plan:
            raise ValueError("resume configuration or input changed; use a new output")
        if not state_path.exists():
            raise FileNotFoundError("resume database missing")
    else:
        if output.exists() or state_path.exists():
            raise FileExistsError("output/state exists without a matching plan")
        write_json(plan_path, plan)
    connection = sqlite3.connect(state_path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("CREATE TABLE IF NOT EXISTS records (ordinal INTEGER PRIMARY KEY, status TEXT NOT NULL, payload TEXT)")
    connection.commit()
    completed = {row[0] for row in connection.execute("SELECT ordinal FROM records")}
    start = time.monotonic()
    report: dict[str, Any] = {"format": "xiangqi-pikafish-label-report-v1", "status": "running", "plan": plan}
    print(f"selected={len(selected)} resumed={len(completed)} partition={args.partition}", flush=True)
    try:
        with UciTeacher(args.engine, args.network, threads=args.threads, hash_mb=args.hash_mb, timeout=args.timeout) as teacher:
            report["teacher_identity"] = teacher.identity
            for ordinal, position in enumerate(selected):
                if ordinal in completed:
                    continue
                analysis = teacher.analyze(position["fen"], nodes=args.nodes, depth=args.depth)
                if analysis is None:
                    status = "terminal"
                elif analysis.score_kind == "mate":
                    status = "mate"
                elif analysis.bound is not None:
                    status = "bound"
                elif abs(analysis.score * args.cp_multiplier) > args.max_score:
                    status = "extreme"
                else:
                    status = "accepted"
                payload = None
                if status == "accepted":
                    record = {
                        **position, "score": analysis.score * args.cp_multiplier, "weight": 1.0,
                        "depth": analysis.depth, "best_move": analysis.best_move, "result": None,
                        "source": "pikafish-offline", "teacher": {**asdict(analysis),
                            "engine_sha256": plan["engine_sha256"], "network_sha256": plan["network_sha256"],
                            "cp_multiplier": args.cp_multiplier, "history": "standalone-fen"},
                    }
                    payload = json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                connection.execute("INSERT INTO records VALUES (?,?,?)", (ordinal, status, payload))
                connection.commit()
                completed.add(ordinal)
                if len(completed) % 250 == 0 or len(completed) == len(selected):
                    counts = dict(connection.execute("SELECT status, COUNT(*) FROM records GROUP BY status"))
                    print(f"progress={len(completed)}/{len(selected)} {counts} elapsed={time.monotonic()-start:.1f}s", flush=True)
                    report.update(processed=len(completed), counts=counts)
                    write_json(report_path, report)
        exported = export_records(connection, output)
        if not exported:
            raise ValueError("all samples filtered; no training dataset produced")
        report.update(status="complete", processed=len(completed), exported=exported,
                      counts=dict(connection.execute("SELECT status, COUNT(*) FROM records GROUP BY status")),
                      output_sha256=file_fingerprint(output), elapsed_seconds=time.monotonic()-start)
        print(f"complete: {exported} labels -> {output}", flush=True)
    except BaseException as error:
        report.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      processed=len(completed), error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(report_path, report)
        connection.close()
    return report


def parse_args() -> argparse.Namespace:
    """解析数据源、教师、预算与受保护测试集；默认单线程固定 5000 节点。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+", type=Path)
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument("--network", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--test-manifest", type=Path, default=DEFAULT_TEST_MANIFEST)
    parser.add_argument("--partition", choices=("development", "fixed-test"), default="development")
    parser.add_argument("--limit", type=int, default=10000, help="balanced phase sample cap; 0 means all eligible positions")
    parser.add_argument("--seed", type=int, default=2026)
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument("--nodes", type=int, default=5000)
    budget.add_argument("--depth", type=int)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--hash-mb", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--cp-multiplier", type=float, default=1.0)
    parser.add_argument("--max-score", type=float, default=3000.0)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    with run_lock(arguments.output.resolve().with_suffix(".lock")):
        label(arguments)
