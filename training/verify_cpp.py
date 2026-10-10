"""比较相同权重和局面在 Python 与 C++ 中的原始 NNUE 输出。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import tempfile
from pathlib import Path

import torch

from .checkpoint import load_model
from .dataset import TrainingSample, collate_samples
from .export_nnue import export_model
from .halfka import encode_fen

DEFAULT_FENS = [
    "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w",
    "4k4/9/9/9/4p4/9/9/4R4/9/4K4 w",
    "3k5/9/9/9/4p4/9/4P4/9/9/4K4 b",
]
RAW_PATTERN = re.compile(
    r"nnue raw: ([+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|inf(?:inity)?|nan))",
    re.IGNORECASE,
)


def compare_scores(
    expected: float,
    actual: float,
    absolute_tolerance: float = 1.0e-3,
    relative_tolerance: float = 1.0e-6,
) -> dict[str, float | bool | None]:
    """Compare finite raw scores using an absolute-plus-relative tolerance."""

    if (
        not math.isfinite(absolute_tolerance)
        or not math.isfinite(relative_tolerance)
        or absolute_tolerance < 0.0
        or relative_tolerance < 0.0
    ):
        raise ValueError("score tolerances must be finite and non-negative")
    finite_expected = math.isfinite(expected)
    finite_actual = math.isfinite(actual)
    finite = finite_expected and finite_actual
    difference = abs(expected - actual) if finite else None
    allowed_limit = (
        absolute_tolerance + relative_tolerance * abs(expected)
        if finite_expected
        else None
    )
    return {
        "expected": expected if finite_expected else None,
        "actual": actual if finite_actual else None,
        "difference": difference,
        "allowed_limit": allowed_limit,
        "finite": finite,
        "matches": bool(finite and difference <= allowed_limit),
    }


def python_score(model: torch.nn.Module, fen: str) -> float:
    """使用 Python 网络评估一个局面。

    参数:
        model: 已加载参数的 PyTorch NNUE 网络。
        fen: 包含行棋方的中国象棋 FEN。

    返回:
        当前行棋方视角、未经取整的 float32 分数。
    """

    sample = TrainingSample(encode_fen(fen), 0.0, fen=fen)
    batch = collate_samples([sample])
    model.eval()
    with torch.no_grad():
        return float(model.score_batch(batch)[0])


def cpp_score(engine: Path, model_path: Path, fen: str) -> float:
    """调用 C++ CLI 获取同一局面的原始 NNUE 分数。

    参数:
        engine: 已构建的 ``xiangqi_cli`` 路径。
        model_path: ``XQNNUEF1`` 模型路径。
        fen: 包含行棋方的中国象棋 FEN。

    返回:
        从 ``nnue raw:`` 输出中解析出的浮点分数。

    异常:
        subprocess.CalledProcessError: C++ 引擎执行失败。
        RuntimeError: 引擎输出中没有原始 NNUE 分数。
    """

    completed = subprocess.run(
        [str(engine), "--fen", fen, "--nnue", str(model_path), "--nnue-raw"],
        check=True,
        capture_output=True,
        text=True,
    )
    match = RAW_PATTERN.search(completed.stdout)
    if not match:
        raise RuntimeError("C++ output did not contain an NNUE raw score")
    return float(match.group(1))


def parse_args() -> argparse.Namespace:
    """解析检查点、引擎、测试局面和误差容限。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--engine", type=Path, default=Path("build/make/xiangqi_cli"))
    parser.add_argument("--fen", action="append", dest="fens")
    parser.add_argument("--fen-file", type=Path)
    parser.add_argument(
        "--tolerance", type=float, default=1.0e-3,
        help="absolute raw-score tolerance in engine units (default: 0.001)",
    )
    parser.add_argument(
        "--relative-tolerance", type=float, default=1.0e-6,
        help="relative raw-score tolerance (default: 1e-6)",
    )
    parser.add_argument(
        "--output-json", type=Path,
        help="optionally write per-position parity diagnostics as JSON",
    )
    return parser.parse_args()


def main() -> None:
    """临时导出模型并逐局面对比 Python/C++ 前向传播。"""

    args = parse_args()
    if (
        not math.isfinite(args.tolerance)
        or not math.isfinite(args.relative_tolerance)
        or args.tolerance < 0.0
        or args.relative_tolerance < 0.0
    ):
        raise SystemExit("tolerances must be finite and non-negative")
    if not args.engine.is_file():
        raise FileNotFoundError(f"C++ engine not found: {args.engine}; run make first")
    fens = list(args.fens or DEFAULT_FENS)
    if args.fen_file:
        fens.extend(
            line.strip()
            for line in args.fen_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )

    model = load_model(args.checkpoint, "cpu")
    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    results: list[dict[str, object]] = []
    # 使用临时模型确保验证对象就是当前检查点，结束后自动清理。
    with tempfile.TemporaryDirectory(prefix="xiangqi-nnue-") as directory:
        model_path = export_model(model, Path(directory) / "verify.nnue")
        for fen in fens:
            expected = python_score(model, fen)
            actual = cpp_score(args.engine, model_path, fen)
            comparison = compare_scores(
                expected, actual, args.tolerance, args.relative_tolerance
            )
            comparison["fen"] = fen
            results.append(comparison)
            expected_text = f"{expected:.9g}" if math.isfinite(expected) else str(expected)
            actual_text = f"{actual:.9g}" if math.isfinite(actual) else str(actual)
            difference = comparison["difference"]
            difference_text = f"{difference:.9g}" if difference is not None else "non-finite"
            allowed_limit = comparison["allowed_limit"]
            allowed_text = (
                f"{allowed_limit:.9g}" if allowed_limit is not None else "unavailable"
            )
            print(
                f"difference={difference_text} allowed={allowed_text} "
                f"python={expected_text} cpp={actual_text} fen={fen}"
            )
    mismatches = [result for result in results if not result["matches"]]
    if args.output_json:
        report = {
            "format": "xiangqi-nnue-parity-report-v1",
            "model": {
                "checkpoint": str(args.checkpoint),
                "checkpoint_sha256": checkpoint_hash,
                "score_scale_engine_units_per_raw_output": model.score_scale,
            },
            "engine": str(args.engine),
            "tolerances": {
                "absolute_engine_units": args.tolerance,
                "relative": args.relative_tolerance,
            },
            "position_count": len(results),
            "matched_count": len(results) - len(mismatches),
            "mismatch_count": len(mismatches),
            "positions": results,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    if mismatches:
        raise SystemExit(
            f"Python/C++ mismatch for {len(mismatches)}/{len(results)} positions; "
            f"see {args.output_json}" if args.output_json else
            f"Python/C++ mismatch for {len(mismatches)}/{len(results)} positions"
        )
    print(f"verified {len(fens)} positions")


if __name__ == "__main__":
    main()
