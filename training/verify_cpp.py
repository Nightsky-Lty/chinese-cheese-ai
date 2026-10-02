"""比较相同权重和局面在 Python 与 C++ 中的原始 NNUE 输出。"""

from __future__ import annotations

import argparse
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
RAW_PATTERN = re.compile(r"nnue raw: ([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")


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
        return float(model.forward_batch(batch)[0])


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
    parser.add_argument("--tolerance", type=float, default=1.0e-4)
    return parser.parse_args()


def main() -> None:
    """临时导出模型并逐局面对比 Python/C++ 前向传播。"""

    args = parse_args()
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
    # 使用临时模型确保验证对象就是当前检查点，结束后自动清理。
    with tempfile.TemporaryDirectory(prefix="xiangqi-nnue-") as directory:
        model_path = export_model(model, Path(directory) / "verify.nnue")
        for fen in fens:
            expected = python_score(model, fen)
            actual = cpp_score(args.engine, model_path, fen)
            difference = abs(expected - actual)
            print(
                f"difference={difference:.9g} python={expected:.9g} "
                f"cpp={actual:.9g} fen={fen}"
            )
            if difference > args.tolerance:
                raise SystemExit(
                    f"Python/C++ mismatch exceeds tolerance {args.tolerance}"
                )
    print(f"verified {len(fens)} positions")


if __name__ == "__main__":
    main()
