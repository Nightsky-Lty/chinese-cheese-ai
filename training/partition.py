"""按整局拆分训练数据，并用固定测试局面阻止后续回放泄漏。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_FORMAT = "xiangqi-fixed-test-v1"
DEFAULT_TEST_MANIFEST = PROJECT_ROOT / "training/splits/hand-teacher-v2-test.json"
DEFAULT_QUARANTINE_MANIFEST = PROJECT_ROOT / "training/splits/legacy-narrow-scores.json"
QUARANTINE_FORMAT = "xiangqi-score-quarantine-v1"
TEST_SOURCE_SIDECAR_FORMAT = "xiangqi-fixed-test-sources-v1"


def source_id(path: Path) -> str:
    """返回跨工作目录稳定的输入文件标识。"""

    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def file_fingerprint(path: str | Path) -> str:
    """以流式 SHA-256 标识数据源文件的精确字节内容。"""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_source_path(identifier: str) -> Path:
    """把 manifest 内的数据源 ID 解析回本机路径。"""

    path = Path(identifier)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _source_from_group_key(group: str) -> str | None:
    """从 ``source#game=...`` 等稳定分组键中还原源文件 ID。"""

    source, separator, suffix = group.rpartition("#")
    if separator and source and (
        suffix.startswith("game=") or suffix in {"ungrouped", "file"}
    ):
        return source
    return None


def _test_identity_fingerprint(payload: dict[str, Any]) -> str:
    identity = {
        key: payload[key]
        for key in ("format", "seed", "fraction", "groups", "position_keys")
    }
    if "source_fingerprints" in payload:
        identity["source_fingerprints"] = payload["source_fingerprints"]
    encoded = json.dumps(
        identity, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def canonical_position_key(fen: str) -> str:
    """把局面及其水平镜像映射到同一标识，并保留行棋方。"""

    fields = fen.split()
    if len(fields) < 2:
        raise ValueError("FEN must contain a board and side to move")
    ranks = []
    for rank in fields[0].split("/"):
        expanded = "".join("." * int(char) if char.isdigit() else char for char in rank)
        if len(expanded) != 9:
            raise ValueError("FEN rank must contain 9 squares")
        ranks.append(expanded)
    if len(ranks) != 10:
        raise ValueError("FEN must contain 10 ranks")
    board = "/".join(ranks)
    mirror = "/".join(rank[::-1] for rank in ranks)
    side = "w" if fields[1] in ("w", "r") else fields[1]
    canonical = min(board, mirror) + " " + side
    return hashlib.blake2b(canonical.encode("ascii"), digest_size=16).hexdigest()


def game_group_key(path: Path, record: dict[str, Any]) -> str:
    """同一数据源内同一局的样本共享分组；无编号时整文件成组。"""

    game = record.get("game_id")
    if game is None:
        game = record.get("game")
    if (type(game) is int and game >= 0) or (
        isinstance(game, str) and game.strip()
    ):
        # Keep the historical integer key format. Percent-encoding prevents
        # arbitrary string IDs from colliding with the source delimiter.
        from urllib.parse import quote

        encoded = str(game) if type(game) is int else quote(game.strip(), safe="-_.")
        return f"{source_id(path)}#game={encoded}"
    return f"{source_id(path)}#ungrouped"


def stable_fraction(seed: int, key: str) -> float:
    """稳定地将分组映射到 [0, 1)，不依赖 Python 哈希随机化。"""

    digest = hashlib.blake2b(
        f"{seed}:{key}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big") / 2**64


def create_test_manifest(samples: Sequence[Any], fraction: float, seed: int) -> dict[str, Any]:
    """从完整对局中选择永久保留的测试分组及其镜像局面标识。"""

    if not 0.0 < fraction < 1.0:
        raise ValueError("test fraction must be in (0, 1)")
    groups = {sample.group_key for sample in samples}
    if not groups or "" in groups:
        raise ValueError("test manifest requires samples with game group keys")
    selected = {group for group in groups if stable_fraction(seed, group) < fraction}
    if not selected:
        selected = {min(groups, key=lambda group: stable_fraction(seed, group))}
    identities = {
        sample.position_key for sample in samples if sample.group_key in selected
    }
    source_fingerprints: dict[str, str] = {}
    for sample in samples:
        if sample.group_key not in selected:
            continue
        source = _source_from_group_key(sample.group_key)
        fingerprint = getattr(sample, "source_fingerprint", "")
        if source:
            if not fingerprint:
                raise ValueError(f"test source has no fingerprint: {source}")
            previous = source_fingerprints.setdefault(source, fingerprint)
            if previous != fingerprint:
                raise ValueError(f"inconsistent source fingerprint for {source}")
    payload = {
        "format": MANIFEST_FORMAT,
        "seed": seed,
        "fraction": fraction,
        "groups": sorted(selected),
        "position_keys": sorted(identities),
        "source_fingerprints": dict(sorted(source_fingerprints.items())),
    }
    payload["identity_sha256"] = _test_identity_fingerprint(payload)
    return payload


def load_test_manifest(path: Path) -> dict[str, Any]:
    """读取并校验固定测试集清单。"""

    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"invalid fixed-test manifest: {path}")
    if (
        payload.get("format") != MANIFEST_FORMAT
        or not isinstance(payload.get("groups"), list)
        or not isinstance(payload.get("position_keys"), list)
        or not payload["groups"]
        or not payload["position_keys"]
        or any(not isinstance(value, str) or not value for value in payload["groups"])
        or any(not isinstance(value, str) or not value for value in payload["position_keys"])
        or any(
            len(value) != 32 or any(char not in "0123456789abcdef" for char in value)
            for value in payload["position_keys"]
        )
        or len(set(payload["groups"])) != len(payload["groups"])
        or len(set(payload["position_keys"])) != len(payload["position_keys"])
        or type(payload.get("seed")) is not int
        or type(payload.get("fraction")) not in (int, float)
        or not math.isfinite(float(payload["fraction"]))
        or not 0.0 < float(payload["fraction"]) < 1.0
    ):
        raise ValueError(f"invalid fixed-test manifest: {path}")

    sidecar_path = path.with_name(f"{path.stem}-sources.json")
    sidecar_verified = False
    if "source_fingerprints" not in payload and sidecar_path.is_file():
        with sidecar_path.open("r", encoding="utf-8") as stream:
            sidecar = json.load(stream)
        if (
            not isinstance(sidecar, dict)
            or sidecar.get("format") != TEST_SOURCE_SIDECAR_FORMAT
            or sidecar.get("manifest_sha256") != file_fingerprint(path)
        ):
            raise ValueError(f"fixed-test source sidecar does not match manifest: {sidecar_path}")
        payload["source_fingerprints"] = sidecar.get("source_fingerprints")
        sidecar_verified = True

    fingerprints = payload.get("source_fingerprints")
    if (
        not isinstance(fingerprints, dict)
        or any(
            not isinstance(source, str)
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
            for source, digest in fingerprints.items()
        )
    ):
        raise ValueError(f"invalid source fingerprints in fixed-test manifest: {path}")
    identity_fingerprint = payload.get("identity_sha256")
    if identity_fingerprint is not None and identity_fingerprint != _test_identity_fingerprint(payload):
        raise ValueError(f"fixed-test identity was modified: {path}")
    if fingerprints and identity_fingerprint is None and not sidecar_verified:
        raise ValueError(f"fixed-test identity fingerprint is missing: {path}")
    for group in payload["groups"]:
        source = _source_from_group_key(group)
        if source and source not in payload["source_fingerprints"]:
            raise ValueError(f"fixed-test source is not fingerprinted: {source}")
    for source, expected in payload["source_fingerprints"].items():
        source_path = manifest_source_path(source)
        if not source_path.is_file():
            raise FileNotFoundError(f"fixed-test source is missing: {source_path}")
        actual = file_fingerprint(source_path)
        if actual != expected:
            raise ValueError(f"fixed-test source changed since manifest creation: {source}")
    return payload


def load_quarantine_manifest(path: Path) -> tuple[set[str], float]:
    """读取需要隔离窄范围旧评分的精确数据源清单。"""

    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if (
        not isinstance(payload, dict)
        or payload.get("format") != QUARANTINE_FORMAT
        or not isinstance(payload.get("sources"), list)
        or any(not isinstance(value, str) for value in payload["sources"])
        or type(payload.get("max_absolute_score")) not in (int, float)
        or payload["max_absolute_score"] < 0
    ):
        raise ValueError(f"invalid score quarantine manifest: {path}")
    return set(payload["sources"]), float(payload["max_absolute_score"])


def write_test_manifest(path: Path, payload: dict[str, Any]) -> None:
    """原子写入不可随训练重新抽样的测试集清单。"""

    if path.exists():
        raise FileExistsError(f"fixed-test manifest already exists: {path}")
    payload = dict(payload)
    payload.setdefault("identity_sha256", _test_identity_fingerprint(payload))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def split_indices(
    samples: Sequence[Any], validation_fraction: float, seed: int,
    manifest: dict[str, Any] | None,
) -> tuple[list[int], list[int], list[int], int]:
    """按整局分区，并删除跨分区重复的局面及其水平镜像。"""

    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation fraction must be in [0, 1)")
    test_groups = set(manifest["groups"]) if manifest else set()
    test_positions = set(manifest["position_keys"]) if manifest else set()

    source_fingerprints = manifest.get("source_fingerprints", {}) if manifest else {}
    if not isinstance(source_fingerprints, dict):
        raise ValueError("fixed-test source fingerprints must be a mapping")
    for sample in samples:
        if not sample.group_key or not sample.position_key:
            raise ValueError("every sample needs a game group key and position key")
        source = _source_from_group_key(sample.group_key)
        expected = source_fingerprints.get(source) if source else None
        actual = getattr(sample, "source_fingerprint", "")
        if expected is not None and actual != expected:
            raise ValueError(f"fixed-test source changed in loaded samples: {source}")

    test = [i for i, sample in enumerate(samples) if sample.group_key in test_groups]
    unexpected_test_positions = {
        samples[i].position_key for i in test
    } - test_positions
    if unexpected_test_positions:
        raise ValueError("fixed-test group contains positions absent from its manifest")

    candidates = [
        i for i, sample in enumerate(samples)
        if sample.group_key not in test_groups
        and sample.position_key not in test_positions
    ]
    groups = {samples[i].group_key for i in candidates}
    validation_groups = {
        group for group in groups
        if validation_fraction > 0.0
        and stable_fraction(seed + 1, group) < validation_fraction
    }
    if validation_fraction > 0.0 and len(groups) > 1:
        if not validation_groups:
            validation_groups = {min(groups, key=lambda group: stable_fraction(seed + 1, group))}
        if validation_groups == groups:
            validation_groups.remove(max(groups, key=lambda group: stable_fraction(seed + 1, group)))
    else:
        validation_groups.clear()

    # For each canonical position, choose one stable owning game across all
    # non-test groups. This removes mirror duplicates between train and
    # validation while keeping every remaining sample of a game in one split.
    position_owners: dict[str, str] = {}
    for index in candidates:
        sample = samples[index]
        current = position_owners.get(sample.position_key)
        if current is None or sample.group_key < current:
            position_owners[sample.position_key] = sample.group_key

    seen_test_positions: set[str] = set()
    deduplicated_test: list[int] = []
    for index in test:
        key = samples[index].position_key
        if key not in seen_test_positions:
            seen_test_positions.add(key)
            deduplicated_test.append(index)

    train: list[int] = []
    validation: list[int] = []
    seen_group_positions: set[tuple[str, str]] = set()
    for index in candidates:
        sample = samples[index]
        if position_owners[sample.position_key] != sample.group_key:
            continue
        group_position = (sample.group_key, sample.position_key)
        if group_position in seen_group_positions:
            continue
        seen_group_positions.add(group_position)
        if sample.group_key in validation_groups:
            validation.append(index)
        else:
            train.append(index)

    test = deduplicated_test
    dropped = len(samples) - len(train) - len(validation) - len(test)
    return train, validation, test, dropped


def main() -> None:
    """从可信数据文件一次性创建固定测试集清单。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_TEST_MANIFEST)
    parser.add_argument("--fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    from .dataset import JsonlPositionDataset

    samples = [sample for path in args.datasets for sample in JsonlPositionDataset(path).samples]
    payload = create_test_manifest(samples, args.fraction, args.seed)
    write_test_manifest(args.output, payload)
    print(
        f"fixed test: {len(payload['groups'])} games, "
        f"{len(payload['position_keys'])} unique positions -> {args.output}"
    )


if __name__ == "__main__":
    main()
