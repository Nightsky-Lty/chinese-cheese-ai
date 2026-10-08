"""窗口截图、交互标定、走法预览和点击的命令行入口。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from .capture import CapturedFrame, capture_window, save_frame
from .config import GuiConfig, load_config, save_config
from .control import execute_move
from .engine_client import analyze_fen
from .geometry import BoardCalibration, Point, Rect
from .recognition import (
    TemplateLibrary,
    TemplatePieceRecognizer,
    build_initial_templates,
    draw_recognition_overlay,
)
from .windows import WindowInfo, find_windows


def _select_window(query: str, index: int) -> WindowInfo:
    """按搜索文本和序号选择当前可见窗口。"""

    matches = find_windows(query)
    if not matches:
        raise RuntimeError(f"no visible window matches {query!r}")
    if not 0 <= index < len(matches):
        raise RuntimeError(f"window index {index} is outside 0..{len(matches) - 1}")
    return matches[index]


def _print_window(index: int, window: WindowInfo) -> None:
    bounds = window.bounds
    print(
        f"[{index}] id={window.window_id} pid={window.process_id} "
        f"bounds=({bounds.left:.0f},{bounds.top:.0f},{bounds.width:.0f},{bounds.height:.0f}) "
        f"{window.label}"
    )


def _interactive_calibration(
    pixels: object, red_at_bottom: bool
) -> BoardCalibration:
    """显示截图并让用户依次点击棋盘的四个角交叉点。"""

    try:
        import cv2
        import numpy as np
    except ImportError as error:
        raise RuntimeError("interactive calibration requires opencv-python and numpy") from error

    image = np.asarray(pixels)
    height, width = image.shape[:2]
    scale = min(1.0, 1400.0 / width, 900.0 / height)
    shown_width = max(1, round(width * scale))
    shown_height = max(1, round(height * scale))
    original = cv2.resize(image, (shown_width, shown_height))
    selected: list[Point] = []
    window_name = "Board calibration: TL, TR, BR, BL | R reset | ESC cancel"

    def on_mouse(event: int, x: int, y: int, _flags: int, _parameter: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(selected) < 4:
            selected.append(Point(x / scale, y / scale))

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, on_mouse)
    try:
        while True:
            display = original.copy()
            for number, point in enumerate(selected, start=1):
                x = round(point.x * scale)
                y = round(point.y * scale)
                cv2.circle(display, (x, y), 7, (0, 0, 255), -1)
                cv2.putText(
                    display,
                    str(number),
                    (x + 9, y - 9),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
            cv2.imshow(window_name, display)
            key = cv2.waitKey(30) & 0xFF
            if key == 27:
                raise RuntimeError("calibration cancelled")
            if key in (ord("r"), ord("R")):
                selected.clear()
            if len(selected) == 4:
                break
    finally:
        cv2.destroyWindow(window_name)
    return BoardCalibration.from_pixel_corners(
        selected, width, height, red_at_bottom=red_at_bottom
    )


def _draw_preview(
    window: WindowInfo,
    config: GuiConfig,
    move: str,
    output: str | Path,
) -> Path:
    """在最新窗口截图上绘制走法起终点和箭头。"""

    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("move preview requires opencv-python") from error

    frame = capture_window(window)
    source_screen, target_screen = config.calibration.move_screen_points(move, window.bounds)

    def screen_to_pixel(point: Point) -> tuple[int, int]:
        horizontal = (point.x - window.bounds.left) / window.bounds.width
        vertical = (point.y - window.bounds.top) / window.bounds.height
        return round(horizontal * frame.width), round(vertical * frame.height)

    source = screen_to_pixel(source_screen)
    target = screen_to_pixel(target_screen)
    cv2.circle(frame.pixels, source, 12, (0, 255, 255), 3)
    cv2.circle(frame.pixels, target, 12, (0, 0, 255), 3)
    cv2.arrowedLine(frame.pixels, source, target, (255, 80, 20), 4, tipLength=0.12)
    return save_frame(frame, output)


def _read_image(path: str | Path) -> np.ndarray:
    """读取 OpenCV BGR 图片，并对无效路径提供明确错误。"""

    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"cannot read image: {path}")
    return image


def _save_pixels(pixels: np.ndarray, output: str | Path) -> Path:
    """使用现有截图保存器写出任意 BGR 图像。"""

    height, width = pixels.shape[:2]
    return save_frame(
        CapturedFrame(pixels, Rect(0, 0, float(width), float(height))),
        output,
    )


def build_parser() -> argparse.ArgumentParser:
    """创建 GUI 自动化命令行参数解析器。"""

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    windows = commands.add_parser("windows", help="列出可见窗口")
    windows.add_argument("--query", default="微信", help="按应用名或标题过滤")

    capture = commands.add_parser("capture", help="截取指定窗口")
    capture.add_argument("--query", default="微信", help="按应用名或标题过滤")
    capture.add_argument("--index", type=int, default=0, help="匹配结果序号")
    capture.add_argument("--output", default="gui/output/window.png", help="截图路径")

    calibrate = commands.add_parser("calibrate", help="交互点击棋盘四角并保存标定")
    calibrate.add_argument("--query", default="微信", help="按应用名或标题过滤")
    calibrate.add_argument("--index", type=int, default=0, help="匹配结果序号")
    calibrate.add_argument("--output", default="gui/wechat_board.json", help="标定 JSON")
    calibrate.add_argument(
        "--orientation",
        choices=("red-bottom", "black-bottom"),
        default="red-bottom",
        help="屏幕下方是哪一方",
    )

    preview = commands.add_parser("preview", help="截图并标出走法，不发送点击")
    preview.add_argument("--config", default="gui/wechat_board.json", help="标定 JSON")
    preview.add_argument("--query", help="覆盖配置中的窗口搜索文本")
    preview.add_argument("--index", type=int, default=0, help="匹配结果序号")
    preview.add_argument("--move", required=True, help="引擎走法，例如 b0c2")
    preview.add_argument("--output", default="gui/output/preview.png", help="预览图路径")

    move = commands.add_parser("move", help="换算坐标；加 --execute 后才实际点击")
    move.add_argument("--config", default="gui/wechat_board.json", help="标定 JSON")
    move.add_argument("--query", help="覆盖配置中的窗口搜索文本")
    move.add_argument("--index", type=int, default=0, help="匹配结果序号")
    move.add_argument("--move", required=True, help="引擎走法，例如 b0c2")
    move.add_argument("--execute", action="store_true", help="确实发送两次鼠标点击")
    move.add_argument("--interval", type=float, default=0.25, help="两次点击的间隔秒数")

    templates = commands.add_parser("learn-templates", help="从标准初始局面生成棋子模板")
    templates.add_argument("--config", default="gui/wechat_board.json", help="标定 JSON")
    templates.add_argument("--image", required=True, help="标准初始局面截图")
    templates.add_argument(
        "--output", default="gui/templates/jj_default.npz", help="模板输出文件"
    )

    recognize = commands.add_parser("recognize", help="识别棋盘、生成 FEN 并可选调用引擎")
    recognize.add_argument("--config", default="gui/wechat_board.json", help="标定 JSON")
    recognize.add_argument(
        "--templates", default="gui/templates/jj_default.npz", help="棋子模板文件"
    )
    recognize.add_argument("--image", help="识别静态图片；省略则截图当前窗口")
    recognize.add_argument("--query", help="覆盖配置中的窗口搜索文本")
    recognize.add_argument("--index", type=int, default=0, help="匹配结果序号")
    recognize.add_argument(
        "--side", choices=("red", "black"), required=True, help="当前行棋方"
    )
    recognize.add_argument(
        "--output", default="gui/output/recognized.png", help="识别叠加图"
    )
    recognize.add_argument("--json", dest="json_output", help="可选结构化结果 JSON")
    recognize.add_argument("--engine", help="可选 xiangqi_cli 路径")
    recognize.add_argument("--depth", type=int, default=4, help="引擎搜索深度")
    recognize.add_argument("--nnue", help="可选 NNUE 权重")
    return parser


def main(argv: list[str] | None = None) -> int:
    """执行 GUI 工具命令并将可预期错误转换为退出码 1。"""

    try:
        arguments = build_parser().parse_args(argv)
        if arguments.command == "windows":
            matches = find_windows(arguments.query)
            for index, window in enumerate(matches):
                _print_window(index, window)
            if not matches:
                print("没有找到匹配窗口", file=sys.stderr)
                return 1
            return 0

        if arguments.command == "capture":
            window = _select_window(arguments.query, arguments.index)
            path = save_frame(capture_window(window), arguments.output)
            print(f"已截图: {path.resolve()}")
            return 0

        if arguments.command == "calibrate":
            window = _select_window(arguments.query, arguments.index)
            frame = capture_window(window)
            calibration = _interactive_calibration(
                frame.pixels, arguments.orientation == "red-bottom"
            )
            path = save_config(GuiConfig(arguments.query, calibration), arguments.output)
            print(f"已保存标定: {path.resolve()}")
            return 0

        config = load_config(arguments.config)
        if arguments.command == "learn-templates":
            image = _read_image(arguments.image)
            library = build_initial_templates(image, config.calibration)
            path = library.save(arguments.output)
            print(f"已生成棋子模板: {path.resolve()}")
            return 0

        if arguments.command == "recognize":
            if arguments.image:
                image = _read_image(arguments.image)
            else:
                query = arguments.query or config.window_query
                window = _select_window(query, arguments.index)
                image = capture_window(window).pixels
            recognizer = TemplatePieceRecognizer(TemplateLibrary.load(arguments.templates))
            result = recognizer.recognize(image, config.calibration)
            overlay = draw_recognition_overlay(image, config.calibration, result)
            overlay_path = _save_pixels(overlay, arguments.output)
            fen = result.to_fen(arguments.side)
            print(f"FEN: {fen}")
            print(f"识别叠加图: {overlay_path.resolve()}")
            payload: dict[str, object] = {
                "fen": fen,
                "unknown_count": result.unknown_count,
                "pieces": [
                    {
                        "file": observation.file_index,
                        "rank": observation.rank,
                        "piece": observation.piece,
                        "confidence": observation.confidence,
                        "occupancy": observation.occupancy,
                    }
                    for observation in result.observations
                    if observation.piece != "."
                ],
            }
            if arguments.engine:
                analysis = analyze_fen(
                    arguments.engine,
                    fen,
                    depth=arguments.depth,
                    nnue=arguments.nnue,
                )
                payload["analysis"] = {
                    "best_move": analysis.best_move,
                    "score": analysis.score,
                    "depth": analysis.depth,
                }
                print(
                    f"引擎建议: {analysis.best_move} "
                    f"score={analysis.score} depth={analysis.depth}"
                )
            if arguments.json_output:
                destination = Path(arguments.json_output)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                print(f"结构化结果: {destination.resolve()}")
            return 0

        query = arguments.query or config.window_query
        window = _select_window(query, arguments.index)
        if arguments.command == "preview":
            path = _draw_preview(window, config, arguments.move, arguments.output)
            print(f"已生成预览: {path.resolve()}")
            return 0
        if arguments.command == "move":
            result = execute_move(
                window,
                config.calibration,
                arguments.move,
                execute=arguments.execute,
                click_interval=arguments.interval,
            )
            action = "已点击" if result.executed else "仅预览坐标，未点击"
            print(
                f"{action}: ({result.source.x:.1f}, {result.source.y:.1f}) -> "
                f"({result.target.x:.1f}, {result.target.y:.1f})"
            )
            return 0
        raise RuntimeError(f"unsupported command: {arguments.command}")
    except (RuntimeError, ValueError, OSError) as error:
        print(f"错误: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
