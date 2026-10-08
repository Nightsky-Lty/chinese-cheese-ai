"""窗口截图、交互标定、走法预览和点击的命令行入口。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .capture import CapturedFrame, capture_window, save_frame
from .config import GuiConfig, load_config, save_config
from .control import execute_move
from .engine_client import analyze_fen
from .game_controller import ControllerEvent, GameController
from .geometry import BoardCalibration, Point, Rect
from .observer import GameObserver, ObserverEvent
from .protocol_client import ProtocolEngineClient
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


def _print_observer_event(event: ObserverEvent) -> None:
    """以便于终端阅读的格式输出观察器事件。"""

    if event.kind == "rejected":
        print(f"[rejected] {event.message}")
        return
    if event.kind == "initialized":
        print(f"[initialized] {event.fen}")
    else:
        print(f"[move] {event.move} -> {event.fen}")
    if event.best_move is not None:
        print(
            f"[suggestion] {event.best_move} score={event.score} depth={event.depth}"
        )


def _print_controller_event(event: ControllerEvent, *, execute: bool) -> None:
    """输出连续对局控制器的状态变化。"""

    if event.kind == "initialized":
        fen = event.observer_event.fen if event.observer_event is not None else None
        print(f"[initialized] {fen or 'board accepted'}")
    elif event.kind == "opponent_move":
        print(f"[opponent] {event.move}")
    elif event.kind == "move_requested":
        action = "clicked" if execute else "preview-only"
        print(f"[ai-move] {event.move} ({action}, waiting for confirmation)")
    elif event.kind == "move_confirmed":
        print(f"[confirmed] {event.move}")
    elif event.kind == "paused":
        print(f"[paused] {event.message}")
    elif event.kind == "finished":
        print(f"[finished] {event.message}")


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

    observe = commands.add_parser("observe", help="只读跟踪稳定局面和合法走法")
    observe.add_argument("--config", default="gui/wechat_board.json", help="标定 JSON")
    observe.add_argument(
        "--templates", default="gui/templates/jj_default.npz", help="棋子模板文件"
    )
    source = observe.add_mutually_exclusive_group(required=True)
    source.add_argument("--live", action="store_true", help="持续只读截图当前窗口")
    source.add_argument("--image-dir", help="按文件名顺序读取离线截图目录")
    observe.add_argument("--query", help="覆盖配置中的窗口搜索文本")
    observe.add_argument("--index", type=int, default=0, help="匹配结果序号")
    observe.add_argument(
        "--side", choices=("red", "black"), required=True, help="初始行棋方"
    )
    observe.add_argument(
        "--protocol", default="build/make/xiangqi_protocol", help="常驻协议程序"
    )
    observe.add_argument("--nnue", help="可选 NNUE 权重")
    observe.add_argument("--depth", type=int, default=4, help="提示着搜索深度")
    observe.add_argument("--stable-frames", type=int, default=3, help="稳定帧门槛")
    observe.add_argument("--interval", type=float, default=0.2, help="实时截图间隔秒数")
    observe.add_argument("--max-frames", type=int, help="最多处理帧数，便于测试")
    observe.add_argument(
        "--repeat-each", type=int, default=1, help="离线模式下每张图片重复提交次数"
    )

    control = commands.add_parser("control", help="连续对局闭环；默认只预览不点击")
    control.add_argument("--config", default="gui/wechat_board.json", help="标定 JSON")
    control.add_argument(
        "--templates", default="gui/templates/jj_default.npz", help="棋子模板文件"
    )
    control_source = control.add_mutually_exclusive_group(required=True)
    control_source.add_argument("--live", action="store_true", help="持续截图目标窗口")
    control_source.add_argument("--image-dir", help="按文件名顺序读取离线截图目录")
    control.add_argument("--query", help="覆盖配置中的窗口搜索文本")
    control.add_argument("--index", type=int, default=0, help="匹配结果序号")
    control.add_argument(
        "--side", choices=("red", "black"), required=True, help="初始行棋方"
    )
    control.add_argument(
        "--ai-side", choices=("red", "black"), required=True, help="引擎控制方"
    )
    control.add_argument(
        "--protocol", default="build/make/xiangqi_protocol", help="常驻协议程序"
    )
    control.add_argument("--nnue", help="可选 NNUE 权重")
    control.add_argument("--depth", type=int, default=4, help="引擎搜索深度")
    control.add_argument("--stable-frames", type=int, default=3, help="稳定帧门槛")
    control.add_argument("--interval", type=float, default=0.2, help="截图间隔秒数")
    control.add_argument("--click-interval", type=float, default=0.25, help="两次点击间隔")
    control.add_argument(
        "--confirmation-frames", type=int, default=30, help="落子确认超时帧数"
    )
    control.add_argument("--max-frames", type=int, help="最多处理帧数，便于测试")
    control.add_argument(
        "--repeat-each", type=int, default=1, help="离线模式下每张图片重复提交次数"
    )
    control.add_argument(
        "--execute", action="store_true", help="显式允许实时模式发送鼠标点击"
    )
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

        if arguments.command == "observe":
            if arguments.interval <= 0:
                raise ValueError("capture interval must be positive")
            if arguments.repeat_each <= 0:
                raise ValueError("repeat-each must be positive")
            library = TemplateLibrary.load(arguments.templates)
            recognizer = TemplatePieceRecognizer(library)
            live_window = None
            image_paths: list[Path] = []
            if arguments.live:
                query = arguments.query or config.window_query
                live_window = _select_window(query, arguments.index)
            else:
                directory = Path(arguments.image_dir)
                if not directory.is_dir():
                    raise RuntimeError(f"image directory does not exist: {directory}")
                image_paths = sorted(
                    path
                    for path in directory.iterdir()
                    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
                )
                if not image_paths:
                    raise RuntimeError(f"image directory contains no supported images: {directory}")

            frame_count = 0
            with ProtocolEngineClient(arguments.protocol, nnue=arguments.nnue) as engine:
                observer = GameObserver(
                    engine,
                    initial_side=arguments.side,
                    stable_frames=arguments.stable_frames,
                    search_depth=arguments.depth,
                )
                try:
                    while arguments.max_frames is None or frame_count < arguments.max_frames:
                        if live_window is not None:
                            images = [capture_window(live_window).pixels]
                        else:
                            if frame_count >= len(image_paths) * arguments.repeat_each:
                                break
                            path = image_paths[frame_count // arguments.repeat_each]
                            images = [_read_image(path)]
                        for image in images:
                            result = recognizer.recognize(image, config.calibration)
                            event = observer.process_recognition(result)
                            if event is not None:
                                _print_observer_event(event)
                            frame_count += 1
                        if live_window is not None:
                            time.sleep(arguments.interval)
                except KeyboardInterrupt:
                    print("观察器已停止")
            return 0

        if arguments.command == "control":
            if arguments.interval <= 0:
                raise ValueError("capture interval must be positive")
            if arguments.click_interval < 0:
                raise ValueError("click interval cannot be negative")
            if arguments.repeat_each <= 0:
                raise ValueError("repeat-each must be positive")
            if arguments.confirmation_frames <= 0:
                raise ValueError("confirmation-frames must be positive")
            if arguments.execute and not arguments.live:
                raise ValueError("--execute is only valid together with --live")

            recognizer = TemplatePieceRecognizer(
                TemplateLibrary.load(arguments.templates)
            )
            live_window = None
            live_window_id: int | None = None
            query = arguments.query or config.window_query
            image_paths: list[Path] = []
            if arguments.live:
                live_window = _select_window(query, arguments.index)
                live_window_id = live_window.window_id
            else:
                directory = Path(arguments.image_dir)
                if not directory.is_dir():
                    raise RuntimeError(f"image directory does not exist: {directory}")
                image_paths = sorted(
                    path
                    for path in directory.iterdir()
                    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
                )
                if not image_paths:
                    raise RuntimeError(
                        f"image directory contains no supported images: {directory}"
                    )

            def execute_requested_move(move_text: str) -> None:
                """在实时窗口换算坐标，并仅在显式授权时点击。"""

                if live_window is None:
                    return
                execute_move(
                    live_window,
                    config.calibration,
                    move_text,
                    execute=arguments.execute,
                    click_interval=arguments.click_interval,
                )

            if arguments.execute:
                print("[armed] real mouse clicks are enabled for the selected window")
            frame_count = 0
            with ProtocolEngineClient(arguments.protocol, nnue=arguments.nnue) as engine:
                observer = GameObserver(
                    engine,
                    initial_side=arguments.side,
                    stable_frames=arguments.stable_frames,
                    search_depth=arguments.depth,
                )
                controller = GameController(
                    observer,
                    ai_side=arguments.ai_side,
                    move_executor=execute_requested_move,
                    confirmation_frame_limit=arguments.confirmation_frames,
                )
                try:
                    while (
                        controller.active
                        and (
                            arguments.max_frames is None
                            or frame_count < arguments.max_frames
                        )
                    ):
                        if live_window_id is not None:
                            refreshed = next(
                                (
                                    candidate
                                    for candidate in find_windows(query)
                                    if candidate.window_id == live_window_id
                                ),
                                None,
                            )
                            if refreshed is None:
                                event = controller.pause(
                                    "selected window disappeared or became unavailable"
                                )
                                _print_controller_event(event, execute=arguments.execute)
                                break
                            live_window = refreshed
                            image = capture_window(live_window).pixels
                        else:
                            if frame_count >= len(image_paths) * arguments.repeat_each:
                                break
                            path = image_paths[frame_count // arguments.repeat_each]
                            image = _read_image(path)

                        recognition = recognizer.recognize(
                            image, config.calibration
                        )
                        events = controller.process_recognition(recognition)
                        for event in events:
                            _print_controller_event(event, execute=arguments.execute)
                        frame_count += 1
                        if live_window_id is not None:
                            time.sleep(arguments.interval)
                except KeyboardInterrupt:
                    event = controller.pause("stopped by user")
                    _print_controller_event(event, execute=arguments.execute)
                except (RuntimeError, OSError) as error:
                    event = controller.pause(f"capture or control failed: {error}")
                    _print_controller_event(event, execute=arguments.execute)
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
