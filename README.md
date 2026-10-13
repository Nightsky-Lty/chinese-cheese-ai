# Chinese Chess AI

从零实现的中国象棋 AI，包含 C++ 规则与搜索内核、Python NNUE 训练工具，以及独立的棋盘 GUI 适配层。

实际行棋由本项目的 Alpha-Beta/PVS 搜索器完成，可选择手工评估或 HalfKA NNUE。
Pikafish 仅作为可选的离线教师，不替代自己的引擎。项目也支持从 ModelScope 棋谱数据中抽取局面进行监督训练。

本 README 介绍项目结构、已实现能力和使用方法。具体训练进度、指标、对战成绩与运行记录保存在本地任务目录，不作为项目功能说明。

## 项目结构

```text
chinese-cheese-ai/
├── engine/
│   ├── include/xiangqi/              # C++ 公共接口
│   │   ├── types.hpp                # 棋子、颜色、坐标与走法
│   │   ├── position.hpp             # FEN、合法走法、执行与撤销
│   │   ├── game_history.hpp         # 对局历史与重复局面
│   │   ├── cycle_adjudicator.hpp    # 简化长将、长捉与和棋裁决
│   │   ├── evaluation.hpp           # 统一评估接口与手工评估
│   │   ├── halfka.hpp               # 双视角稀疏特征
│   │   ├── nnue.hpp                 # 可配置网络、模型加载与增量推理
│   │   ├── move_ordering.hpp        # SEE、Killer、Counter 与 History
│   │   ├── search.hpp               # Negamax、Alpha-Beta/PVS 与迭代加深
│   │   ├── transposition_table.hpp  # 分数边界与最佳着缓存
│   │   └── training_data.hpp        # FEN 标注与自我对弈采样
│   ├── src/                         # 实现与 CLI / 协议 / 数据生成入口
│   └── tests/                       # 规则、历史、评估和搜索检查
├── training/
│   ├── halfka.py / model.py          # 特征编码与 PyTorch 网络
│   ├── dataset.py / partition.py    # 样本读取、分区与固定测试保护
│   ├── train.py / checkpoint.py     # 训练、模型元数据与断点续训
│   ├── diagnostics.py               # 训练指标与分阶段诊断
│   ├── export_nnue.py / verify_cpp.py # 权重导出与跨语言推理检查
│   ├── modelscope_data.py           # SQLite 棋谱审计、重放与断点抽样
│   ├── modelscope_baseline.py       # 数据基线训练与导出
│   ├── modelscope_competition.py    # 抽样、训练与冠军竞争流水线
│   ├── uci_teacher.py / relabel.py  # 外部教师通信与离线重标注
│   ├── pikafish_pipeline.py         # 教师标注到训练、导出的流水线
│   ├── arena.py / baseline.py       # 对战评测与晋升闭环
│   ├── iterate.py / run_lock.py     # 多代迭代、恢复与任务锁
│   ├── DATA_FORMAT.md               # 数据、模型及运行产物格式
│   ├── examples/ / tests/           # 示例与训练工具检查
│   └── data/ / runs/ / tools/       # 本地数据、任务产物与教师工具
├── gui/
│   ├── windows.py / capture.py      # macOS / Windows 窗口与截图
│   ├── geometry.py / config.py      # 棋盘标定、坐标换算与配置
│   ├── recognition.py              # 棋子模板、识别与 FEN
│   ├── control.py                   # 走法预览与显式鼠标操作
│   ├── engine_client.py             # 单次 C++ 分析调用
│   ├── protocol_client.py           # 常驻引擎通信与完整历史
│   ├── board_state.py / observer.py # 稳定候选与合法走法推断
│   ├── game_controller.py / cli.py  # 对局状态机与命令行入口
│   └── tests/                       # 坐标、识别与控制器检查
├── .vscode/                         # 编辑器构建与调试配置
├── CMakeLists.txt / Makefile         # C++ 构建入口
└── compile_flags.txt                # clangd 参数
```

`training/data/`、`training/runs/`、`training/tools/`、GUI 截图、模板和本地标定配置均不提交 Git。
克隆仓库后，需要自行准备训练数据、模型和棋盘模板。

## 技术栈

| 层 | 技术与职责 |
| --- | --- |
| 引擎 | C++20、Make/CMake；规则、历史裁决、搜索与 NNUE 推理 |
| 训练 | Python、PyTorch、NumPy；AdamW、评分监督与检查点 |
| 数据 | SQLite 原库及抽样断点、JSONL 样本、FEN 局面、版本化二进制权重 |
| 视觉 | OpenCV、HSV 颜色检测、灰度字形模板匹配 |
| 窗口与输入 | macOS PyObjC/Quartz/Cocoa；Windows Win32/pywin32/mss；pyautogui |
| 通信 | Python subprocess 与常驻 C++ 进程的单行请求/响应协议 |

## 核心实现

### 规则、局面与历史

- 90 格棋盘、FEN 解析与输出、七类棋子的走法生成。
- 处理马腿、象眼、炮架、九宫、河界、将帅照面及将军状态。
- 伪合法走法经过己方将帅安全检查后成为合法走法；支持将死和困毙。
- 走法执行与撤销同步维护棋子、行棋方和增量 Zobrist 哈希。
- `GameHistory` 保存真实走法、吃子、将军、未吃子计数与历史哈希。

历史裁决使用项目约定的简化规则：同一棋盘且同一行棋方出现三次时，判断纯长将、
持续追逐同一目标的长捉责任，其余情况判和；连续 120 个半步无吃子判和。
长捉包含稳定棋子 ID、新增真捉集合、合法反吃及将帅/兵卒等例外。

这不是完整竞赛规则实现：混合“一将一捉”等复杂责任仍未补齐，不能宣称与所有平台裁决完全一致。
搜索也维护历史；置换表键包含规则上下文，避免不同历史下误用同一棋盘的缓存。

### 搜索与走法排序

- Negamax、Alpha-Beta 剪枝及 PVS 零窗口试探。
- 静态搜索延伸吃子变化；被将时搜索合法应将。
- 迭代加深与每步毫秒预算；超时返回上一层完整结果，必要时使用合法兜底着。
- 置换表保存深度、分数边界、最佳着和代际信息，支持将杀距离归一化。
- 排序结合哈希着、SEE/MVV-LVA、安全将军、双 Killer、Counter Move、
  Main History 与 Piece-to History。

搜索器使用统一 `Evaluator` 接口。未指定模型时使用手工子力与位置评估；
加载 NNUE 时改用神经网络评估，但规则和搜索流程保持一致。

### HalfKA NNUE

每个视角使用 `9 × 14 × 90 = 11340` 个稀疏输入特征：
观察方将帅九宫位置、棋子相对阵营与类型、棋子位置。包含双方将帅，黑方视角旋转 180°。

默认结构：

```text
HalfKA 11340 → 共享特征变换 256
                 ↓ 红黑双视角拼接
               512 → 隐藏层 32 → 输出 1
```

- `--accumulator-size` 配置每个视角的表达维度，默认 256。
- `--hidden-sizes` 配置隐藏层列表，默认 `32`；例如 `64`、`32 32`。
- 宽度支持 1～2048，隐藏层数量支持 1～8；输入定义及单分数输出固定。
- 特征变换与隐藏层使用 ClippedReLU，输出层为线性层。
- Python 训练与 C++ 推理共享编号和参数布局。
- 搜索中增量维护红黑累加器；将帅换桶时刷新对应视角，撤销时恢复快照。
- 默认架构使用兼容旧模型的 `XQNNUEF1`，自定义架构使用携带尺寸的 `XQNNUEF2`。
- C++ 按文件头构造网络，校验尺寸、文件长度与参数数值，再加载权重。

训练拟合 `score/600`，Huber 阈值同步归一化；导出时只把末层还原到引擎分值单位。
检查点记录结构与评分约定。续训继承原结构，改变尺寸需要另建任务，不会自动扩充旧权重。
详见 [数据与模型格式](training/DATA_FORMAT.md)。

## 构建与命令行使用

核心引擎不依赖第三方库，需要支持 C++20 的编译器。

```bash
make

# 展示初始局面并搜索
./build/make/xiangqi_cli --depth 4

# 显式加载 NNUE；路径需要指向已经导出的模型
./build/make/xiangqi_cli --nnue training/runs/champions/champion.nnue --depth 4
```

也可使用 CMake：

```bash
cmake -S . -B build
cmake --build build
```

上述 `build/make/` 路径对应 Make 构建。CMake 生成的程序使用相应构建目录中的路径。

创建 Python 环境并安装所需依赖：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r training/requirements.txt
.venv/bin/python -m pip install -r gui/requirements.txt
```

已有虚拟环境可以直接使用，无需重新创建。Windows 使用
`.venv\Scripts\python.exe` 代替 `.venv/bin/python`。

坐标为 `a0`～`i9`，红方底线为 0，黑方底线为 9；走法采用四字符 ICCS，例如 `b0c2`。
FEN 大写表示红棋、小写表示黑棋，`w` 为红方行棋，`b` 为黑方行棋。

## GUI 使用

### 截图、标定与模板

macOS 需要为实际启动程序的 Terminal、IDE 等开启屏幕录制与辅助功能权限，并重新启动该应用。
Windows 的控制程序与目标窗口应使用匹配的权限级别。

```bash
# 先列出窗口，再根据实际结果选择 index
.venv/bin/python -m gui.cli windows --query 微信

# 以下 index 1 仅为示例，不是固定窗口编号
.venv/bin/python -m gui.cli capture --query 微信 --index 1 --output gui/output/window.png
.venv/bin/python -m gui.cli calibrate --query 微信 --index 1 --orientation red-bottom

# 使用同主题的标准初始局面截图学习棋子模板
.venv/bin/python -m gui.cli learn-templates \
  --image gui/output/window.png --output gui/templates/jj_default.npz
```

标定按左上、右上、右下、左下依次选择棋盘最外侧交叉点。
黑方在屏幕下方时使用 `--orientation black-bottom`。窗口坐标采用归一化保存，
并换算 Retina / DPI 比例；更换棋盘主题或布局后应重新校验。

识别先检测金色棋子外圈，再区分红黑、匹配字形。在格点附近搜索实际中心，
容忍轻微上浮，并排除绿色光圈、白色提示圈。低置信度结果保留为未知格，不直接当作空位。

### 连续观察与下棋控制

```bash
.venv/bin/python -m gui.cli control \
  --live --query 微信 --index 1 \
  --side red --ai-side red \
  --nnue training/runs/champions/champion.nnue \
  --depth 8 --move-time-ms 500 \
  --stable-frames 4 --interval 0.1 --click-interval 0.25 \
  --confirmation-frames 200 --interactive-recovery
```

默认只预览，不发送真实点击。`--side` 是初始行棋方，`--ai-side` 是 AI 阵营，
屏幕朝向由标定配置决定；三者不可混为一谈。`--depth` 是搜索上限，
`--move-time-ms` 是每步时间预算，不表示一定搜索到该深度。

仅在自建棋盘或明确允许自动化的环境中，额外加入 `--execute` 开启真实点击。
第三方平台可能限制自动化，项目不绕过平台限制。

GUI 不会每帧重建引擎历史。Python 根据连续截图推断走法，
通过常驻协议把已验证的 `move` 交给 C++，由引擎持续维护 `GameHistory`。
协议提供 `position fen`、`state`、`result`、`legal`、`move`、`undo`、
`go depth N [movetime M]` 等接口；只有显式重设起始局面才清空历史。

控制器的主要保护机制：

- 连续帧产生候选棋盘，再用合法走法检验；视觉稳定不等于规则上可接受。
- 一格变化等动画中间状态继续等待，不写入历史。
- 部分未知格可结合跨帧信息与唯一合法走法恢复，但必须有足够变化证据。
- 对方快速应手时，支持“已知 AI 走法＋唯一合法应手”的双步追赶。
- 恢复候选需连续一致并满足时间门槛；下一次点击前仍等待可信棋盘。
- 必要时撤销动画中间应手并更正历史；证据不足不猜测。
- 点击发送不等于落子成功，必须通过后续视觉确认。
- 无法恢复、确认超时或执行异常时暂停，不自动重复点击。

`--interactive-recovery` 提供状态允许的人工继续操作；悔棋与重新开局同步未实现。
`recognize`、`observe` 和本地截图序列也可用于只读分析，完整参数见
`.venv/bin/python -m gui.cli --help` 及各子命令的 `--help`。

## 数据与训练

### 数据来源与组织

支持三种数据入口：

1. C++ 生成器：对 FEN 搜索标注，或通过自我对弈采样。
2. Pikafish 离线教师：通过 UCI 标注已有合法局面，过滤将杀与边界评分。
3. ModelScope [nowcan/xiangqi_train_data](https://modelscope.cn/datasets/nowcan/xiangqi_train_data)：
   只读解析 SQLite 棋谱，合法重放后抽取局面。

```text
training/data/modelscope-nowcan/
├── raw/train/、raw/val/              # 下载的原始压缩包
├── unpacked/train/、unpacked/val/    # 按来源分类的 SQLite 原库
└── processed/<数据集名称>/          # 固定样本、抽样计划与断点

training/runs/<任务名称>/             # 检查点、推理模型及运行报告
training/runs/champions/              # 稳定冠军路径与选择登记
```

ModelScope 抽样支持来源配额、每局采样上限、非法棋谱与极端评分过滤、
局面及镜像去重、SQLite 断点恢复，以及复用独立验证集。
原始评分须确认视角后转换；不同教师的分值单位不能未经校准就混合。
外部教师程序和数据集的获取、使用与分发应遵守各自许可证及使用约定。

训练 JSONL 的一行是一个完整局面，而不是一盘棋。最小记录：

```json
{"fen":"4k4/9/9/9/4p4/9/9/4R4/9/4K4 w","score":780}
```

`score` 为当前行棋方视角，正数表示有利。可选字段包括权重、对局身份、步数、阶段、
来源和赛果；未知赛果使用 `null`，不伪装成和棋。具体字段及目录约定见
[training/DATA_FORMAT.md](training/DATA_FORMAT.md)。

### 训练、续训与导出

以下示例读取已准备好的固定样本；数据和测试清单需先存在。
`my-nnue` 是示例任务目录，实际使用时应选择新的目录，避免覆盖既有实验。

```bash
.venv/bin/python -u -m training.train \
  training/data/modelscope-nowcan/processed/competition-1m-v1/train.jsonl \
  --validation-dataset training/data/modelscope-nowcan/processed/competition-1m-v1/validation.jsonl \
  --test-manifest training/data/modelscope-nowcan/processed/competition-1m-v1/test-manifest.json \
  --output-dir training/runs/my-nnue \
  --accumulator-size 256 --hidden-sizes 32 \
  --epochs 20 --batch-size 512 --score-scale 600 --device cpu
```

训练使用 AdamW 和带权重的 Smooth L1/Huber 损失，可选叠加已知赛果监督。
独立验证集选择最佳检查点，固定测试仅在训练结束后评估；
没有独立验证文件时，支持按对局分组划分，避免相邻局面随机跨区。

`best.pt` 保存验证最优模型，`latest.pt` 保存续训状态。
续训使用相同数据、结构与输出目录，增加 `--resume training/runs/my-nnue/latest.pt`；
`--epochs` 表示目标总轮数，而不是额外增加的轮数。跨尺度迁移会校验约定并处理优化器状态。

```bash
.venv/bin/python -m training.export_nnue \
  training/runs/my-nnue/best.pt training/runs/my-nnue/best.nnue
.venv/bin/python -m training.verify_cpp training/runs/my-nnue/best.pt \
  --engine build/make/xiangqi_cli --output-json training/runs/my-nnue/parity.json
```

ModelScope 基线与竞争流水线、Pikafish 标注流水线可自动串联相应阶段。
`training.iterate` 支持多代生成、历史数据回放、训练与恢复；
默认使用手工教师标注，使用冠军标注需要显式选择，避免无条件放大旧模型的错误。

训练、数据准备、续训或评测配置发生变化时，使用新的任务目录。
流水线任务锁防止并发写入；抽样与评测会拒绝覆盖不匹配或已存在的产物。
直接调用训练器前仍需确认输出目录，不能把示例目录反复当作新实验使用。
当前样本读取器会预编码数据到内存，大数据训练需要考虑内存开销。

## 对战评测与冠军管理

竞技场支持固定开局换先、多对手、同深度或同时间预算。
下面是独立评测示例，不会修改冠军登记：

```bash
.venv/bin/python -m training.arena \
  --candidate training/runs/my-nnue/best.nnue \
  --champion training/runs/champions/champion.nnue --no-hand \
  --search-mode time --depth 64 --time-limit-ms 100 \
  --min-games 100 --min-independent-pairs 50 \
  --output training/runs/my-nnue/arena.json
```

可加入手工对手及多个 `--history` 模型。竞技场保留完整走法、结束原因、输入指纹和预算，
分别统计候选和对手执红、执黑的胜/和/负、未完成局、胜率与得分率，
方便比较双方的同色表现。胜率为胜局除以完成局，得分率为（胜局 + 0.5 × 和局）
除以完成局；未完成局不计作和棋，也不进入这两个比例的分母。

自动晋升须满足验证、Python/C++ 推理一致性及对战门禁，而不是训练完成就晋升。
正式门禁至少 100 局、红黑安排数量相同。默认允许最多 10 局因步数上限未完成，
且不能超过总局数的 10%；默认至少 40 个不同开局的完整换先对。
晋升使用完整换先对的等权得分（重复开局先在开局内平均），默认达到 55% 即通过棋力门禁，
不要求红黑各自超过 50%。95% 区间保留作不确定性参考，不再要求其下界达到阈值；
这是开发用晋升标准，不代表已经统计证明棋力提升。

有现任冠军时，只有对冠军的棋力成绩决定晋升；手工和历史对手保留作诊断。
没有冠军时，所有配置对手仍需通过基线门禁。任何对手的协议故障都会阻止晋升。
`--required-score` 可调整配对得分门槛，`--max-unfinished 0` 可要求全部完成；
自动迭代对应 `--arena-required-score` 和 `--arena-max-unfinished`。
少量诊断对局不能替代正式门禁。旧报告保持原规则与原结论，不自动重算或替换冠军；
变更策略后使用新的实验目录。

共同冠军文件位于 `training/runs/champions/`。GUI 需显式指定 `--nnue` 才使用神经网络，
不会因为产生新模型就自动切换。手动选择冠军与自动晋升分别记录，不能改写旧评测结果。
旧自我迭代脚本使用自己的迭代状态，与共同冠军登记不是同一份状态。

## 功能更新摘要

以下记录持久的实现能力，不记录单次训练或比赛的过程：

- 规则层：合法走法、无损撤销、历史记录与简化循环裁决。
- 搜索层：PVS、迭代加深、限时搜索、规则上下文置换表和多级走法排序。
- NNUE：统一评估接口、搜索内增量累加器、可配置表达层与隐藏层、F1/F2 兼容。
- 训练：评分归一化、版本化检查点、断点续训、独立验证与固定测试保护。
- 数据：自我对弈生成、Pikafish 教师接入、ModelScope 棋谱导入与可恢复抽样。
- 评测：固定开局换先、多对手与时间/深度预算、红黑分项统计与配对得分晋升门禁。
- GUI：跨平台截图与标定、模板识别、双步追赶、动画过滤、视觉确认和安全暂停。

## 已知限制与后续方向

- 补齐复杂长将、长捉及“一将一捉”等完整循环责任。
- 在性能分析基础上改进搜索剪枝、时间分配、NNUE SIMD 和整数量化。
- 完善多进程数据生成、分片/流式读取及受控比例回放。
- 扩充高质量局面覆盖与独立战术评测，不能仅凭拟合误差宣称棋力提高。
- 完善多主题、多分辨率识别与标准引擎通信协议。
- 悔棋和重新开局同步暂缓；GUI 使用需要主题适配与人工监督。

README 仅在项目结构、代码功能、配置接口或使用方法变化时更新。
具体抽样进度、训练轮次、耗时、损失值、模型对战成绩和任务中断信息留在本地
`training/runs/` 与数据任务目录；其文件格式由技术文档说明，不逐次写进 README。
