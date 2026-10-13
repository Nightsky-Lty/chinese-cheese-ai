# ModelScope 数据、模型与日志格式

当前保留 `baseline-100k-v1` 与 `competition-1m-v1` 固定样本。网络默认仍为
HalfKA `11340 → 256×2 → 32 → 1` float32，现在可配置累加器宽度和多隐藏层。
已训练的 100k/1m 模型仍使用原默认结构，没有修改其权重。模型和大数据不提交 Git。

## 原始数据

`data/modelscope-nowcan/raw` 保存六个 `.7z` 原包，`unpacked` 按 train/val 和
depth11-mg、depth11-eg、depth16-eg 存放数据库。SQLite 表 `XQPGN` 的 `PGN` 字段
实际上保存 JSON 棋局，不是标准文本 PGN：

```json
{"start_fen":"... w","skip_step":10,"result":2,"mvs":[{"mv":43946,"score":0}]}
```

走法整数的低字节为起点、高字节为终点，使用 16×16 mailbox 坐标；
文件坐标 `x-3`，ICCS 行号 `12-y`。跳过 `skip_step` 内的随机零分前缀，
由 C++ 合法走法引擎重放，过滤绝对评分大于 3000 的样本。

## 训练样本 JSONL

每行一个落子后的局面，核心必需字段是 `fen` 和 `score`。示例：

```json
{"fen":"4k4/4a4/5a2b/9/9/2P6/8P/9/9/3K5 b","score":-8.0,"game_id":"nowcan-example","ply":7,"phase":"endgame","result":null,"source":"modelscope/nowcan/xiangqi_train_data","provenance":{"database":"train/depth11-eg","rowid":1896,"move":"e0d0","raw_score":8,"skip_step":3,"raw_result":2,"score_convention":"post-mover","score_multiplier":1.0}}
```

- `fen`：棋盘与当前行棋方；大写红方、小写黑方，`w` 红方、`b` 黑方。
- `score`：当前行棋方视角的原数据评分；原始评分以刚行棋一方为视角，因此取负号。
  JSONL 不预先除以 600，也不宣称其单位与新版 Pikafish cp 相同。
- `game_id`：整局及水平镜像的统一身份，用于去重和分区隔离。
- `ply`：从原始起点开始的半步数，不等同于真实整盘棋的开局步数。
- `phase`：按剩余子力粗分的阶段；高子力 `opening` 不代表确认来自真实开局。
- `result`：当前行棋方视角的胜 1、和 0、负 -1；原记录 2 表示未知，转为 null。
- `provenance`：数据库、记录、原始分数和转换规则等溯源信息。

`train.jsonl` 更新参数；`validation.jsonl` 选择最佳 epoch；`test.jsonl` 仅用于
最终测试。默认沿用百万实验的固定测试清单，后续采样不得把其局面送回训练。

## 配置与断点

- `score-audit.json`：此前用 Pikafish 抽查的评分视角证据；抽样时读取，不逐条重标注。
- `sampling-plan.json`：抽样参数与输入身份；`sampling.sqlite3`：逐局提交的抽样断点。
- `sampling-report.json`：数量、过滤统计、评分范围与导出文件 SHA-256。
- `test-manifest.json`：固定测试局面/对局身份及源文件指纹，不是样本本身。
- `splits/score-quarantine.json`：来源过滤配置，旧数据删除后 `sources` 为空。

训练在损失计算时使用 `score/600`，Huber 阈值同步换算；检查点保存评分尺度，
导出时末层权重和偏置还原到原始分数单位。

### 可配置网络的兼容约定

训练及 ModelScope 两条流水线支持 `--accumulator-size 256` 和
`--hidden-sizes 32`，省略时保持旧结构。隐藏层可写为 `64`、`32 32`、
`128 64 32`；宽度范围 1～2048，隐藏层数量 1～8，输入 HalfKA 11340 和输出 1 固定。

- 默认架构检查点继续为 `xiangqi-halfka-training-v2`，保留 `hidden_size: 32`。
- 非默认架构为 `xiangqi-halfka-training-v3`，记录 `accumulator_size` 和
  `hidden_sizes` 列表，名称包含全部尺寸；参数名称和形状严格核验。
- 默认架构 `.nnue` 继续为 F1（与原导出字节兼容）；扩展架构为 F2。
  F2 头部是 `<8sIIII`：`XQNNUEF2`、版本 2、11340、累加器宽度、隐藏层数量；
  后接各层宽度 `uint32`，再接特征偏置/权重、各隐藏层偏置/权重、输出偏置/权重。
  隐藏权重排列为 `[输出神经元][输入神经元]`，全部数值使用小端 float32。
- 加载检查点或 C++ 模型会自动选择对应结构。原 v1/v2 检查点及 F1 模型仍可加载；
  旧版未升级的 C++ 引擎不能加载 F2，需使用本次更新后的构建。
- 续训省略尺寸参数时继承检查点；显式指定不同结构则拒绝。
  加宽或加层需新实验目录、随机初始化，不会自动扩展旧权重。
- `config.json` 增加 `architecture` 和 `parameter_count`；一致性报告也记录架构。
  隐藏层激活诊断按全部隐藏神经元汇总；原数据与评分尺度不随网络尺寸变化。

现有百万样本可以直接用于架构对照，不必重新抽样。冠军仍由原门禁决定，
不能凭层数更深、维度更宽或训练完成就替换冠军。

## 模型与日志

- `best.pt`：最佳验证轮次的模型和尺度元数据；`latest.pt`：最新轮次及优化器状态，可续训。
- `best.nnue`：供 C++ 推理加载的 float32 权重。
- `metrics.jsonl`：每行一次 epoch 或最终测试事件，含损失、MAE、分位数、激活饱和率等。
- `*.log`：各阶段命令、标准输出和错误；不是训练样本。
- `report.json`、`parity.json`、`arena.json`：训练、推理一致性和对战的结构化结果。
- `competition-plan.json` 与 `competition-state.json`：完整实验计划与阶段/晋升状态。
- `champions/state.json`：当前冠军路径、指纹和选择来源；`origin` 区分人工选择与自动晋升。
  人工选择已有比赛时，在 `champions/promotions/` 另存新策略的重裁决证据，
  保留 `previous` 和原登记备份，不把旧 `arena.json` 的失败结果改成成功。

旧实验产物已删除，保留报告中的旧输入路径仅是历史证据，不应拿它们启动新任务。
已完成的 ModelScope 实验可读取或检查完成状态；改变参数时使用新的实验目录。

## 新门禁

`xiangqi-arena-v4` 有冠军时使用 `same-color-win-improvement-v1` 策略，
无冠军时使用 `color-balanced-paired-development-v1` 基线策略。
默认每个对手 100 局，红黑安排相等；最多容许 10 局因 `ply_cap` 未完成，
另有总局数 10% 上限，不计作和棋。默认至少 40 个不同开局的完整换先对。
实际完整配对下限为 `max(ceil(开局要求 × 0.8), 开局要求 − 有效未完成额度)`，
正式开局要求至少 50；`--max-unfinished 0` 要求至少 50 个完整配对。

每个对手的 `summary.by_color.red/black` 是候选的分色统计，
`opponent_by_color.red/black` 是对手的同色统计，均含胜/和/负、未完成、
安排局数、完成局数、`win_fraction`（胜局/完成局）和
`score_fraction`（胜局加半数和局/完成局）；无完成局时比例为 `null`。
总 `score_fraction` 保留原聚合口径，`color_balanced_score_fraction` 是候选红黑得分率均值。
它们都是诊断字段，不能代替门禁指标。

`win_improvement_by_color.red/black` 为候选胜率减对手的同色胜率，
直接使用本次比赛的完成局统计，不读取冠军旧比赛胜率。无可比完成局时为 `null`，不能通过。
有冠军时 `required_win_improvement` 默认为 `0.10`，即 **10 个百分点**，不是相对增长 10%。
红黑两项都达到该阈值才通过棋力门禁，不额外约束配对得分。
`gate_metric` 为 `same_color_win_fraction_difference`，`gate_score_fraction` 为 `null`。

`paired_score_fraction` 使用每个完整换先对的红黑得分均值，
重复开局先在开局内平均，再对不同开局等权平均，仅作有冠军时的诊断。
无冠军时 `gate_metric` 为 `paired_score_fraction`，`gate_score_fraction` 等于配对得分，
`required_win_improvement` 为 `null`，默认要求配对得分达到 55%。
`score_fraction_ci95` 仍报告配对开局的不确定性区间，但不参与门禁。
这是一项允许逐步改进的开发策略，不是统计显著性证明。
`rejection_reasons` 提供拒绝原因；`effective_max_unfinished` 记录绝对额度与 10% 限额的较小值。

`config.promotion_opponents` 记录晋升对象：存在 `champion` 时只要求对冠军的棋力门禁，
其他对手仅作棋力诊断；无冠军时要求所有配置对手通过。任何对手的协议错误仍阻止晋升。
训练验证与 Python/C++ 一致性要求保持不变。同色胜率阈值使用 `--required-win-improvement`，
自动迭代对应 `--arena-required-win-improvement`；`--required-score` / `--arena-required-score`
仅控制无冠军时的基线。未完成额度对应 `--arena-max-unfinished`。
旧 v2/v3 报告保留原结论，不会因为默认值更新就追溯晋升；新策略使用新实验目录。
