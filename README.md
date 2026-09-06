# Chinese Chess AI

当前版本提供一个零第三方依赖的 C++20 中国象棋基础规则引擎，包括：

- 90 格棋盘、棋子和走法表示
- 中国象棋 FEN 解析和输出
- 车、马、炮、兵、象、士、将的走法生成
- 马腿、象眼、炮架、九宫、河界和将帅照面
- 将军检测、伪合法走法与合法走法过滤
- 走法执行与无损撤销
- 增量 Zobrist 局面哈希
- 对局历史、连续悔棋、吃子/将军元数据、完整哈希时间线、未吃子计数和重复局面计数
- 三次相同局面自动裁决：普通重复判和，单方连续长将判负
- 连续 60 个整回合（120 个半回合）未吃子自动判和
- 可解释的手工评估：基础子力、中心控制、推进和过河兵奖励
- 固定深度 Negamax 与 Alpha-Beta 搜索、吃子排序和节点统计
- 静态搜索：延伸吃子序列，并在被将军时搜索全部合法应将
- 迭代加深：从深度 1 逐层搜索，并保留每一层的评分和节点统计
- 固定容量置换表：EXACT/LOWER/UPPER 边界、哈希走法和代际替换
- 将死/困毙状态判断
- Perft 规则树测试

历史裁决器按本项目采用的简化规则处理：相同局面（含行棋方）出现三次时，
单方连续长将者判负，其他重复判和；连续 120 个半回合未吃子判和。

## 构建与测试

当前项目同时支持 Make 和 CMake。macOS 自带的编译工具即可使用 Make：

```bash
make test
make run
```

运行四层 Alpha-Beta 搜索：

```bash
./build/make/xiangqi_cli --depth 4
```

使用无剪枝 Negamax 作为对照：

```bash
./build/make/xiangqi_cli --depth 3 --negamax
```

如果已经安装 CMake，也可以运行：

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

运行命令行演示：

```bash
./build/xiangqi_cli
./build/xiangqi_cli --fen "4k4/9/9/9/9/4R4/9/9/9/4K4 w"
```

VS Code 配置位于 `.vscode/`：默认构建任务为 `Build engine`，测试任务为
`Test engine`。clangd 等语言服务器可读取根目录的 `compile_flags.txt`。

坐标使用 `a0` 到 `i9`，红方底线为第 0 行。例如红方初始位置的马二进三编码为 `b0c2`。
