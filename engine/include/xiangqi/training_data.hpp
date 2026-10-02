#pragma once

#include "xiangqi/game_history.hpp"
#include "xiangqi/search.hpp"

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <optional>
#include <random>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace xiangqi {

/// @brief 按剩余子力粗略划分训练局面所处的阶段。
enum class GamePhase {
    Opening,
    Middlegame,
    Endgame,
};

/// @brief 将局面阶段转换为写入数据集的稳定英文名称。
/// @param phase 待转换的局面阶段。
/// @return `opening`、`middlegame` 或 `endgame`。
[[nodiscard]] std::string_view game_phase_name(GamePhase phase);

/// @brief 根据双方剩余非将帅子力判断局面阶段。
/// @param position 待分类的合法象棋局面。
/// @return 子力不少于初始值 80% 时为开局，不少于 35% 时为中局，否则为残局。
[[nodiscard]] GamePhase classify_game_phase(const Position& position);

/// @brief FEN 批量标注模式的配置。
struct LabelPositionsConfig {
    int depth{6};                    ///< 教师搜索的固定深度。
    int quiescence_depth{32};        ///< 静态搜索最大延伸半回合数。
    int maximum_absolute_score{28000}; ///< 保留标签的最大绝对值，避开将杀分数。
    double weight{1.0};              ///< 写入每条样本的训练权重。
    bool deduplicate{true};          ///< 是否按包含行棋方的局面哈希去重。
};

/// @brief 自我对弈数据生成模式的配置。
struct SelfPlayConfig {
    std::size_t games{1};             ///< 要生成的对局数量。
    int play_depth{4};                ///< 选择实际走法使用的搜索深度。
    int label_depth{6};               ///< 生成监督分数使用的搜索深度。
    int quiescence_depth{32};         ///< 静态搜索最大延伸半回合数。
    std::size_t maximum_plies{240};   ///< 单局最多执行的半回合数。
    std::size_t sample_start_ply{8};  ///< 从第几个半回合边界开始采样。
    std::size_t minimum_sample_gap{2}; ///< 两次采样之间的最小半回合数。
    std::size_t maximum_sample_gap{4}; ///< 两次采样之间的最大半回合数。
    std::size_t random_opening_plies{10}; ///< 前多少半回合允许从近似最佳着中随机选择。
    std::size_t random_top_k{3};       ///< 随机候选最多取评分最高的多少步。
    int random_score_margin{150};      ///< 随机候选与最佳着允许相差的最大分数。
    int maximum_absolute_score{28000}; ///< 保留标签的最大绝对值。
    double weight{1.0};                ///< 写入每条样本的训练权重。
    std::uint64_t seed{1};             ///< 控制采样和开局选择的随机种子。
    bool deduplicate{true};            ///< 是否跨全部对局按局面哈希去重。
};

/// @brief 一次数据生成任务的统计结果。
struct TrainingDataStats {
    std::size_t games{0};              ///< 实际处理或完成的对局数量。
    std::size_t input_positions{0};    ///< FEN 标注模式读取的有效输入行数。
    std::size_t samples_written{0};    ///< 成功写出的训练样本数量。
    std::size_t duplicate_positions{0}; ///< 因重复而跳过的局面数量。
    std::size_t terminal_positions{0}; ///< 因已经终局而跳过的局面数量。
    std::size_t extreme_scores{0};     ///< 因进入将杀分数区间而跳过的标签数量。
};

/// @brief 使用现有搜索器生成 NNUE 监督训练数据。
///
/// 输出采用 Python `JsonlPositionDataset` 可直接读取的 JSON Lines 格式。`score`
/// 始终来自当前行棋方视角；自我对弈附带的 `result` 也采用样本行棋方视角。
class TrainingDataGenerator {
public:
    /// @brief 创建使用指定教师评估器的数据生成器。
    /// @param evaluator 搜索叶子节点使用的评估器；它必须比生成器存活更久。
    /// @param transposition_table_mb 每个内部搜索器的置换表容量，单位为 MB。
    explicit TrainingDataGenerator(
        const Evaluator& evaluator = handcrafted_evaluator(),
        std::size_t transposition_table_mb = 16);

    /// @brief 逐行读取纯文本 FEN，用固定深度搜索分数生成 JSONL 标签。
    /// @param input 输入流；忽略空行及以 `#` 开头的注释行。
    /// @param output 接收 JSON Lines 样本的输出流。
    /// @param config 深度、分数过滤、权重和去重配置。
    /// @return 本次标注的输入、输出和过滤统计。
    /// @throws std::invalid_argument 配置或某一行 FEN 非法时抛出。
    TrainingDataStats label_positions(
        std::istream& input, std::ostream& output,
        const LabelPositionsConfig& config = {});

    /// @brief 从标准初始局面进行若干盘自我对弈并生成完整阶段样本。
    /// @param output 接收 JSON Lines 样本的输出流。
    /// @param config 对局数、搜索深度、采样间隔和随机开局配置。
    /// @return 本次对弈的局数、输出和过滤统计。
    /// @throws std::invalid_argument 配置非法时抛出。
    TrainingDataStats self_play(
        std::ostream& output, const SelfPlayConfig& config = {});

private:
    struct PendingSample;
    struct ScoredMove;

    Searcher play_searcher_;
    Searcher label_searcher_;
    std::mt19937_64 random_;
    std::unordered_set<std::uint64_t> seen_positions_;

    /// @brief 从不比最佳着差太多的前 K 个候选中随机选择开局着法。
    /// @param history 当前完整对局历史；返回前会恢复原状。
    /// @param config 自我对弈搜索和随机候选配置。
    /// @return 有合法走法时返回选中的走法，否则返回空。
    std::optional<Move> choose_diverse_move(
        GameHistory& history, const SelfPlayConfig& config);
};

}  // namespace xiangqi
