#pragma once

#include "xiangqi/game_history.hpp"
#include "xiangqi/move_ordering.hpp"
#include "xiangqi/transposition_table.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace xiangqi {

constexpr int kMateScore = 30000;
constexpr int kMateThreshold = 29000;
constexpr int kSearchInfinity = 32000;

/// @brief 选择用于固定深度搜索的算法。
enum class SearchAlgorithm {
    Negamax,   ///< 不剪枝的完整 Negamax，用作正确性基准。
    AlphaBeta, ///< 使用 Alpha-Beta 窗口剪枝的 Negamax。
};

/// @brief 一次搜索的限制参数。
struct SearchLimits {
    int depth{4}; ///< 迭代加深要完成的最大深度。
    SearchAlgorithm algorithm{SearchAlgorithm::AlphaBeta};
    int quiescence_depth{32}; ///< 静态搜索最大延伸半回合数；设为 0 可关闭延伸。
    bool use_transposition_table{true}; ///< 是否在 Alpha-Beta 中使用置换表。
};

/// @brief 迭代加深中某一个完整深度的搜索结果。
struct IterationResult {
    int depth{0};                    ///< 本轮完成的固定搜索深度。
    std::optional<Move> best_move{}; ///< 本轮找到的根节点最佳走法。
    int score{0};                    ///< 当前行棋方视角的本轮评分。
    std::uint64_t nodes{0};          ///< 本轮新增的总节点数量。
    std::uint64_t beta_cutoffs{0};   ///< 本轮新增的 Beta 截断次数。
    std::uint64_t quiescence_nodes{0}; ///< 本轮进入静态搜索的节点数量。
    std::uint64_t tt_hits{0};          ///< 本轮完整键匹配的置换表查询次数。
    std::uint64_t tt_cutoffs{0};       ///< 本轮直接复用缓存边界返回的次数。
    std::uint64_t tt_move_orderings{0}; ///< 本轮使用缓存最佳着进行排序的次数。
};

/// @brief 一次搜索返回的最佳走法、评分和统计信息。
struct SearchResult {
    std::optional<Move> best_move{}; ///< 根节点最佳走法；终局或深度为 0 时为空。
    int score{0};                    ///< 当前行棋方视角的搜索分数。
    int depth{0};                    ///< 最深的完整迭代深度。
    std::uint64_t nodes{0};          ///< 访问的搜索节点数量，包含根节点。
    std::uint64_t beta_cutoffs{0};   ///< Alpha-Beta 发生 Beta 截断的次数。
    std::uint64_t quiescence_nodes{0}; ///< 进入静态搜索的节点数量。
    std::uint64_t tt_hits{0};          ///< 完整键匹配的置换表查询总次数。
    std::uint64_t tt_cutoffs{0};       ///< 直接复用缓存边界返回的总次数。
    std::uint64_t tt_move_orderings{0}; ///< 使用缓存最佳着进行排序的总次数。
    std::vector<IterationResult> iterations{}; ///< 按深度递增排列的完整迭代结果。
};

/// @brief 提供迭代加深的 Negamax 与 Alpha-Beta 搜索。
class Searcher {
public:
    /// @brief 创建搜索器及其持久化置换表。
    /// @param transposition_table_mb 置换表目标容量，单位为 MB。
    explicit Searcher(std::size_t transposition_table_mb = 16);

    /// @brief 清空搜索器持有的全部置换表缓存。
    void clear_transposition_table();

    /// @brief 从深度 1 迭代加深到目标深度并搜索最佳走法。
    /// @param position 待搜索局面；返回前会被完整恢复。
    /// @param limits 搜索深度和算法选择；深度必须大于或等于 0。
    /// @return 当前行棋方视角的搜索结果。
    /// @throws std::invalid_argument 深度小于 0 时抛出。
    [[nodiscard]] SearchResult search(Position& position, const SearchLimits& limits);

    /// @brief 从一盘包含真实历史的对局开始搜索最佳走法。
    /// @param history 待搜索对局；搜索会临时追加分支走法，返回前完整恢复。
    /// @param limits 搜索深度和算法选择；深度必须大于或等于 0。
    /// @return 当前行棋方视角的搜索结果；已经裁决的对局不返回最佳走法。
    /// @throws std::invalid_argument 深度或静态搜索深度小于 0 时抛出。
    [[nodiscard]] SearchResult search(GameHistory& history, const SearchLimits& limits);

private:
    std::uint64_t nodes_{0};
    std::uint64_t beta_cutoffs_{0};
    std::uint64_t quiescence_nodes_{0};
    std::uint64_t tt_hits_{0};
    std::uint64_t tt_cutoffs_{0};
    std::uint64_t tt_move_orderings_{0};
    int quiescence_depth_{32};
    bool use_transposition_table_{true};
    TranspositionTable transposition_table_;
    MoveOrdering move_ordering_;
    std::optional<int> cached_quiescence_depth_{};

    /// @brief 执行迭代加深中的一个固定深度根节点搜索。
    /// @param history 待搜索对局；返回前会被完整恢复。
    /// @param depth 本轮固定搜索深度。
    /// @param algorithm 本轮使用的搜索算法。
    /// @param previous_best 上一轮最佳走法，用作本轮根节点排序提示。
    /// @return 仅包含本轮最佳走法、评分与深度的临时结果。
    SearchResult search_iteration(GameHistory& history, int depth,
                                  SearchAlgorithm algorithm,
                                  std::optional<Move> previous_best);

    /// @brief 递归执行无剪枝 Negamax 搜索。
    /// @param history 当前节点的局面与完整搜索路径。
    /// @param depth 当前节点剩余搜索深度。
    /// @param ply 当前节点距离根节点的半回合数，用于偏好更快将死。
    /// @return 当前行棋方视角的节点分数。
    int negamax(GameHistory& history, int depth, int ply);

    /// @brief 递归执行带 Alpha-Beta 剪枝的 Negamax 搜索。
    /// @param history 当前节点的局面与完整搜索路径。
    /// @param depth 当前节点剩余搜索深度。
    /// @param alpha 当前行棋方已经保证可以得到的分数下界。
    /// @param beta 对手允许当前行棋方得到的分数上界。
    /// @param ply 当前节点距离根节点的半回合数，用于偏好更快将死。
    /// @return 当前行棋方视角的节点分数。
    int alpha_beta(GameHistory& history, int depth, int alpha, int beta, int ply);

    /// @brief 延伸搜索叶子节点中的吃子序列与被将军时的全部应将。
    /// @param history 当前静态搜索节点的局面与完整搜索路径。
    /// @param alpha 当前行棋方已经保证可以得到的分数下界。
    /// @param beta 对手允许当前行棋方得到的分数上界。
    /// @param ply 当前节点距离根节点的半回合数。
    /// @param qply 当前节点在静态搜索内已经延伸的半回合数。
    /// @return 当前行棋方视角的稳定局面分数或 Alpha-Beta 边界分数。
    int quiescence(GameHistory& history, int alpha, int beta, int ply, int qply);

    /// @brief 将当前历史裁决转换为当前行棋方视角的搜索分数。
    /// @param history 当前节点的完整历史。
    /// @param ply 当前节点距离根节点的半回合数。
    /// @return 已结束时返回胜负或和棋分数；对局继续时返回空。
    static std::optional<int> adjudication_score(const GameHistory& history, int ply);

    /// @brief 生成同时包含棋盘和规则路径的置换表键。
    /// @param history 当前节点的完整历史。
    /// @return 可隔离不同重复和未吃子上下文的缓存键。
    static std::uint64_t transposition_key(const GameHistory& history);

    /// @brief 将相对当前节点的将死分数转换为与层数无关的缓存分数。
    /// @param score 搜索返回的分数。
    /// @param ply 当前节点距离根节点的半回合数。
    /// @return 适合写入置换表的分数。
    static int score_to_tt(int score, int ply);

    /// @brief 将缓存中的将死分数恢复为相对当前根节点的分数。
    /// @param score 置换表保存的分数。
    /// @param ply 当前节点距离根节点的半回合数。
    /// @return 当前搜索可直接使用的分数。
    static int score_from_tt(int score, int ply);

};

}  // namespace xiangqi
