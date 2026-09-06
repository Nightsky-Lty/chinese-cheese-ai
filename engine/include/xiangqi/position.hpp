#pragma once

#include "xiangqi/types.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace xiangqi {

struct UndoInfo {
    Piece captured{kEmpty};
    std::array<int, 2> king_squares{kNoSquare, kNoSquare};
    std::uint64_t zobrist_hash{0};
};

enum class GameStatus {
    Ongoing,
    Checkmate,
    Stalemate,
};

class Position {
public:
    /// @brief 创建一个空棋盘，默认由红方行棋。
    Position();

    /// @brief 创建中国象棋标准初始局面。
    /// @return 包含标准开局棋子并由红方先行的局面。
    static Position initial();

    /// @brief 从中国象棋 FEN 字符串创建局面。
    /// @param fen 至少包含棋子布局和行棋方的 FEN；`w`/`r` 表示红方，`b` 表示黑方。
    /// @return 解析得到的有效局面。
    /// @throws std::invalid_argument FEN 格式错误或任一方不恰好拥有一个将帅时抛出。
    static Position from_fen(const std::string& fen);

    /// @brief 将当前局面序列化为中国象棋 FEN。
    /// @return 包含棋子布局和行棋方的 FEN 字符串。
    [[nodiscard]] std::string to_fen() const;

    /// @brief 生成人类可读的 ASCII 棋盘。
    /// @return 带坐标和当前行棋方的多行字符串。
    [[nodiscard]] std::string pretty() const;

    /// @brief 查询指定位置上的棋子。
    /// @param square 棋盘下标，范围为 0～89。
    /// @return 指定位置的棋子编码，空位置返回 `kEmpty`。
    /// @throws std::out_of_range 下标不在棋盘范围内时抛出。
    [[nodiscard]] Piece piece_at(int square) const;

    /// @brief 查询当前轮到哪一方行棋。
    /// @return 当前行棋方。
    [[nodiscard]] Color side_to_move() const { return side_to_move_; }

    /// @brief 查询当前局面的 Zobrist 哈希值。
    /// @return 同时编码棋子位置与行棋方的 64 位哈希。
    [[nodiscard]] std::uint64_t hash() const { return zobrist_hash_; }

    /// @brief 查询指定阵营将帅的位置。
    /// @param color 要查询的阵营。
    /// @return 将帅的棋盘下标；不存在时返回 `kNoSquare`。
    [[nodiscard]] int king_square(Color color) const;

    /// @brief 清空棋盘并恢复为红方行棋。
    void clear();

    /// @brief 在指定位置放置或移除棋子，并同步将帅位置缓存。
    /// @param square 棋盘下标，范围为 0～89。
    /// @param piece 要放置的棋子；传入 `kEmpty` 表示清空该位置。
    /// @throws std::out_of_range 下标不在棋盘范围内时抛出。
    void set_piece(int square, Piece piece);

    /// @brief 设置当前行棋方，主要用于构造测试局面或解析外部局面。
    /// @param color 新的当前行棋方。
    void set_side_to_move(Color color);

    /// @brief 执行一步棋并切换行棋方。
    /// @param move 要执行的内部走法。
    /// @return 撤销该走法所需的状态快照。
    /// @throws std::out_of_range 起点或终点越界时抛出。
    /// @throws std::invalid_argument 起点为空时抛出。
    /// @note 这是搜索使用的底层操作，不检查走法是否合法；调用者应传入已生成的走法。
    UndoInfo do_move(Move move);

    /// @brief 撤销最近执行的走法并恢复行棋方和将帅位置。
    /// @param move 之前传给 `do_move` 的同一步走法。
    /// @param undo 对应 `do_move` 返回的撤销信息。
    /// @throws std::out_of_range 起点或终点越界时抛出。
    /// @note `move` 与 `undo` 必须来自当前局面的上一层，否则恢复结果未定义。
    void undo_move(Move move, const UndoInfo& undo);

    /// @brief 生成当前行棋方所有符合棋子移动规则的伪合法走法。
    /// @return 伪合法走法列表；其中可能包含走后导致己方被将军的走法。
    [[nodiscard]] std::vector<Move> generate_pseudo_legal_moves() const;

    /// @brief 生成当前行棋方所有合法走法。
    /// @return 已排除走后己方被将军或将帅照面的走法列表。
    /// @note 函数会临时执行并撤销候选走法，因此不是 `const`。
    [[nodiscard]] std::vector<Move> generate_legal_moves();

    /// @brief 判断指定走法是否属于当前局面的合法走法。
    /// @param move 待检查的走法。
    /// @return 合法返回 `true`，否则返回 `false`。
    [[nodiscard]] bool is_legal_move(Move move);

    /// @brief 判断某阵营是否攻击指定位置。
    /// @param square 被攻击位置的棋盘下标。
    /// @param by_side 发起攻击的阵营。
    /// @return 至少有一个该方棋子攻击此位置时返回 `true`。
    /// @note 该函数包含马腿、象眼、炮架、兵的河界和将帅照面规则。
    [[nodiscard]] bool is_square_attacked(int square, Color by_side) const;

    /// @brief 判断指定阵营的将帅当前是否被攻击。
    /// @param color 要检查的阵营。
    /// @return 正在被将军或该方将帅不存在时返回 `true`。
    [[nodiscard]] bool in_check(Color color) const;

    /// @brief 判断当前局面的基础终局状态。
    /// @return 有合法走法时为 `Ongoing`；无合法走法时根据是否被将军返回对应状态。
    /// @note 中国象棋中无合法走法的一方判负，`Stalemate` 不应被上层当作和棋。
    [[nodiscard]] GameStatus status();

    /// @brief 检查局面是否恰好包含一个红帅和一个黑将。
    /// @return 将帅数量及缓存位置一致时返回 `true`。
    [[nodiscard]] bool is_valid() const;

    /// @brief 比较两个局面的棋盘、行棋方和将帅位置是否完全一致。
    friend bool operator==(const Position&, const Position&) = default;

private:
    std::array<Piece, kBoardSize> board_{};
    Color side_to_move_{Color::Red};
    std::array<int, 2> king_squares_{kNoSquare, kNoSquare};
    std::uint64_t zobrist_hash_{0};

    /// @brief 尝试向走法列表添加一个单目标走法。
    /// @param moves 接收生成走法的列表。
    /// @param from 起点棋盘下标。
    /// @param row 目标位置的内部行号。
    /// @param col 目标位置的内部列号。
    /// @param side 正在生成走法的阵营。
    /// @note 越界、己方棋子占据或直接吃将的目标不会被加入。
    void add_step_move(std::vector<Move>& moves, int from, int row, int col,
                       Color side) const;

    /// @brief 为指定位置的车生成伪合法走法。
    /// @param moves 接收生成走法的列表。
    /// @param from 车的起点棋盘下标。
    /// @param side 车所属阵营。
    void generate_rook_moves(std::vector<Move>& moves, int from, Color side) const;

    /// @brief 为指定位置的炮生成包含炮架规则的伪合法走法。
    /// @param moves 接收生成走法的列表。
    /// @param from 炮的起点棋盘下标。
    /// @param side 炮所属阵营。
    void generate_cannon_moves(std::vector<Move>& moves, int from, Color side) const;

    /// @brief 为指定位置的马生成包含马腿规则的伪合法走法。
    /// @param moves 接收生成走法的列表。
    /// @param from 马的起点棋盘下标。
    /// @param side 马所属阵营。
    void generate_horse_moves(std::vector<Move>& moves, int from, Color side) const;

    /// @brief 为指定位置的相/象生成包含象眼和河界规则的伪合法走法。
    /// @param moves 接收生成走法的列表。
    /// @param from 相/象的起点棋盘下标。
    /// @param side 相/象所属阵营。
    void generate_elephant_moves(std::vector<Move>& moves, int from, Color side) const;

    /// @brief 为指定位置的仕/士生成九宫内的伪合法走法。
    /// @param moves 接收生成走法的列表。
    /// @param from 仕/士的起点棋盘下标。
    /// @param side 仕/士所属阵营。
    void generate_advisor_moves(std::vector<Move>& moves, int from, Color side) const;

    /// @brief 为指定位置的帅/将生成九宫内的伪合法走法。
    /// @param moves 接收生成走法的列表。
    /// @param from 帅/将的起点棋盘下标。
    /// @param side 帅/将所属阵营。
    void generate_king_moves(std::vector<Move>& moves, int from, Color side) const;

    /// @brief 为指定位置的兵/卒生成包含过河规则的伪合法走法。
    /// @param moves 接收生成走法的列表。
    /// @param from 兵/卒的起点棋盘下标。
    /// @param side 兵/卒所属阵营。
    void generate_pawn_moves(std::vector<Move>& moves, int from, Color side) const;
};

/// @brief 统计指定深度下的合法走法树叶子节点数，用于验证规则实现。
/// @param position 起始局面；计算结束后会恢复到原状态。
/// @param depth 搜索半回合深度，必须大于或等于 0。
/// @return 深度为 0 时返回 1，否则返回所有合法分支的叶子节点总数。
/// @throws std::invalid_argument 深度小于 0 时抛出。
[[nodiscard]] std::uint64_t perft(Position& position, int depth);

}  // namespace xiangqi
