#include "xiangqi/cycle_adjudicator.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <utility>

namespace xiangqi {
namespace {

using PieceId = int;
using ChaseMask = std::uint64_t;

constexpr PieceId kNoPieceId = -1;

/// @brief 判断兵卒是否已经越过楚河汉界。
/// @param color 兵卒所属阵营。
/// @param square 兵卒当前所在位置。
/// @return 已过河返回 `true`。
bool pawn_crossed_river(Color color, int square) {
    const int row = square_row(square);
    return color == Color::Red ? row <= 4 : row >= 5;
}

/// @brief 判断是否属于无须考虑目标保护的“弱子捉强子”。
/// @param attacker 发起攻击的棋子类型。
/// @param target 被攻击的棋子类型。
/// @return 按当前简化棋规应直接视为真捉时返回 `true`。
bool is_weak_piece_chasing_stronger_piece(PieceType attacker, PieceType target) {
    if ((attacker == PieceType::Horse || attacker == PieceType::Cannon) &&
        target == PieceType::Rook) {
        return true;
    }
    if (attacker == PieceType::Advisor || attacker == PieceType::Elephant) {
        return target == PieceType::Rook || target == PieceType::Cannon ||
               target == PieceType::Horse;
    }
    return false;
}

/// @brief 判断目标是否属于可被长捉裁决追踪的棋子。
/// @param target 目标棋子。
/// @param target_square 目标所在位置。
/// @return 将帅和未过河兵卒返回 `false`，其余敌方棋子返回 `true`。
bool is_chase_target(Piece target, int target_square) {
    const PieceType type = piece_type(target);
    if (type == PieceType::King || type == PieceType::Empty) {
        return false;
    }
    return type != PieceType::Pawn ||
           pawn_crossed_river(piece_color(target), target_square);
}

/// @brief 判断目标棋子能否从原位置合法反向吃掉攻击者。
/// @param position 攻击发生前的局面。
/// @param attack 攻击者可能吃目标的走法。
/// @return 同类棋子间存在合法对称互吃时返回 `true`。
bool target_can_capture_attacker(const Position& position, Move attack) {
    Position reverse = position;
    reverse.set_side_to_move(opposite(position.side_to_move()));
    const Move counter{
        .from = attack.to,
        .to = attack.from,
    };
    return reverse.is_legal_move(counter);
}

/// @brief 判断攻击者吃掉目标后，目标方能否合法吃回攻击者。
/// @param position 攻击发生前且轮到攻击方行棋的局面。
/// @param attack 攻击者吃目标的合法走法。
/// @return 至少存在一个合法反吃时返回 `true`。
bool target_is_legally_protected(const Position& position, Move attack) {
    Position after_capture = position;
    static_cast<void>(after_capture.do_move(attack));
    const std::vector<Move> replies = after_capture.generate_legal_moves();
    return std::any_of(replies.begin(), replies.end(), [attack](Move reply) {
        return reply.to == attack.to;
    });
}

/// @brief 计算指定一方当前正在“真捉”的敌方棋子 ID 集合。
/// @param source 待分析局面；函数内部会复制，不修改调用者状态。
/// @param attacker 发起捉子的阵营。
/// @param piece_ids 当前棋盘位置到稳定棋子 ID 的映射。
/// @return 位掩码；第 N 位表示 ID 为 N 的棋子正受到有效捉子威胁。
ChaseMask chased_targets(const Position& source, Color attacker,
                          const std::array<PieceId, kBoardSize>& piece_ids) {
    Position position = source;
    position.set_side_to_move(attacker);
    const std::vector<Move> moves = position.generate_legal_moves();
    ChaseMask chased = 0;

    for (Move move : moves) {
        const Piece attacking_piece = position.piece_at(move.from);
        const Piece target_piece = position.piece_at(move.to);
        if (is_empty(target_piece) || piece_color(target_piece) == attacker) {
            continue;
        }

        const PieceType attacker_type = piece_type(attacking_piece);
        const PieceType target_type = piece_type(target_piece);

        // 按 Pikafish 风格的简化规则，将帅和兵卒可以连续追逐，不记作违规长捉。
        if (attacker_type == PieceType::King || attacker_type == PieceType::Pawn ||
            !is_chase_target(target_piece, move.to)) {
            continue;
        }

        bool true_chase =
            is_weak_piece_chasing_stronger_piece(attacker_type, target_type);
        if (!true_chase) {
            true_chase = !target_is_legally_protected(position, move);
        }

        // 同类棋子能够合法互吃时属于对称攻击，不视为单方面的捉。
        if (true_chase && attacker_type == target_type &&
            target_can_capture_attacker(position, move)) {
            true_chase = false;
        }

        const PieceId target_id = piece_ids[move.to];
        if (true_chase && target_id >= 0 && target_id < 64) {
            chased |= ChaseMask{1} << static_cast<unsigned int>(target_id);
        }
    }

    return chased;
}

/// @brief 在重复区间内判断双方是否持续新捉同一枚棋子。
/// @param history 完整对局历史。
/// @param window 已确认的重复区间。
/// @return 红、黑双方的长捉目标掩码；只有一方非零时该方承担长捉责任。
std::array<ChaseMask, 2> detect_perpetual_chases(
    const GameHistory& history, const RepetitionWindow& window) {
    Position rollback = history.position();
    std::array<PieceId, kBoardSize> piece_ids{};
    piece_ids.fill(kNoPieceId);

    PieceId next_id = 0;
    for (int square = 0; square < kBoardSize; ++square) {
        if (!is_empty(rollback.piece_at(square))) {
            piece_ids[static_cast<std::size_t>(square)] = next_id++;
        }
    }

    std::array<ChaseMask, 2> common_targets{
        ~ChaseMask{0},
        ~ChaseMask{0},
    };
    std::array<bool, 2> moved{false, false};
    const auto& entries = history.entries();

    for (std::size_t ply = window.end_ply(); ply-- > window.begin_ply();) {
        const HistoryEntry& entry = entries[ply];
        const std::size_t side = static_cast<std::size_t>(entry.mover);
        moved[side] = true;

        // 相同局面之间不可能真实减少子力；若出现吃子，只能是哈希碰撞或损坏历史。
        if (!is_empty(entry.captured)) {
            return {0, 0};
        }

        const ChaseMask after = chased_targets(rollback, entry.mover, piece_ids);

        const int from = static_cast<int>(entry.move.from);
        const int to = static_cast<int>(entry.move.to);
        piece_ids[static_cast<std::size_t>(from)] =
            piece_ids[static_cast<std::size_t>(to)];
        piece_ids[static_cast<std::size_t>(to)] = kNoPieceId;
        rollback.undo_move(entry.move, entry.undo);

        const ChaseMask before = chased_targets(rollback, entry.mover, piece_ids);
        const ChaseMask newly_chased = after & ~before;
        common_targets[side] &= newly_chased;
    }

    for (std::size_t side = 0; side < moved.size(); ++side) {
        if (!moved[side]) {
            common_targets[side] = 0;
        }
    }
    return common_targets;
}

}  // namespace

std::optional<RepetitionWindow> find_repetition_window(
    const GameHistory& history, std::size_t required_occurrences) {
    if (required_occurrences < 2) {
        return std::nullopt;
    }

    const auto& hashes = history.position_hashes();
    const std::uint64_t current_hash = hashes.back();
    std::vector<std::size_t> occurrences;
    occurrences.reserve(required_occurrences);

    // 倒序只收集最近的所需次数，避免扫描后再保留无关的早期重复。
    for (std::size_t ply = hashes.size(); ply-- > 0;) {
        if (hashes[ply] == current_hash) {
            occurrences.push_back(ply);
            if (occurrences.size() == required_occurrences) {
                std::reverse(occurrences.begin(), occurrences.end());
                return RepetitionWindow{.occurrence_plies = std::move(occurrences)};
            }
        }
    }

    return std::nullopt;
}

RepetitionAdjudication adjudicate_repetition(
    const GameHistory& history, std::size_t required_occurrences) {
    const std::optional<RepetitionWindow> window =
        find_repetition_window(history, required_occurrences);
    if (!window) {
        return {};
    }

    bool red_moved = false;
    bool black_moved = false;
    bool red_always_checked = true;
    bool black_always_checked = true;
    bool any_check = false;

    const auto& entries = history.entries();
    for (std::size_t ply = window->begin_ply(); ply < window->end_ply(); ++ply) {
        const HistoryEntry& entry = entries[ply];
        any_check = any_check || entry.gave_check;
        if (entry.mover == Color::Red) {
            red_moved = true;
            red_always_checked = red_always_checked && entry.gave_check;
        } else {
            black_moved = true;
            black_always_checked = black_always_checked && entry.gave_check;
        }
    }

    red_always_checked = red_moved && red_always_checked;
    black_always_checked = black_moved && black_always_checked;

    RepetitionVerdict verdict = RepetitionVerdict::Draw;
    if (red_always_checked && !black_always_checked) {
        verdict = RepetitionVerdict::RedPerpetualCheck;
    } else if (black_always_checked && !red_always_checked) {
        verdict = RepetitionVerdict::BlackPerpetualCheck;
    } else if (!any_check) {
        const auto chase = detect_perpetual_chases(history, *window);
        const bool red_chases = chase[static_cast<std::size_t>(Color::Red)] != 0;
        const bool black_chases = chase[static_cast<std::size_t>(Color::Black)] != 0;
        if (red_chases && !black_chases) {
            verdict = RepetitionVerdict::RedPerpetualChase;
        } else if (black_chases && !red_chases) {
            verdict = RepetitionVerdict::BlackPerpetualChase;
        }
    }

    return RepetitionAdjudication{
        .verdict = verdict,
        .window = window,
    };
}

HistoryVerdict adjudicate_history(const GameHistory& history) {
    const RepetitionAdjudication repetition = adjudicate_repetition(history);
    switch (repetition.verdict) {
        case RepetitionVerdict::RedPerpetualCheck:
            return HistoryVerdict::BlackWinsByRedPerpetualCheck;
        case RepetitionVerdict::BlackPerpetualCheck:
            return HistoryVerdict::RedWinsByBlackPerpetualCheck;
        case RepetitionVerdict::RedPerpetualChase:
            return HistoryVerdict::BlackWinsByRedPerpetualChase;
        case RepetitionVerdict::BlackPerpetualChase:
            return HistoryVerdict::RedWinsByBlackPerpetualChase;
        case RepetitionVerdict::Draw:
            return HistoryVerdict::DrawByThreefoldRepetition;
        case RepetitionVerdict::None:
            break;
    }

    if (history.no_capture_plies() >= kNoCaptureDrawPlies) {
        return HistoryVerdict::DrawByNoCapture;
    }
    return HistoryVerdict::Ongoing;
}

}  // namespace xiangqi
