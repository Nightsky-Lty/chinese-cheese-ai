#include "xiangqi/cycle_adjudicator.hpp"

#include <algorithm>
#include <utility>

namespace xiangqi {

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

    const auto& entries = history.entries();
    for (std::size_t ply = window->begin_ply(); ply < window->end_ply(); ++ply) {
        const HistoryEntry& entry = entries[ply];
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
