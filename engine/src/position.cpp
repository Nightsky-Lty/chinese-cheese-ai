#include "xiangqi/position.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace xiangqi {
namespace {

/// 将阵营枚举转换为数组下标：红方为 0，黑方为 1。
/// @param color 要转换的阵营。
/// @return 可用于双方状态数组的下标。
constexpr std::size_t color_index(Color color) {
    return static_cast<std::size_t>(color);
}

/// 判断指定坐标是否位于对应阵营的九宫内。
/// @param color 九宫所属阵营。
/// @param row 内部行号。
/// @param col 内部列号。
/// @return 位于九宫内返回 `true`。
bool inside_palace(Color color, int row, int col) {
    if (col < 3 || col > 5) {
        return false;
    }
    return color == Color::Red ? row >= 7 && row <= 9 : row >= 0 && row <= 2;
}

/// 判断指定行是否位于某阵营自己的半边棋盘。
/// @param color 棋子所属阵营。
/// @param row 内部行号。
/// @return 尚未越过楚河汉界返回 `true`。
bool on_own_side_of_river(Color color, int row) {
    return color == Color::Red ? row >= 5 : row <= 4;
}

/// 判断指定阵营的棋子在该行是否已经过河。
/// @param color 棋子所属阵营。
/// @param row 内部行号。
/// @return 已经过河返回 `true`。
bool crossed_river(Color color, int row) {
    return !on_own_side_of_river(color, row);
}

/// 将内部棋子编码转换为 FEN 字符。
/// @param piece 要转换的棋子编码。
/// @return 红方为大写、黑方为小写的 FEN 字符；空位置返回 `'1'`。
char piece_to_fen(Piece piece) {
    char letter = '?';
    switch (piece_type(piece)) {
        case PieceType::King: letter = 'k'; break;
        case PieceType::Advisor: letter = 'a'; break;
        case PieceType::Elephant: letter = 'b'; break;
        case PieceType::Horse: letter = 'n'; break;
        case PieceType::Rook: letter = 'r'; break;
        case PieceType::Cannon: letter = 'c'; break;
        case PieceType::Pawn: letter = 'p'; break;
        case PieceType::Empty: return '1';
    }
    return piece_color(piece) == Color::Red
               ? static_cast<char>(std::toupper(static_cast<unsigned char>(letter)))
               : letter;
}

/// 将 FEN 棋子字符转换为内部棋子编码。
/// @param letter FEN 棋子字符，支持马的 `n/h` 和象的 `b/e` 两种写法。
/// @return 对应的内部棋子编码。
/// @throws std::invalid_argument 字符不是受支持的棋子时抛出。
Piece fen_to_piece(char letter) {
    const Color color = std::isupper(static_cast<unsigned char>(letter))
                            ? Color::Red
                            : Color::Black;
    switch (static_cast<char>(std::tolower(static_cast<unsigned char>(letter)))) {
        case 'k': return make_piece(color, PieceType::King);
        case 'a': return make_piece(color, PieceType::Advisor);
        case 'b':
        case 'e': return make_piece(color, PieceType::Elephant);
        case 'n':
        case 'h': return make_piece(color, PieceType::Horse);
        case 'r': return make_piece(color, PieceType::Rook);
        case 'c': return make_piece(color, PieceType::Cannon);
        case 'p': return make_piece(color, PieceType::Pawn);
        default: throw std::invalid_argument("unknown FEN piece");
    }
}

/// 判断两个位置是否处于同一横线或竖线上。
/// @param from 第一个棋盘下标。
/// @param target 第二个棋盘下标。
/// @return 同行或同列返回 `true`。
bool same_line(int from, int target) {
    return square_row(from) == square_row(target) || square_col(from) == square_col(target);
}

/// 统计两个同行或同列位置之间的阻挡棋子数量，不包含端点。
/// @param board 当前棋盘数组。
/// @param from 起始位置。
/// @param target 目标位置。
/// @return 阻挡棋子数；端点相同或不在同一直线上时返回 -1。
int blockers_between(const std::array<Piece, kBoardSize>& board, int from, int target) {
    if (!same_line(from, target) || from == target) {
        return -1;
    }

    const int from_row = square_row(from);
    const int from_col = square_col(from);
    const int target_row = square_row(target);
    const int target_col = square_col(target);
    const int dr = (target_row > from_row) - (target_row < from_row);
    const int dc = (target_col > from_col) - (target_col < from_col);

    int row = from_row + dr;
    int col = from_col + dc;
    int blockers = 0;
    while (row != target_row || col != target_col) {
        if (!is_empty(board[static_cast<std::size_t>(make_square(row, col))])) {
            ++blockers;
        }
        row += dr;
        col += dc;
    }
    return blockers;
}

/// 使用 SplitMix64 的混合步骤为 Zobrist 特征生成稳定的伪随机键。
/// @param value 唯一标识某个棋盘特征的整数。
/// @return 充分混合后的 64 位键。
constexpr std::uint64_t mix_zobrist(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

/// 取得“某棋子位于某位置”对应的 Zobrist 键。
/// @param piece 非空棋子编码。
/// @param square 棋盘下标。
/// @return 对应该棋盘特征的稳定 64 位键。
constexpr std::uint64_t piece_zobrist_key(Piece piece, int square) {
    return mix_zobrist(0x7869616e677169ULL +
                       static_cast<std::uint64_t>(piece) * 128ULL +
                       static_cast<std::uint64_t>(square));
}

/// 取得行棋方 Zobrist 键；哈希中异或该键表示黑方行棋。
/// @return 固定的行棋方 64 位键。
constexpr std::uint64_t side_zobrist_key() {
    return mix_zobrist(0x736964655f6b6579ULL);
}

}  // namespace

std::string move_to_string(Move move) {
    const auto encode = [](int square) {
        std::string result;
        result.push_back(static_cast<char>('a' + square_col(square)));
        result.push_back(static_cast<char>('0' + (9 - square_row(square))));
        return result;
    };
    return encode(move.from) + encode(move.to);
}

std::optional<Move> move_from_string(const std::string& text) {
    if (text.size() != 4) {
        return std::nullopt;
    }
    const auto decode = [](char file, char rank) -> int {
        if (file < 'a' || file > 'i' || rank < '0' || rank > '9') {
            return kNoSquare;
        }
        return make_square(9 - (rank - '0'), file - 'a');
    };
    const int from = decode(text[0], text[1]);
    const int to = decode(text[2], text[3]);
    if (from == kNoSquare || to == kNoSquare) {
        return std::nullopt;
    }
    return Move{static_cast<std::uint8_t>(from), static_cast<std::uint8_t>(to)};
}

Position::Position() {
    clear();
}

Position Position::initial() {
    return from_fen("rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w");
}

Position Position::from_fen(const std::string& fen) {
    std::istringstream stream(fen);
    std::string placement;
    std::string side;
    if (!(stream >> placement >> side)) {
        throw std::invalid_argument("FEN must contain placement and side-to-move");
    }

    Position position;
    int row = 0;
    int col = 0;
    for (char token : placement) {
        if (token == '/') {
            if (col != kBoardCols || row >= kBoardRows - 1) {
                throw std::invalid_argument("invalid FEN rank width");
            }
            ++row;
            col = 0;
            continue;
        }
        if (std::isdigit(static_cast<unsigned char>(token))) {
            col += token - '0';
            if (col > kBoardCols) {
                throw std::invalid_argument("invalid FEN empty-square count");
            }
            continue;
        }
        if (row >= kBoardRows || col >= kBoardCols) {
            throw std::invalid_argument("too many squares in FEN");
        }
        position.set_piece(make_square(row, col), fen_to_piece(token));
        ++col;
    }
    if (row != kBoardRows - 1 || col != kBoardCols) {
        throw std::invalid_argument("FEN must contain exactly ten ranks");
    }

    if (side == "w" || side == "r") {
        position.set_side_to_move(Color::Red);
    } else if (side == "b") {
        position.set_side_to_move(Color::Black);
    } else {
        throw std::invalid_argument("invalid FEN side-to-move");
    }
    if (!position.is_valid()) {
        throw std::invalid_argument("FEN must contain exactly one king per side");
    }
    return position;
}

std::string Position::to_fen() const {
    std::string result;
    for (int row = 0; row < kBoardRows; ++row) {
        int empty_count = 0;
        for (int col = 0; col < kBoardCols; ++col) {
            const Piece piece = board_[static_cast<std::size_t>(make_square(row, col))];
            if (is_empty(piece)) {
                ++empty_count;
                continue;
            }
            if (empty_count != 0) {
                result.push_back(static_cast<char>('0' + empty_count));
                empty_count = 0;
            }
            result.push_back(piece_to_fen(piece));
        }
        if (empty_count != 0) {
            result.push_back(static_cast<char>('0' + empty_count));
        }
        if (row != kBoardRows - 1) {
            result.push_back('/');
        }
    }
    result += side_to_move_ == Color::Red ? " w" : " b";
    return result;
}

std::string Position::pretty() const {
    std::ostringstream out;
    for (int row = 0; row < kBoardRows; ++row) {
        out << (9 - row) << "  ";
        for (int col = 0; col < kBoardCols; ++col) {
            const Piece piece = board_[static_cast<std::size_t>(make_square(row, col))];
            out << (is_empty(piece) ? '.' : piece_to_fen(piece));
            if (col != kBoardCols - 1) {
                out << ' ';
            }
        }
        out << '\n';
    }
    out << "\n   a b c d e f g h i\n";
    out << "side: " << (side_to_move_ == Color::Red ? "red" : "black") << '\n';
    return out.str();
}

Piece Position::piece_at(int square) const {
    if (square < 0 || square >= kBoardSize) {
        throw std::out_of_range("square outside board");
    }
    return board_[static_cast<std::size_t>(square)];
}

int Position::king_square(Color color) const {
    return king_squares_[color_index(color)];
}

void Position::clear() {
    board_.fill(kEmpty);
    side_to_move_ = Color::Red;
    king_squares_ = {kNoSquare, kNoSquare};
    zobrist_hash_ = 0;
}

void Position::set_piece(int square, Piece piece) {
    if (square < 0 || square >= kBoardSize) {
        throw std::out_of_range("square outside board");
    }
    const Piece old = board_[static_cast<std::size_t>(square)];
    if (!is_empty(old)) {
        zobrist_hash_ ^= piece_zobrist_key(old, square);
    }
    if (!is_empty(old) && piece_type(old) == PieceType::King) {
        king_squares_[color_index(piece_color(old))] = kNoSquare;
    }
    board_[static_cast<std::size_t>(square)] = piece;
    if (!is_empty(piece) && piece_type(piece) == PieceType::King) {
        king_squares_[color_index(piece_color(piece))] = square;
    }
    if (!is_empty(piece)) {
        zobrist_hash_ ^= piece_zobrist_key(piece, square);
    }
}

void Position::set_side_to_move(Color color) {
    if (side_to_move_ != color) {
        side_to_move_ = color;
        zobrist_hash_ ^= side_zobrist_key();
    }
}

UndoInfo Position::do_move(Move move) {
    if (move.from >= kBoardSize || move.to >= kBoardSize) {
        throw std::out_of_range("move outside board");
    }
    const Piece moving = board_[move.from];
    if (is_empty(moving)) {
        throw std::invalid_argument("cannot move an empty square");
    }

    UndoInfo undo{board_[move.to], king_squares_, zobrist_hash_};
    zobrist_hash_ ^= piece_zobrist_key(moving, move.from);
    if (!is_empty(undo.captured)) {
        zobrist_hash_ ^= piece_zobrist_key(undo.captured, move.to);
    }
    zobrist_hash_ ^= piece_zobrist_key(moving, move.to);
    zobrist_hash_ ^= side_zobrist_key();
    board_[move.to] = moving;
    board_[move.from] = kEmpty;
    if (piece_type(moving) == PieceType::King) {
        king_squares_[color_index(piece_color(moving))] = move.to;
    }
    if (!is_empty(undo.captured) && piece_type(undo.captured) == PieceType::King) {
        king_squares_[color_index(piece_color(undo.captured))] = kNoSquare;
    }
    side_to_move_ = opposite(side_to_move_);
    return undo;
}

void Position::undo_move(Move move, const UndoInfo& undo) {
    if (move.from >= kBoardSize || move.to >= kBoardSize) {
        throw std::out_of_range("move outside board");
    }
    board_[move.from] = board_[move.to];
    board_[move.to] = undo.captured;
    king_squares_ = undo.king_squares;
    side_to_move_ = opposite(side_to_move_);
    zobrist_hash_ = undo.zobrist_hash;
}

void Position::add_step_move(std::vector<Move>& moves, int from, int row, int col,
                             Color side) const {
    if (!inside_board(row, col)) {
        return;
    }
    const int to = make_square(row, col);
    const Piece target = board_[static_cast<std::size_t>(to)];
    if (is_empty(target) ||
        (piece_color(target) != side && piece_type(target) != PieceType::King)) {
        moves.push_back(Move{static_cast<std::uint8_t>(from),
                             static_cast<std::uint8_t>(to)});
    }
}

void Position::generate_rook_moves(std::vector<Move>& moves, int from, Color side) const {
    constexpr int directions[4][2]{{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    for (const auto& direction : directions) {
        int row = square_row(from) + direction[0];
        int col = square_col(from) + direction[1];
        while (inside_board(row, col)) {
            const int to = make_square(row, col);
            const Piece target = board_[static_cast<std::size_t>(to)];
            if (is_empty(target)) {
                moves.push_back(Move{static_cast<std::uint8_t>(from),
                                     static_cast<std::uint8_t>(to)});
            } else {
                if (piece_color(target) != side && piece_type(target) != PieceType::King) {
                    moves.push_back(Move{static_cast<std::uint8_t>(from),
                                         static_cast<std::uint8_t>(to)});
                }
                break;
            }
            row += direction[0];
            col += direction[1];
        }
    }
}

void Position::generate_cannon_moves(std::vector<Move>& moves, int from, Color side) const {
    constexpr int directions[4][2]{{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    for (const auto& direction : directions) {
        bool crossed_screen = false;
        int row = square_row(from) + direction[0];
        int col = square_col(from) + direction[1];
        while (inside_board(row, col)) {
            const int to = make_square(row, col);
            const Piece target = board_[static_cast<std::size_t>(to)];
            if (!crossed_screen) {
                if (is_empty(target)) {
                    moves.push_back(Move{static_cast<std::uint8_t>(from),
                                         static_cast<std::uint8_t>(to)});
                } else {
                    crossed_screen = true;
                }
            } else if (!is_empty(target)) {
                if (piece_color(target) != side && piece_type(target) != PieceType::King) {
                    moves.push_back(Move{static_cast<std::uint8_t>(from),
                                         static_cast<std::uint8_t>(to)});
                }
                break;
            }
            row += direction[0];
            col += direction[1];
        }
    }
}

void Position::generate_horse_moves(std::vector<Move>& moves, int from, Color side) const {
    struct HorseStep { int dr; int dc; int leg_dr; int leg_dc; };
    constexpr HorseStep steps[]{
        {-2, -1, -1, 0}, {-2, 1, -1, 0}, {2, -1, 1, 0}, {2, 1, 1, 0},
        {-1, -2, 0, -1}, {1, -2, 0, -1}, {-1, 2, 0, 1}, {1, 2, 0, 1},
    };
    const int row = square_row(from);
    const int col = square_col(from);
    for (const HorseStep& step : steps) {
        const int leg_row = row + step.leg_dr;
        const int leg_col = col + step.leg_dc;
        if (!inside_board(leg_row, leg_col) ||
            !is_empty(board_[static_cast<std::size_t>(make_square(leg_row, leg_col))])) {
            continue;
        }
        add_step_move(moves, from, row + step.dr, col + step.dc, side);
    }
}

void Position::generate_elephant_moves(std::vector<Move>& moves, int from, Color side) const {
    constexpr int directions[4][2]{{-2, -2}, {-2, 2}, {2, -2}, {2, 2}};
    const int row = square_row(from);
    const int col = square_col(from);
    for (const auto& direction : directions) {
        const int to_row = row + direction[0];
        const int to_col = col + direction[1];
        const int eye_row = row + direction[0] / 2;
        const int eye_col = col + direction[1] / 2;
        if (!inside_board(to_row, to_col) || !on_own_side_of_river(side, to_row) ||
            !is_empty(board_[static_cast<std::size_t>(make_square(eye_row, eye_col))])) {
            continue;
        }
        add_step_move(moves, from, to_row, to_col, side);
    }
}

void Position::generate_advisor_moves(std::vector<Move>& moves, int from, Color side) const {
    constexpr int directions[4][2]{{-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
    for (const auto& direction : directions) {
        const int row = square_row(from) + direction[0];
        const int col = square_col(from) + direction[1];
        if (inside_palace(side, row, col)) {
            add_step_move(moves, from, row, col, side);
        }
    }
}

void Position::generate_king_moves(std::vector<Move>& moves, int from, Color side) const {
    constexpr int directions[4][2]{{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    for (const auto& direction : directions) {
        const int row = square_row(from) + direction[0];
        const int col = square_col(from) + direction[1];
        if (inside_palace(side, row, col)) {
            add_step_move(moves, from, row, col, side);
        }
    }
}

void Position::generate_pawn_moves(std::vector<Move>& moves, int from, Color side) const {
    const int row = square_row(from);
    const int col = square_col(from);
    const int forward = side == Color::Red ? -1 : 1;
    add_step_move(moves, from, row + forward, col, side);
    if (crossed_river(side, row)) {
        add_step_move(moves, from, row, col - 1, side);
        add_step_move(moves, from, row, col + 1, side);
    }
}

std::vector<Move> Position::generate_pseudo_legal_moves() const {
    std::vector<Move> moves;
    moves.reserve(128);
    for (int from = 0; from < kBoardSize; ++from) {
        const Piece piece = board_[static_cast<std::size_t>(from)];
        if (is_empty(piece) || piece_color(piece) != side_to_move_) {
            continue;
        }
        switch (piece_type(piece)) {
            case PieceType::King: generate_king_moves(moves, from, side_to_move_); break;
            case PieceType::Advisor: generate_advisor_moves(moves, from, side_to_move_); break;
            case PieceType::Elephant: generate_elephant_moves(moves, from, side_to_move_); break;
            case PieceType::Horse: generate_horse_moves(moves, from, side_to_move_); break;
            case PieceType::Rook: generate_rook_moves(moves, from, side_to_move_); break;
            case PieceType::Cannon: generate_cannon_moves(moves, from, side_to_move_); break;
            case PieceType::Pawn: generate_pawn_moves(moves, from, side_to_move_); break;
            case PieceType::Empty: break;
        }
    }
    return moves;
}

bool Position::is_square_attacked(int target, Color by_side) const {
    if (target < 0 || target >= kBoardSize) {
        return false;
    }
    const Piece target_piece = board_[static_cast<std::size_t>(target)];
    const bool target_is_enemy_king =
        !is_empty(target_piece) && piece_type(target_piece) == PieceType::King &&
        piece_color(target_piece) != by_side;
    const int target_row = square_row(target);
    const int target_col = square_col(target);

    for (int from = 0; from < kBoardSize; ++from) {
        const Piece piece = board_[static_cast<std::size_t>(from)];
        if (is_empty(piece) || piece_color(piece) != by_side) {
            continue;
        }
        const int row = square_row(from);
        const int col = square_col(from);
        const int dr = target_row - row;
        const int dc = target_col - col;

        switch (piece_type(piece)) {
            case PieceType::Rook:
                if (blockers_between(board_, from, target) == 0) return true;
                break;
            case PieceType::Cannon:
                if (blockers_between(board_, from, target) == 1) return true;
                break;
            case PieceType::Horse: {
                const int adr = std::abs(dr);
                const int adc = std::abs(dc);
                if (!((adr == 2 && adc == 1) || (adr == 1 && adc == 2))) break;
                const int leg_row = row + (adr == 2 ? dr / 2 : 0);
                const int leg_col = col + (adc == 2 ? dc / 2 : 0);
                if (is_empty(board_[static_cast<std::size_t>(make_square(leg_row, leg_col))])) {
                    return true;
                }
                break;
            }
            case PieceType::Elephant:
                if (std::abs(dr) == 2 && std::abs(dc) == 2 &&
                    on_own_side_of_river(by_side, target_row) &&
                    is_empty(board_[static_cast<std::size_t>(
                        make_square(row + dr / 2, col + dc / 2))])) {
                    return true;
                }
                break;
            case PieceType::Advisor:
                if (std::abs(dr) == 1 && std::abs(dc) == 1 &&
                    inside_palace(by_side, target_row, target_col)) {
                    return true;
                }
                break;
            case PieceType::King:
                if (std::abs(dr) + std::abs(dc) == 1 &&
                    inside_palace(by_side, target_row, target_col)) {
                    return true;
                }
                if (target_is_enemy_king && dc == 0 &&
                    blockers_between(board_, from, target) == 0) {
                    return true;
                }
                break;
            case PieceType::Pawn: {
                const int forward = by_side == Color::Red ? -1 : 1;
                if (dr == forward && dc == 0) return true;
                if (crossed_river(by_side, row) && dr == 0 && std::abs(dc) == 1) return true;
                break;
            }
            case PieceType::Empty: break;
        }
    }
    return false;
}

bool Position::in_check(Color color) const {
    const int king = king_square(color);
    return king == kNoSquare || is_square_attacked(king, opposite(color));
}

std::vector<Move> Position::generate_legal_moves() {
    const Color moving_side = side_to_move_;
    const std::vector<Move> pseudo = generate_pseudo_legal_moves();
    std::vector<Move> legal;
    legal.reserve(pseudo.size());
    for (Move move : pseudo) {
        const UndoInfo undo = do_move(move);
        if (!in_check(moving_side)) {
            legal.push_back(move);
        }
        undo_move(move, undo);
    }
    return legal;
}

bool Position::is_legal_move(Move move) {
    const std::vector<Move> legal = generate_legal_moves();
    return std::find(legal.begin(), legal.end(), move) != legal.end();
}

GameStatus Position::status() {
    if (!generate_legal_moves().empty()) {
        return GameStatus::Ongoing;
    }
    return in_check(side_to_move_) ? GameStatus::Checkmate : GameStatus::Stalemate;
}

bool Position::is_valid() const {
    int red_kings = 0;
    int black_kings = 0;
    for (Piece piece : board_) {
        if (!is_empty(piece) && piece_type(piece) == PieceType::King) {
            if (piece_color(piece) == Color::Red) ++red_kings;
            else ++black_kings;
        }
    }
    return red_kings == 1 && black_kings == 1 &&
           king_squares_[color_index(Color::Red)] != kNoSquare &&
           king_squares_[color_index(Color::Black)] != kNoSquare;
}

std::uint64_t perft(Position& position, int depth) {
    if (depth < 0) {
        throw std::invalid_argument("perft depth cannot be negative");
    }
    if (depth == 0) {
        return 1;
    }
    const std::vector<Move> moves = position.generate_legal_moves();
    if (depth == 1) {
        return moves.size();
    }
    std::uint64_t nodes = 0;
    for (Move move : moves) {
        const UndoInfo undo = position.do_move(move);
        nodes += perft(position, depth - 1);
        position.undo_move(move, undo);
    }
    return nodes;
}

}  // namespace xiangqi
