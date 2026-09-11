#include "xiangqi/nnue.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace xiangqi {
namespace {

constexpr std::array<char, 8> kFileMagic{'X', 'Q', 'N', 'N', 'U', 'E', 'F', '1'};
constexpr std::uint32_t kFileVersion = 1;

/// @brief 对 NNUE 隐藏层值应用范围为 [0, 1] 的 ClippedReLU。
/// @param value 激活前的浮点值。
/// @return 截断后的激活值。
float clipped_relu(float value) {
    return std::clamp(value, 0.0F, 1.0F);
}

/// @brief 以小端格式写入一个无符号 32 位整数。
/// @param stream 目标二进制输出流。
/// @param value 要写入的整数。
/// @throws std::runtime_error 写入失败时抛出。
void write_u32(std::ostream& stream, std::uint32_t value) {
    const std::array<char, 4> bytes{
        static_cast<char>(value & 0xffU),
        static_cast<char>((value >> 8U) & 0xffU),
        static_cast<char>((value >> 16U) & 0xffU),
        static_cast<char>((value >> 24U) & 0xffU),
    };
    stream.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!stream) {
        throw std::runtime_error("failed to write NNUE file");
    }
}

/// @brief 从流中读取一个小端无符号 32 位整数。
/// @param stream 源二进制输入流。
/// @return 读取的整数。
/// @throws std::runtime_error 数据不足时抛出。
std::uint32_t read_u32(std::istream& stream) {
    std::array<unsigned char, 4> bytes{};
    stream.read(reinterpret_cast<char*>(bytes.data()),
                static_cast<std::streamsize>(bytes.size()));
    if (!stream) {
        throw std::runtime_error("truncated NNUE file");
    }
    return static_cast<std::uint32_t>(bytes[0]) |
           (static_cast<std::uint32_t>(bytes[1]) << 8U) |
           (static_cast<std::uint32_t>(bytes[2]) << 16U) |
           (static_cast<std::uint32_t>(bytes[3]) << 24U);
}

/// @brief 以 IEEE-754 float32 的小端位模式写入浮点数。
/// @param stream 目标二进制输出流。
/// @param value 要写入的有限浮点数。
void write_f32(std::ostream& stream, float value) {
    static_assert(sizeof(float) == sizeof(std::uint32_t));
    static_assert(std::numeric_limits<float>::is_iec559);
    write_u32(stream, std::bit_cast<std::uint32_t>(value));
}

/// @brief 从小端 IEEE-754 位模式读取 float32。
/// @param stream 源二进制输入流。
/// @return 读取的浮点数。
float read_f32(std::istream& stream) {
    static_assert(sizeof(float) == sizeof(std::uint32_t));
    static_assert(std::numeric_limits<float>::is_iec559);
    return std::bit_cast<float>(read_u32(stream));
}

/// @brief 将一段浮点数组写入权重文件。
/// @param stream 目标二进制输出流。
/// @param values 要按顺序写入的浮点数组。
void write_values(std::ostream& stream, std::span<const float> values) {
    for (float value : values) {
        write_f32(stream, value);
    }
}

/// @brief 从权重文件填充一段浮点数组。
/// @param stream 源二进制输入流。
/// @param values 接收数据的浮点数组。
void read_values(std::istream& stream, std::span<float> values) {
    for (float& value : values) {
        value = read_f32(stream);
    }
}

/// @brief 判断一段浮点数组是否全部为有限值。
/// @param values 要检查的数组。
/// @return 不包含 NaN 和无穷大时返回 `true`。
bool all_finite(std::span<const float> values) {
    return std::all_of(values.begin(), values.end(), [](float value) {
        return std::isfinite(value);
    });
}

}  // namespace

NnueNetwork::NnueNetwork()
    : feature_weights_(kHalfKAFeatureDimensions * kNnueAccumulatorSize, 0.0F) {}

NnueNetwork NnueNetwork::load(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("cannot open NNUE file for reading: " + path.string());
    }

    std::array<char, kFileMagic.size()> magic{};
    stream.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (!stream || magic != kFileMagic) {
        throw std::runtime_error("invalid NNUE file magic");
    }

    const std::uint32_t version = read_u32(stream);
    const std::uint32_t feature_dimensions = read_u32(stream);
    const std::uint32_t accumulator_size = read_u32(stream);
    const std::uint32_t hidden_size = read_u32(stream);
    if (version != kFileVersion ||
        feature_dimensions != static_cast<std::uint32_t>(kHalfKAFeatureDimensions) ||
        accumulator_size != static_cast<std::uint32_t>(kNnueAccumulatorSize) ||
        hidden_size != static_cast<std::uint32_t>(kNnueHiddenSize)) {
        throw std::runtime_error("incompatible NNUE file version or dimensions");
    }

    NnueNetwork network;
    read_values(stream, network.feature_biases());
    read_values(stream, network.feature_weights());
    read_values(stream, network.hidden_biases());
    read_values(stream, network.hidden_weights());
    network.output_bias() = read_f32(stream);
    read_values(stream, network.output_weights());

    char trailing = 0;
    if (stream.read(&trailing, 1)) {
        throw std::runtime_error("NNUE file contains trailing data");
    }
    if (!stream.eof()) {
        throw std::runtime_error("failed while reading NNUE file");
    }
    network.validate_finite();
    return network;
}

void NnueNetwork::save(const std::filesystem::path& path) const {
    validate_finite();
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) {
        throw std::runtime_error("cannot open NNUE file for writing: " + path.string());
    }

    stream.write(kFileMagic.data(), static_cast<std::streamsize>(kFileMagic.size()));
    if (!stream) {
        throw std::runtime_error("failed to write NNUE file header");
    }
    write_u32(stream, kFileVersion);
    write_u32(stream, static_cast<std::uint32_t>(kHalfKAFeatureDimensions));
    write_u32(stream, static_cast<std::uint32_t>(kNnueAccumulatorSize));
    write_u32(stream, static_cast<std::uint32_t>(kNnueHiddenSize));
    write_values(stream, feature_biases());
    write_values(stream, feature_weights());
    write_values(stream, hidden_biases());
    write_values(stream, hidden_weights());
    write_f32(stream, output_bias_);
    write_values(stream, output_weights());
}

NnueAccumulator NnueNetwork::refresh_accumulator(
    const Position& position, Color perspective) const {
    NnueAccumulator accumulator = feature_biases_;
    const std::vector<HalfKAFeatureIndex> features =
        active_halfka_features(position, perspective);
    for (HalfKAFeatureIndex feature : features) {
        add_feature(accumulator, feature);
    }
    return accumulator;
}

void NnueNetwork::add_feature(
    NnueAccumulator& accumulator, HalfKAFeatureIndex feature) const {
    if (feature >= kHalfKAFeatureDimensions) {
        throw std::out_of_range("NNUE feature index outside network input");
    }
    const std::size_t offset = feature * kNnueAccumulatorSize;
    for (std::size_t neuron = 0; neuron < kNnueAccumulatorSize; ++neuron) {
        accumulator[neuron] += feature_weights_[offset + neuron];
    }
}

void NnueNetwork::remove_feature(
    NnueAccumulator& accumulator, HalfKAFeatureIndex feature) const {
    if (feature >= kHalfKAFeatureDimensions) {
        throw std::out_of_range("NNUE feature index outside network input");
    }
    const std::size_t offset = feature * kNnueAccumulatorSize;
    for (std::size_t neuron = 0; neuron < kNnueAccumulatorSize; ++neuron) {
        accumulator[neuron] -= feature_weights_[offset + neuron];
    }
}

float NnueNetwork::forward(const Position& position, Color perspective) const {
    const NnueAccumulator red = refresh_accumulator(position, Color::Red);
    const NnueAccumulator black = refresh_accumulator(position, Color::Black);
    return forward_from_accumulators(red, black, perspective);
}

float NnueNetwork::forward_from_accumulators(
    const NnueAccumulator& red_accumulator,
    const NnueAccumulator& black_accumulator,
    Color perspective) const {
    const NnueAccumulator& own =
        perspective == Color::Red ? red_accumulator : black_accumulator;
    const NnueAccumulator& opponent =
        perspective == Color::Red ? black_accumulator : red_accumulator;

    std::array<float, kNnueConcatenatedSize> input{};
    for (std::size_t neuron = 0; neuron < kNnueAccumulatorSize; ++neuron) {
        input[neuron] = clipped_relu(own[neuron]);
        input[kNnueAccumulatorSize + neuron] = clipped_relu(opponent[neuron]);
    }

    std::array<float, kNnueHiddenSize> hidden{};
    for (std::size_t neuron = 0; neuron < kNnueHiddenSize; ++neuron) {
        float sum = hidden_biases_[neuron];
        const std::size_t offset = neuron * kNnueConcatenatedSize;
        for (std::size_t input_index = 0;
             input_index < kNnueConcatenatedSize; ++input_index) {
            sum += hidden_weights_[offset + input_index] * input[input_index];
        }
        hidden[neuron] = clipped_relu(sum);
    }

    float output = output_bias_;
    for (std::size_t neuron = 0; neuron < kNnueHiddenSize; ++neuron) {
        output += output_weights_[neuron] * hidden[neuron];
    }
    return output;
}

int NnueNetwork::evaluate(const Position& position, Color perspective) const {
    const NnueAccumulator red = refresh_accumulator(position, Color::Red);
    const NnueAccumulator black = refresh_accumulator(position, Color::Black);
    return evaluate_from_accumulators(red, black, perspective);
}

int NnueNetwork::evaluate_from_accumulators(
    const NnueAccumulator& red_accumulator,
    const NnueAccumulator& black_accumulator,
    Color perspective) const {
    const float output = forward_from_accumulators(
        red_accumulator, black_accumulator, perspective);
    if (!std::isfinite(output)) {
        throw std::runtime_error("NNUE inference produced a non-finite score");
    }
    const float bounded = std::clamp(
        output,
        -static_cast<float>(kNnueEvaluationLimit),
        static_cast<float>(kNnueEvaluationLimit));
    return static_cast<int>(std::lround(bounded));
}

std::span<float> NnueNetwork::feature_weights() {
    return feature_weights_;
}

std::span<const float> NnueNetwork::feature_weights() const {
    return feature_weights_;
}

std::span<float> NnueNetwork::feature_biases() {
    return feature_biases_;
}

std::span<const float> NnueNetwork::feature_biases() const {
    return feature_biases_;
}

std::span<float> NnueNetwork::hidden_weights() {
    return hidden_weights_;
}

std::span<const float> NnueNetwork::hidden_weights() const {
    return hidden_weights_;
}

std::span<float> NnueNetwork::hidden_biases() {
    return hidden_biases_;
}

std::span<const float> NnueNetwork::hidden_biases() const {
    return hidden_biases_;
}

std::span<float> NnueNetwork::output_weights() {
    return output_weights_;
}

std::span<const float> NnueNetwork::output_weights() const {
    return output_weights_;
}

float& NnueNetwork::output_bias() {
    return output_bias_;
}

float NnueNetwork::output_bias() const {
    return output_bias_;
}

void NnueNetwork::validate_finite() const {
    if (!all_finite(feature_weights()) ||
        !all_finite(feature_biases()) ||
        !all_finite(hidden_weights()) ||
        !all_finite(hidden_biases()) ||
        !all_finite(output_weights()) ||
        !std::isfinite(output_bias_)) {
        throw std::runtime_error("NNUE weights contain NaN or infinity");
    }
}

NnueEvaluationState::NnueEvaluationState(
    const NnueNetwork& network, const Position& root_position)
    : network_(network),
      accumulators_{
          network.refresh_accumulator(root_position, Color::Red),
          network.refresh_accumulator(root_position, Color::Black),
      },
      position_hash_(root_position.hash()) {}

void NnueEvaluationState::push_move(
    const Position& position_after, Move move, Piece moved, Piece captured) {
    if (is_empty(moved) || move.from >= kBoardSize || move.to >= kBoardSize ||
        !is_empty(position_after.piece_at(move.from)) ||
        position_after.piece_at(move.to) != moved) {
        throw std::invalid_argument("invalid move metadata for NNUE accumulator update");
    }

    stack_.push_back(Snapshot{accumulators_, position_hash_});
    try {
        const Color mover = piece_color(moved);
        for (Color perspective : {Color::Red, Color::Black}) {
            const std::size_t index = static_cast<std::size_t>(perspective);
            if (piece_type(moved) == PieceType::King && mover == perspective) {
                accumulators_[index] =
                    network_.refresh_accumulator(position_after, perspective);
                continue;
            }

            const int king_square = position_after.king_square(perspective);
            network_.remove_feature(
                accumulators_[index],
                halfka_feature_index(
                    king_square, perspective, moved, move.from));
            if (!is_empty(captured)) {
                network_.remove_feature(
                    accumulators_[index],
                    halfka_feature_index(
                        king_square, perspective, captured, move.to));
            }
            network_.add_feature(
                accumulators_[index],
                halfka_feature_index(
                    king_square, perspective, moved, move.to));
        }
        position_hash_ = position_after.hash();
    } catch (...) {
        accumulators_ = stack_.back().accumulators;
        position_hash_ = stack_.back().position_hash;
        stack_.pop_back();
        throw;
    }
}

void NnueEvaluationState::pop_move() {
    if (stack_.empty()) {
        throw std::logic_error("cannot pop the root NNUE accumulator state");
    }
    accumulators_ = stack_.back().accumulators;
    position_hash_ = stack_.back().position_hash;
    stack_.pop_back();
}

int NnueEvaluationState::evaluate_for(
    const Position& position, Color perspective) const {
    if (position.hash() != position_hash_) {
        throw std::logic_error("NNUE accumulator is out of sync with position");
    }
    return network_.evaluate_from_accumulators(
        accumulators_[static_cast<std::size_t>(Color::Red)],
        accumulators_[static_cast<std::size_t>(Color::Black)],
        perspective);
}

const NnueAccumulator& NnueEvaluationState::accumulator(
    Color perspective) const {
    return accumulators_[static_cast<std::size_t>(perspective)];
}

NnueEvaluator::NnueEvaluator(NnueNetwork network)
    : network_(std::move(network)) {}

NnueEvaluator::NnueEvaluator(const std::filesystem::path& path)
    : network_(NnueNetwork::load(path)) {}

std::string_view NnueEvaluator::name() const noexcept {
    return "nnue-halfka-f32";
}

int NnueEvaluator::evaluate_for(
    const Position& position, Color perspective) const {
    return network_.evaluate(position, perspective);
}

std::unique_ptr<EvaluationState> NnueEvaluator::create_state(
    const Position& position) const {
    return std::make_unique<NnueEvaluationState>(network_, position);
}

}  // namespace xiangqi
