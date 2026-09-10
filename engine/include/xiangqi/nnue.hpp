#pragma once

#include "xiangqi/evaluation.hpp"
#include "xiangqi/halfka.hpp"

#include <array>
#include <cstddef>
#include <filesystem>
#include <span>
#include <string_view>
#include <vector>

namespace xiangqi {

constexpr std::size_t kNnueAccumulatorSize = 256;
constexpr std::size_t kNnueConcatenatedSize = kNnueAccumulatorSize * 2;
constexpr std::size_t kNnueHiddenSize = 32;
constexpr int kNnueEvaluationLimit = 28000;

using NnueAccumulator = std::array<float, kNnueAccumulatorSize>;

/// @brief 保存并执行第一版 HalfKA float32 NNUE 网络。
///
/// 网络结构固定为 11340→256（双方共享特征变换）→拼接 512→32→1。
/// 两个隐藏层均使用范围为 [0, 1] 的 ClippedReLU，最终输出直接使用引擎分值单位。
class NnueNetwork {
public:
    /// @brief 创建全部权重和偏置均为零的网络。
    NnueNetwork();

    /// @brief 从版本化二进制文件加载 NNUE 权重。
    /// @param path 权重文件路径。
    /// @return 完成维度和数值校验的网络。
    /// @throws std::runtime_error 文件无法读取、格式错误、维度不兼容或包含非有限值时抛出。
    [[nodiscard]] static NnueNetwork load(const std::filesystem::path& path);

    /// @brief 将当前网络保存为供 C++ 和未来 Python 导出器共享的二进制格式。
    /// @param path 输出权重文件路径；已存在的普通文件会被覆盖。
    /// @throws std::runtime_error 文件无法写入或权重包含非有限值时抛出。
    void save(const std::filesystem::path& path) const;

    /// @brief 从局面完整计算指定视角的第一层累加器。
    /// @param position 要评估的有效局面。
    /// @param perspective 累加器的观察方。
    /// @return 特征变换偏置加上全部激活 HalfKA 特征权重的 256 维结果。
    [[nodiscard]] NnueAccumulator refresh_accumulator(
        const Position& position, Color perspective) const;

    /// @brief 执行完整 NNUE 前向传播并返回未取整的输出。
    /// @param position 要评估的有效局面。
    /// @param perspective 输出分数所站的阵营；该方累加器会放在拼接向量前半部分。
    /// @return 以引擎分值为单位的 float32 网络输出。
    [[nodiscard]] float forward(const Position& position, Color perspective) const;

    /// @brief 执行前向传播并转换为搜索器使用的整数静态评分。
    /// @param position 要评估的有效局面。
    /// @param perspective 输出分数所站的阵营。
    /// @return 四舍五入并限制在非将杀区间内的整数分数。
    [[nodiscard]] int evaluate(const Position& position, Color perspective) const;

    /// @brief 获取可修改的特征变换权重。
    /// @return 按 `[feature][accumulator_neuron]` 排列的连续数组。
    [[nodiscard]] std::span<float> feature_weights();

    /// @brief 获取只读特征变换权重。
    /// @return 按 `[feature][accumulator_neuron]` 排列的连续数组。
    [[nodiscard]] std::span<const float> feature_weights() const;

    /// @brief 获取可修改的特征变换偏置。
    /// @return 256 个累加器神经元的偏置。
    [[nodiscard]] std::span<float> feature_biases();

    /// @brief 获取只读特征变换偏置。
    /// @return 256 个累加器神经元的偏置。
    [[nodiscard]] std::span<const float> feature_biases() const;

    /// @brief 获取可修改的隐藏层权重。
    /// @return 按 `[hidden_neuron][concatenated_input]` 排列的连续数组。
    [[nodiscard]] std::span<float> hidden_weights();

    /// @brief 获取只读隐藏层权重。
    /// @return 按 `[hidden_neuron][concatenated_input]` 排列的连续数组。
    [[nodiscard]] std::span<const float> hidden_weights() const;

    /// @brief 获取可修改的隐藏层偏置。
    /// @return 32 个隐藏神经元的偏置。
    [[nodiscard]] std::span<float> hidden_biases();

    /// @brief 获取只读隐藏层偏置。
    /// @return 32 个隐藏神经元的偏置。
    [[nodiscard]] std::span<const float> hidden_biases() const;

    /// @brief 获取可修改的输出层权重。
    /// @return 与 32 个隐藏神经元对应的权重。
    [[nodiscard]] std::span<float> output_weights();

    /// @brief 获取只读输出层权重。
    /// @return 与 32 个隐藏神经元对应的权重。
    [[nodiscard]] std::span<const float> output_weights() const;

    /// @brief 获取可修改的输出层偏置。
    /// @return 输出层唯一标量偏置的引用。
    [[nodiscard]] float& output_bias();

    /// @brief 获取只读输出层偏置。
    /// @return 输出层唯一标量偏置。
    [[nodiscard]] float output_bias() const;

private:
    std::vector<float> feature_weights_;
    NnueAccumulator feature_biases_{};
    std::array<float, kNnueHiddenSize * kNnueConcatenatedSize> hidden_weights_{};
    std::array<float, kNnueHiddenSize> hidden_biases_{};
    std::array<float, kNnueHiddenSize> output_weights_{};
    float output_bias_{0.0F};

    /// @brief 检查全部权重和偏置是否为有限浮点数。
    /// @throws std::runtime_error 发现 NaN 或无穷大时抛出。
    void validate_finite() const;
};

/// @brief 通过统一 Evaluator 接口提供 HalfKA NNUE 全量前向推理。
class NnueEvaluator final : public Evaluator {
public:
    /// @brief 使用已经构造好的网络创建评估器。
    /// @param network 要由评估器持有的 NNUE 网络权重。
    explicit NnueEvaluator(NnueNetwork network);

    /// @brief 直接从二进制权重文件创建评估器。
    /// @param path 权重文件路径。
    /// @throws std::runtime_error 权重文件无法正确加载时抛出。
    explicit NnueEvaluator(const std::filesystem::path& path);

    /// @brief 返回 NNUE 评估器名称。
    /// @return 固定字符串 `nnue-halfka-f32`。
    [[nodiscard]] std::string_view name() const noexcept override;

    /// @brief 从指定阵营视角执行 HalfKA NNUE 前向推理。
    /// @param position 待评估局面。
    /// @param perspective 评分所站的阵营。
    /// @return 正数表示 `perspective` 有利，负数表示其不利。
    [[nodiscard]] int evaluate_for(
        const Position& position, Color perspective) const override;

    /// @brief 获取评估器持有的只读网络，供诊断和测试使用。
    /// @return 当前 NNUE 网络。
    [[nodiscard]] const NnueNetwork& network() const { return network_; }

private:
    NnueNetwork network_;
};

}  // namespace xiangqi
