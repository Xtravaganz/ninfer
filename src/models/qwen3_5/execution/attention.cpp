#include "models/qwen3_5/execution/attention.h"

#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/rope.h"

#include <cmath>
#include <cstdlib>
#include <optional>
#include <stdexcept>

namespace ninfer::models::qwen3_5::execution {
namespace {

std::optional<float> target_iattn_dx_override() {
    static const std::optional<float> value = []() -> std::optional<float> {
        const char* raw = std::getenv("NINFER_TARGET_IATTN_DX_OVERRIDE");
        if (raw == nullptr || raw[0] == '\0') return std::nullopt;

        char* end = nullptr;
        const float parsed = std::strtof(raw, &end);

        if (end == raw || *end != '\0' || !std::isfinite(parsed) || parsed <= 0.0F) {
            throw std::runtime_error(
                "NINFER_TARGET_IATTN_DX_OVERRIDE must be a finite positive float");
        }

        return parsed;
    }();

    return value;
}

void require_rope_axes(const Tensor& positions, const RopeConfig& config) {
    if (positions.ne[1] != 3) { return; }
    for (std::size_t i = 0; i < config.pair_axes.size(); ++i) {
        if (config.pair_axes[i] != i % 3) {
            throw std::invalid_argument("text RoPE: this MRoPE axis mapping has no native route");
        }
    }
}

} // namespace

std::size_t attention_projection_workspace_bytes(const AttentionParameters& parameters,
                                                 std::int32_t first, std::int32_t last) {
    if (first <= 0 || last < first) {
        throw std::invalid_argument("attention projection: invalid column interval");
    }
    if (const auto* single = std::get_if<LinearParameters>(&parameters.projection)) {
        const auto& weight = single->weight;
        return ops::attn_input_proj_workspace_capacity_bytes(weight.qtype, weight.n, weight.k,
                                                             single->policy, first, last);
    }
    return 0;
}

void attention_projection(const Tensor& hidden, const AttentionParameters& parameters,
                          Tensor& query, Tensor& gate, Tensor& key, Tensor& value,
                          WorkspaceArena& workspace, cudaStream_t stream) {
    if (const auto* pair = std::get_if<ops::PairedProjectionWeights>(&parameters.projection)) {
        ops::attn_input_proj(hidden, pair->first, pair->second, query, gate, key, value, stream);
    } else {
        const auto& single = std::get<LinearParameters>(parameters.projection);

        if (single.weight.qtype == QType::NVFP4) {
            if (const auto dx = target_iattn_dx_override(); dx.has_value()) {
                Weight runtime_weight = single.weight;
                runtime_weight.input_scale_divisor = *dx;
                ops::attn_input_proj(
                    hidden, runtime_weight, query, gate, key, value,
                    single.policy, workspace, stream);
            } else {
                ops::attn_input_proj(
                    hidden, single.weight, query, gate, key, value,
                    single.policy, workspace, stream);
            }
        } else {
            ops::attn_input_proj(
                hidden, single.weight, query, gate, key, value,
                single.policy, workspace, stream);
        }
    }
}

void text_rope(const Tensor& positions, const RopeConfig& config, Tensor& query,
               cudaStream_t stream) {
    require_rope_axes(positions, config);
    ops::rope(positions, dimension(config.rotary_dim), config.rope_theta, query, stream);
}

void text_rope(const Tensor& positions, const RopeConfig& config, Tensor& query, Tensor& key,
               cudaStream_t stream) {
    require_rope_axes(positions, config);
    ops::rope(positions, dimension(config.rotary_dim), config.rope_theta, query, key, stream);
}

} // namespace ninfer::models::qwen3_5::execution
