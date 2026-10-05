#include "models/qwen3_5/execution/ffn.h"

#include "core/layout.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/silu_mul.h"

#include <cmath>
#include <cstdlib>
#include <optional>
#include <stdexcept>

namespace ninfer::models::qwen3_5::execution {
namespace {

std::optional<float> target_d48_dx_override() {
    static const std::optional<float> value = []() -> std::optional<float> {
        const char* raw = std::getenv("NINFER_TARGET_D48_DX_OVERRIDE");
        if (raw == nullptr || raw[0] == '\0') return std::nullopt;

        char* end = nullptr;
        const float parsed = std::strtof(raw, &end);
        if (end == raw || *end != '\0' || !std::isfinite(parsed) || parsed <= 0.0F) {
            throw std::runtime_error(
                "NINFER_TARGET_D48_DX_OVERRIDE must be a finite positive float");
        }
        return parsed;
    }();
    return value;
}

std::optional<float> target_g64_dx_override() {
    static const std::optional<float> value = []() -> std::optional<float> {
        const char* raw = std::getenv("NINFER_TARGET_G64_DX_OVERRIDE");
        if (raw == nullptr || raw[0] == '\0') return std::nullopt;

        char* end = nullptr;
        const float parsed = std::strtof(raw, &end);
        if (end == raw || *end != '\0' || !std::isfinite(parsed) || parsed <= 0.0F) {
            throw std::runtime_error(
                "NINFER_TARGET_G64_DX_OVERRIDE must be a finite positive float");
        }
        return parsed;
    }();
    return value;
}

std::optional<float> mtp_gateup_dx_override() {
    static const std::optional<float> value = []() -> std::optional<float> {
        const char* raw = std::getenv("NINFER_MTP_GATEUP_DX_OVERRIDE");
        if (raw == nullptr || raw[0] == '\0') return std::nullopt;

        char* end = nullptr;
        const float parsed = std::strtof(raw, &end);
        if (end == raw || *end != '\0' || !std::isfinite(parsed) || parsed <= 0.0F) {
            throw std::runtime_error(
                "NINFER_MTP_GATEUP_DX_OVERRIDE must be a finite positive float");
        }
        return parsed;
    }();
    return value;
}

} // namespace

std::size_t ffn_workspace_bytes(const FfnParameters& parameters, std::int32_t first,
                                std::int32_t last, bool mtp) {
    if (first <= 0 || last < first) { throw std::invalid_argument("FFN: invalid column interval"); }
    if (const auto* moe = std::get_if<ops::SparseMoeWeights>(&parameters)) {
        return ops::sparse_moe_workspace_capacity_bytes(moe->routed_gate_up.qtype,
                                                        moe->routed_down.qtype, first, last);
    }
    const auto& p    = std::get<DenseParameters>(parameters);
    const auto& gu   = p.gate_up.weight;
    const auto& down = p.down.weight;
    WorkspaceLayoutBuilder layout;
    if (mtp) {
        (void)layout.alloc(DType::BF16, {gu.n, last});
        {
            auto scope = layout.scope();
            (void)layout.alloc_bytes(ops::linear_workspace_capacity_bytes(
                gu.qtype, gu.n, gu.k, p.gate_up.policy, first, last));
        }
        (void)layout.alloc(DType::BF16, {gu.n / 2, last});
        (void)layout.alloc(DType::BF16, {down.n, last});
        (void)layout.alloc_bytes(ops::linear_workspace_capacity_bytes(down.qtype, down.n, down.k,
                                                                      p.down.policy, first, last));
    } else {
        (void)layout.alloc(DType::BF16, {gu.n / 2, last});
        {
            auto scope = layout.scope();
            (void)layout.alloc_bytes(ops::linear_swiglu_workspace_capacity_bytes(
                gu.qtype, gu.n, gu.k, p.gate_up.policy, first, last));
        }
        {
            auto scope = layout.scope();
            (void)layout.alloc_bytes(ops::linear_add_workspace_capacity_bytes(
                down.qtype, down.n, down.k, p.down.policy, first, last));
        }
    }
    return layout.peak_bytes(1);
}

void ffn(const Tensor& hidden, const FfnParameters& parameters, Tensor& residual,
         const ops::SparseMoeHints& hints, WorkspaceArena& workspace, cudaStream_t stream,
         bool mtp) {
    auto scope         = workspace.scope();
    const auto columns = hidden.ne[1];
    if (const auto* moe = std::get_if<ops::SparseMoeWeights>(&parameters)) {
        const auto storage =
            workspace.alloc_bytes(ffn_workspace_bytes(parameters, columns, columns));
        WorkspaceArena scratch(storage);
        ops::sparse_moe(hidden, *moe, ops::SparseMoeEpilogue::AddResidual, residual, hints, scratch,
                        stream);
        return;
    }
    const auto& p    = std::get<DenseParameters>(parameters);
    const auto& gu   = p.gate_up.weight;
    const auto& down = p.down.weight;
    if (mtp) {
        Tensor gate_up = workspace.alloc(DType::BF16, {gu.n, columns});
        {
            auto call = workspace.scope();
            if (const auto dx = mtp_gateup_dx_override(); dx.has_value()) {
                Weight runtime_gu = gu;
                runtime_gu.input_scale_divisor = *dx;
                ops::linear(hidden, runtime_gu, gate_up, p.gate_up.policy, workspace, stream);
            } else {
                ops::linear(hidden, gu, gate_up, p.gate_up.policy, workspace, stream);
            }
        }
        Tensor activation = workspace.alloc(DType::BF16, {gu.n / 2, columns});
        ops::silu_mul(gate_up.slice(0, 0, gu.n / 2), gate_up.slice(0, gu.n / 2, gu.n / 2),
                      activation, stream);
        Tensor delta = workspace.alloc(DType::BF16, {down.n, columns});
        ops::linear(activation, down, delta, p.down.policy, workspace, stream);
        ops::residual_add(delta, residual, stream);
        return;
    }
    Tensor activation = workspace.alloc(DType::BF16, {gu.n / 2, columns});
    {
        auto call = workspace.scope();

        if (gu.qtype == QType::NVFP4) {
            if (const auto dx = target_g64_dx_override(); dx.has_value()) {
                Weight runtime_gu = gu;
                runtime_gu.input_scale_divisor = *dx;
                ops::linear_swiglu(
                    hidden, runtime_gu, activation, p.gate_up.policy, workspace, stream);
            } else {
                ops::linear_swiglu(
                    hidden, gu, activation, p.gate_up.policy, workspace, stream);
            }
        } else {
            ops::linear_swiglu(
                hidden, gu, activation, p.gate_up.policy, workspace, stream);
        }
    }
    if (down.qtype == QType::NVFP4) {
        if (const auto dx = target_d48_dx_override(); dx.has_value()) {
            Weight runtime_down = down;
            runtime_down.input_scale_divisor = *dx;
            ops::linear_add(
                activation, runtime_down, residual, p.down.policy, workspace, stream);
        } else {
            ops::linear_add(
                activation, down, residual, p.down.policy, workspace, stream);
        }
    } else {
        ops::linear_add(
            activation, down, residual, p.down.policy, workspace, stream);
    }
}

} // namespace ninfer::models::qwen3_5::execution
