#include "ninfer/ops/mtp_full_probe.h"
#include "ninfer/ops/indexed88_q8.h"

#include "ops/common/host/launch.h"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/kernels/q8/q8_a16_sliced_k_mma.cuh"
#include "ops/linear/kernels/q8/q8_operands.h"

#include <cuda_bf16.h>
#include <limits>
#include <cuda_runtime.h>
#include <climits>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops {
namespace {
using Base = detail::Q8A16SlicedKMmaSchedule<
    8, 8, 1, 2, detail::Q8ScaleAccess::Shared,
    Cache::ca, Cache::cg, detail::Q8ActivationStage::ActiveOnly>;
using Schedule = Base::with_problem<5120, 8>;

struct ExcludedRows {
    static constexpr int kOutputRowsPerCta = 16;
    const std::int32_t* ids;
    __device__ __forceinline__ int weight_row(int output_row0, int local_row) const {
        return ids[output_row0 + local_row];
    }
};
static_assert(Schedule::kStaticK == 5120 && Schedule::kTokenCapacity == 8);

// A single device-side clock/counter controls sampling of each MTP projection call.
// Never sample by a C++ if statement during CUDA-Graph capture: that freezes the decision.
__global__ void gate_probe(std::uint64_t* counters, std::uint32_t* enabled,
                           std::uint32_t stride, int columns) {
    if (threadIdx.x != 0) return;
    const auto index = atomicAdd(reinterpret_cast<unsigned long long*>(counters), 1ULL);
    const unsigned active = (index % stride == 0) ? 1u : 0u;
    *enabled = active;
    if (active) atomicAdd(reinterpret_cast<unsigned long long*>(counters + 1),
                          static_cast<unsigned long long>(columns));
}

// The GPU gate prevents all Q8 weight reads for unsampled calls. This kernel still
// launches lightweight CTAs each call; benchmark the residual launch overhead.
__global__ __launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm)
void gated_excluded_q8(detail::Q8LinearOperands operands,
                       detail::LinearBf16Output output, ExcludedRows rows,
                       const std::uint32_t* enabled) {
    if (!*enabled) return;
    detail::q8_a16_sliced_k_mma<Schedule, true>(
        operands, output, detail::LinearIdentityEpilogue{}, rows, 0);
}

__device__ __forceinline__ bool is_better(float a, int ai, float b, int bi) {
    return a > b || (a == b && ai < bi);
}

// Only 1..8 CTAs launch. 256 threads scan the excluded logits for one column and
// obtain the winning original token ID via a shared-memory reduction.
__global__ void compare_maximum(const __nv_bfloat16* excluded_logits,
                                const __nv_bfloat16* indexed_logits,
                                const std::int32_t* excluded_ids,
                                const std::int32_t* selected_ids,
                                const std::int32_t* indexed_winners,
                                std::uint64_t* histogram, std::uint64_t* counters,
                                const std::uint32_t* enabled,
                                std::int32_t* audit_observed, int columns,
                                int public_rows, int excluded_rows, int excluded_padded_rows) {
    if (!*enabled) return;
    const int col = static_cast<int>(blockIdx.x);
    if (col >= columns) return;
    const int thread = static_cast<int>(threadIdx.x);
    __shared__ float best_scores[256];
    __shared__ int best_ids[256];
    float score = -__int_as_float(0x7f800000);
    int id = INT_MAX;
    const auto* start = excluded_logits + static_cast<std::int64_t>(col) * excluded_padded_rows;
    for (int row = thread; row < excluded_rows; row += 256) {
        const float candidate = __bfloat162float(start[row]);
        const int candidate_id = excluded_ids[row];
        if (is_better(candidate, candidate_id, score, id)) {
            score = candidate;
            id = candidate_id;
        }
    }
    best_scores[thread] = score;
    best_ids[thread] = id;
    __syncthreads();
    for (int step = 128; step; step >>= 1) {
        if (thread < step && is_better(best_scores[thread + step], best_ids[thread + step],
                                       best_scores[thread], best_ids[thread])) {
            best_scores[thread] = best_scores[thread + step];
            best_ids[thread] = best_ids[thread + step];
        }
        __syncthreads();
    }
    if (thread) return;
    const int selected_id = indexed_winners[col];
    int low = 0, high = kFullProbeSelected;
    while (low < high) {
        const int mid = low + (high - low) / 2;
        if (selected_ids[mid] < selected_id) low = mid + 1;
        else high = mid;
    }
    if (selected_id < 0 || selected_id >= public_rows ||
        low == kFullProbeSelected || selected_ids[low] != selected_id ||
        best_ids[0] < 0 || best_ids[0] >= public_rows) {
        atomicAdd(reinterpret_cast<unsigned long long*>(counters + 4), 1ULL);
        if (audit_observed) audit_observed[col] = -1;
        return;
    }
    const float indexed_score = __bfloat162float(
        indexed_logits[static_cast<std::int64_t>(col) * kFullProbeSelected + low]);
    if (!isfinite(indexed_score) || !isfinite(best_scores[0])) {
        atomicAdd(reinterpret_cast<unsigned long long*>(counters + 4), 1ULL);
        if (audit_observed) audit_observed[col] = -1;
        return;
    }
    if (audit_observed)
        audit_observed[col] = is_better(best_scores[0], best_ids[0], indexed_score, selected_id)
                                  ? best_ids[0] : selected_id;
    if (is_better(best_scores[0], best_ids[0], indexed_score, selected_id)) {
        atomicAdd(reinterpret_cast<unsigned long long*>(counters + 3), 1ULL);
        atomicAdd(reinterpret_cast<unsigned long long*>(histogram + best_ids[0]), 1ULL);
    } else {
        atomicAdd(reinterpret_cast<unsigned long long*>(counters + 2), 1ULL);
    }
}
} // namespace

void mtp_full_top1_probe(const Tensor& hidden, const Weight& physical_q8_head,
                         const Tensor& indexed_logits, const Tensor& winner_ids,
                         const MtpFullProbeDevice& probe, CudaExecutionView execution) {
    const int columns = hidden.ne[1];
    if (columns < 1 || columns > kFullProbeColumns || !probe.selected_ids ||
        !probe.excluded_ids || !probe.excluded_logits || !probe.counters ||
        !probe.histogram || !probe.enabled || probe.stride < 1 || probe.stride > 4096 ||
        hidden.dtype != DType::BF16 || hidden.ne[0] != 5120 ||
        hidden.device != execution.device() || !hidden.is_contiguous() ||
        indexed_logits.dtype != DType::BF16 || indexed_logits.ne[0] != kFullProbeSelected ||
        indexed_logits.ne[1] != columns || !indexed_logits.is_contiguous() ||
        indexed_logits.device != execution.device() ||
        winner_ids.dtype != DType::I32 || winner_ids.ne[0] != columns ||
        !winner_ids.is_contiguous() || winner_ids.device != execution.device() ||
        physical_q8_head.device != execution.device() ||
        !thinkingcap_indexed88_geometry(probe.public_rows, physical_q8_head.n) ||
        probe.excluded_rows != probe.public_rows - kFullProbeSelected ||
        probe.excluded_padded_rows != ((probe.excluded_rows + 15) / 16) * 16 ||
        physical_q8_head.k != 5120 ||
        physical_q8_head.padded_shape[1] != 5120 ||
        physical_q8_head.qtype != QType::Q8_G32_FP16 ||
        physical_q8_head.layout != QuantLayout::RowSplit ||
        physical_q8_head.group_size != 32) {
        throw std::invalid_argument("MTP full-top1 probe: incompatible ThinkingCap vocabulary configuration");
    }
    const auto operands = detail::q8_linear_operands(hidden, physical_q8_head);
    detail::validate_q8_operands(operands);
    gate_probe<<<1, 1, 0, execution.stream()>>>(probe.counters, probe.enabled,
                                                 probe.stride, columns);
    CUDA_CHECK(cudaGetLastError());
    constexpr auto kernel = gated_excluded_q8;
    const int shared = common::prepare_shared<Schedule::kSharedBytes, kernel>();
    kernel<<<probe.excluded_padded_rows / 16, Schedule::kThreads, shared, execution.stream()>>>(
        operands,
        detail::LinearBf16Output{static_cast<__nv_bfloat16*>(probe.excluded_logits),
                                 probe.excluded_padded_rows},
        ExcludedRows{probe.excluded_ids}, probe.enabled);
    CUDA_CHECK(cudaGetLastError());
    compare_maximum<<<columns, 256, 0, execution.stream()>>>(
        static_cast<const __nv_bfloat16*>(probe.excluded_logits),
        static_cast<const __nv_bfloat16*>(indexed_logits.data), probe.excluded_ids,
        probe.selected_ids, static_cast<const std::int32_t*>(winner_ids.data),
        probe.histogram, probe.counters, probe.enabled, probe.audit_observed, columns,
        probe.public_rows, probe.excluded_rows, probe.excluded_padded_rows);
    CUDA_CHECK(cudaGetLastError());
}
// Independent numeric audit, comparing Stage4B split-argmax to the
// existing native full-head projection + NInfer's reference argmax operator.
__global__ void audit_compare_kernel(const std::int32_t* reference,
                                     const std::int32_t* observed,
                                     const std::uint32_t* enabled,
                                     std::uint64_t* counts, int columns) {
    const int col = static_cast<int>(threadIdx.x);
    if (!*enabled || col >= columns) return;
    atomicAdd(reinterpret_cast<unsigned long long*>(counts), 1ULL);
    if (reference[col] == observed[col]) {
        atomicAdd(reinterpret_cast<unsigned long long*>(counts + 1), 1ULL);
    } else {
        atomicAdd(reinterpret_cast<unsigned long long*>(counts + 2), 1ULL);
    }
}

void mtp_full_probe_audit_compare(const Tensor& native_full_ids,
                                  const MtpFullProbeDevice& probe, CudaExecutionView execution) {
    if (!probe.audit || !probe.audit_counts || !probe.audit_observed || !probe.enabled ||
        native_full_ids.dtype != DType::I32 || native_full_ids.ne[0] < 1 ||
        native_full_ids.ne[0] > kFullProbeColumns || !native_full_ids.is_contiguous() ||
        native_full_ids.device != execution.device())
        throw std::invalid_argument("MTP full-probe numeric audit: bad input");
    audit_compare_kernel<<<1, 32, 0, execution.stream()>>>(
        static_cast<const std::int32_t*>(native_full_ids.data),
        probe.audit_observed, probe.enabled, probe.audit_counts, native_full_ids.ne[0]);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops
