#include "models/qwen3_5/execution/mtp_indexed_shadow_launch.h"

#include "ops/linear/q8/q8_indexed_bench.cuh"
#include "ops/linear/q8/q8_schedule.cuh"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/speculative_round.h"
#include "core/device.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::models::qwen3_5::execution {
namespace {

namespace q8 = ninfer::ops::detail;

using BaseSchedule = q8::Q8A16SlicedKMmaSchedule<
    8, 8, 1, 2,
    q8::Q8ScaleAccess::Shared,
    ninfer::ops::Cache::ca,
    ninfer::ops::Cache::cg,
    q8::Q8ActivationStage::ActiveOnly>;

using Schedule = typename BaseSchedule::template with_problem<5120, 8>;


__device__ bool selected_contains(
    const std::int32_t* ids, int count, std::int32_t target)
{
    int low = 0;
    int high = count;
    while (low < high) {
        const int mid = low + (high - low) / 2;
        if (ids[mid] < target) low = mid + 1;
        else high = mid;
    }
    return low < count && ids[low] == target;
}

__global__ void compare_indexed_proposals(
    const std::int32_t* full,
    const std::int32_t* indexed96,
    const std::int32_t* ids96,
    unsigned long long* counters,
    std::int32_t* missing_ids,
    int tokens)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= tokens) return;

    const std::int32_t reference = full[i];
    const bool present96 =
        selected_contains(ids96, MtpIndexedShadow::kLargeRows, reference);

    atomicAdd(&counters[0], 1ULL);
    if (present96) {
        atomicAdd(&counters[1], 1ULL);
    } else {
        const auto slot = atomicAdd(&counters[3], 1ULL);
        if (slot < MtpIndexedShadow::kMissingCapacity) {
            missing_ids[slot] = reference;
        }
    }
    if (indexed96[i] == reference) atomicAdd(&counters[2], 1ULL);
}

void indexed_proposal(
    const q8::Q8LinearOperands& operands,
    const DeviceBuffer& ids,
    DeviceBuffer& logits,
    DeviceBuffer& proposals,
    std::int32_t selected_rows,
    cudaStream_t stream)
{
    const int tokens = operands.tokens;

    q8::launch_q8_indexed_bench<Schedule>(
        operands,
        static_cast<const std::int32_t*>(ids.p),
        selected_rows,
        static_cast<__nv_bfloat16*>(logits.p),
        stream);

    // Reuse NInfer's registered Argmax and Token-ID Remap.
    // Argmax owns its winner initialization and tie-breaking.
    Tensor logits_view(
        logits.p, DType::BF16, {selected_rows, tokens});
    Tensor proposals_view(
        proposals.p, DType::I32, {tokens});

    ninfer::ops::argmax(
        logits_view, proposals_view, selected_rows, stream);

    ninfer::ops::proposal_remap_token_ids(
        proposals_view,
        static_cast<const std::int32_t*>(ids.p),
        selected_rows,
        stream);

}

} // namespace

void mtp_indexed_shadow_launch(
    const Tensor& hidden,
    const Weight& physical_q8_head,
    const Tensor& full_proposal_tokens,
    MtpIndexedShadow& shadow,
    cudaStream_t stream)
{
    const int tokens = hidden.ne[1];

    if (tokens <= 0 || tokens > 8 || tokens > shadow.token_capacity) {
        throw std::invalid_argument(
            "Indexed MTP shadow supports only T=1..8");
    }
    if (physical_q8_head.n != MtpIndexedShadow::kPhysicalRows ||
        physical_q8_head.k != MtpIndexedShadow::kHidden) {
        throw std::invalid_argument(
            "Indexed MTP shadow requires physical Q8 135168x5120");
    }
    if (!shadow.ids96.p || !shadow.logits96.p ||
        !shadow.proposals96.p || !shadow.counters.p ||
        !shadow.missing_ids.p) {
        throw std::runtime_error(
            "Indexed-96K shadow buffers were not initialized");
    }

    const auto operands =
        q8::q8_linear_operands(hidden, physical_q8_head);

    indexed_proposal(
        operands, shadow.ids96, shadow.logits96,
        shadow.proposals96, MtpIndexedShadow::kLargeRows, stream);

    compare_indexed_proposals<<<1, 32, 0, stream>>>(
        static_cast<const std::int32_t*>(full_proposal_tokens.data),
        static_cast<const std::int32_t*>(shadow.proposals96.p),
        static_cast<const std::int32_t*>(shadow.ids96.p),
        static_cast<unsigned long long*>(shadow.counters.p),
        static_cast<std::int32_t*>(shadow.missing_ids.p),
        tokens);
    CUDA_CHECK(cudaGetLastError());
}


void mtp_indexed_direct_launch(
    const Tensor& hidden,
    const Weight& physical_q8_head,
    Tensor& decoder_proposal_tokens,
    MtpIndexedShadow& state,
    cudaStream_t stream)
{
    const int tokens = hidden.ne[1];

    if (tokens <= 0 || tokens > 8 || tokens > state.token_capacity)
        throw std::invalid_argument(
            "Indexed MTP direct proposal width outside T=1..8");

    if (physical_q8_head.n != MtpIndexedShadow::kPhysicalRows ||
        physical_q8_head.k != MtpIndexedShadow::kHidden ||
        physical_q8_head.padded_shape[1] != MtpIndexedShadow::kHidden)
        throw std::invalid_argument(
            "Indexed MTP direct requires Q8 135168x5120");

    if (decoder_proposal_tokens.dtype != DType::I32 ||
        decoder_proposal_tokens.ne[0] != tokens ||
        !decoder_proposal_tokens.is_contiguous())
        throw std::invalid_argument(
            "Indexed MTP direct proposal output shape mismatch");

    const auto operands =
        q8::q8_linear_operands(hidden, physical_q8_head);

    if (state.mode != MtpIndexedMode::Indexed96) {
        throw std::invalid_argument(
            "Indexed MTP direct requires indexed96 mode");
    }

    if (!state.ids96.p || !state.logits96.p ||
        !state.proposals96.p) {
        throw std::runtime_error(
            "Indexed-96K direct buffers missing");
    }

    indexed_proposal(
        operands, state.ids96, state.logits96,
        state.proposals96, MtpIndexedShadow::kLargeRows, stream);

    CUDA_CHECK(cudaMemcpyAsync(
        decoder_proposal_tokens.data,
        state.proposals96.p,
        static_cast<std::size_t>(tokens) * sizeof(std::int32_t),
        cudaMemcpyDeviceToDevice,
        stream));
}

} // namespace ninfer::models::qwen3_5::execution
