#pragma once
// EXPERIMENT ONLY: independent Q8 A16 selected-row Sliced-K launcher.
// Not registered in q8_dispatch or any production CMake target.
// The caller owns the device-side row ID array and guarantees its lifetime.

#include "ops/linear/q8/q8_sliced_k_launch.cuh"
#include "ops/linear/q8/q8_operands.h"
#include "ops/linear/common/output.cuh"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

struct Q8IndexedBenchRows {
    static constexpr int kOutputRowsPerCta = 16;
    const std::int32_t* device_row_ids;

    __device__ __forceinline__ int weight_row(int output_row0, int local_row) const {
        return device_row_ids[output_row0 + local_row];
    }
};

// Synthetic benchmark entrypoint, NOT a public NInfer op.
// Inputs:
// - full: full PHYSICAL Q8 matrix (148480 x 5120), layout validated by NInfer
// - device_row_ids: DEVICE pointer, selected_count distinct valid IDs in [0, full.rows)
// - output: DEVICE buffer sized selected_count * full.tokens BF16 elements
// - selected_count: multiple of 16 (64K/96K both satisfy this)
// ID validity/uniqueness is a caller-side test precondition, not checked per row.
template <class Schedule>
void launch_q8_indexed_bench(const Q8LinearOperands& full,
                             const std::int32_t* device_row_ids,
                             std::int32_t selected_count,
                             __nv_bfloat16* output,
                             cudaStream_t stream) {
    validate_q8_operands(full);
    if (full.rows != 135168 || full.k != 5120 || full.padded_k != 5120) {
        throw std::invalid_argument("indexed Q8 benchmark requires physical 135168 x 5120");
    }
    if (!device_row_ids || !output || selected_count <= 0 ||
        selected_count > full.rows || selected_count % 16 != 0 ||
        (reinterpret_cast<std::uintptr_t>(output) % 16) != 0) {
        throw std::invalid_argument("invalid indexed Q8 benchmark selection/output");
    }
    if (full.tokens <= 0 || full.tokens > Schedule::kTokenCapacity) {
        throw std::invalid_argument("benchmark token count outside schedule capacity");
    }
    // Both physical and logical sizes remain distinct. The CUDA kernel receives
    // physical full.rows for weight bounds, but the selected grid/output use selected_count.
    constexpr int logical_rows_per_cta = Q8IndexedBenchRows::kOutputRowsPerCta;
    constexpr int ratio = Schedule::kBlockRows / logical_rows_per_cta;
    static_assert(Schedule::kBlockRows % logical_rows_per_cta == 0);
    static_assert(Schedule::kStaticK == 0 || Schedule::kStaticK == 5120);
    if (selected_count % ratio != 0 || full.k % Schedule::kBlockK != 0) {
        throw std::invalid_argument("unsupported indexed Q8 schedule geometry");
    }
    const dim3 grid((selected_count / ratio + logical_rows_per_cta - 1) /
                        logical_rows_per_cta,
                    1);
    using Output = LinearBf16Output;
    using Epilogue = LinearIdentityEpilogue;
    using Rows = Q8IndexedBenchRows;
    constexpr auto kernel = q8_a16_sliced_k_mma_kernel<Schedule, true, Output, Epilogue, Rows>;
    const int shared = q8_prepare_shared<Schedule::kSharedBytes, kernel>();
    kernel<<<grid, Schedule::kThreads, shared, stream>>>(
        full, Output{output, selected_count}, Epilogue{}, Rows{device_row_ids}, 0);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
