#pragma once

// ThinkingCap 135K and Compact131206 Indexed88 proposal path (88,064 rows).
// This operator borrows the ONE physical Q8 output head and a device-resident,
// validated ascending list of selected vocabulary IDs. No second weight matrix.

#include "core/device.h"
#include "core/tensor.h"
#include "core/weight.h"

#include <cstdint>

namespace ninfer::ops {

inline constexpr std::int32_t kThinkingCapIndexed88Rows = 88064;
inline constexpr std::int32_t kThinkingCap135KPublicRows = 135000;
inline constexpr std::int32_t kThinkingCap135KPhysicalRows = 135168;
inline constexpr std::int32_t kThinkingCapCompact131206PublicRows = 131206;
inline constexpr std::int32_t kThinkingCapCompact131206PhysicalRows = 131328;
inline constexpr std::int32_t kThinkingCapCompact129430PublicRows = 129430;
inline constexpr std::int32_t kThinkingCapCompact129430PhysicalRows = 129536;

[[nodiscard]] constexpr bool thinkingcap_indexed88_geometry(std::int32_t public_rows,
                                                             std::int32_t physical_rows) {
    return (public_rows == kThinkingCap135KPublicRows &&
            physical_rows == kThinkingCap135KPhysicalRows) ||
           (public_rows == kThinkingCapCompact131206PublicRows &&
            physical_rows == kThinkingCapCompact131206PhysicalRows) ||
           (public_rows == kThinkingCapCompact129430PublicRows &&
            physical_rows == kThinkingCapCompact129430PhysicalRows) || (public_rows == 122919 && physical_rows == 123136);
}

[[nodiscard]] constexpr bool thinkingcap_indexed88_physical_rows(std::int32_t rows) {
    return rows == kThinkingCap135KPhysicalRows ||
           rows == kThinkingCapCompact131206PhysicalRows ||
           rows == kThinkingCapCompact129430PhysicalRows || (rows == 123136);
}

// The caller owns selected_ids_device and all tensors throughout asynchronous execution.
// selected_ids_device contains exactly kThinkingCapIndexed88Rows distinct ascending I32
// IDs in [0, public_tokens), validated and uploaded before CUDA Graph capture.
// logits must be contiguous BF16 [88064, T]; proposal_ids contiguous I32 [T].
// T=1..8. No allocations, synchronization, or persistent state in this operation.
void indexed88_q8_propose(const Tensor& hidden, const Weight& physical_q8_head,
                          const std::int32_t* selected_ids_device, Tensor& logits,
                          Tensor& proposal_ids, CudaExecutionView execution);

} // namespace ninfer::ops
