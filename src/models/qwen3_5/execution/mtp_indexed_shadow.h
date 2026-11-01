#pragma once

// EXPERIMENT ONLY.
// Optional shadow state for measuring Indexed-Q8 MTP proposal quality.
// The full Q8 proposal remains authoritative.

#include "core/arena.h"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace ninfer::models::qwen3_5::execution {

enum class MtpIndexedMode {
    Shadow,
    Indexed96
};

struct MtpIndexedShadow final {
    static constexpr std::int32_t kPhysicalRows = 135168;
    static constexpr std::int32_t kPublicRows = 135000;
    static constexpr std::int32_t kHidden = 5120;
    static constexpr std::int32_t kLargeRows = 88064;

    // Persistent device-owned selection arrays.
    DeviceBuffer ids96;

    // Persistent BF16 logits, dimensioned for a later bounded
    // maximum proposal batch width. Allocated during initialization.
    DeviceBuffer logits96;

    // Persistent I32 proposal results.
    DeviceBuffer proposals96;

    // Full-Q8 comparison output, stored separately from decoder outputs.
    DeviceBuffer reference_proposals;

    // GPU-side cumulative counters (four uint64 values):
    // total, present96, match96, missing_total.
    // Updated on the same CUDA stream after each shadow proposal.
    DeviceBuffer counters;

    // EXPERIMENT ONLY: first 4096 missing full-head winner IDs.
    static constexpr std::uint32_t kMissingCapacity = 4096;
    DeviceBuffer missing_ids;
    std::uint64_t reported_missing = 0;

    // Explicit opt-in: default behavior remains unchanged.
    bool enabled = false;
    MtpIndexedMode mode = MtpIndexedMode::Shadow;

    // Future implementation will define the actual capacity and initialize
    // all buffers before enabling the shadow path.
    std::int32_t token_capacity = 0;
};

} // namespace ninfer::models::qwen3_5::execution
