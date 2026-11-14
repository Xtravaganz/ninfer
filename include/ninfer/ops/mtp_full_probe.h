#pragma once
// ThinkingCap 135K / Compact131206: sampled full-vocabulary MTP top-1 reference.
// Read-only observation: NEVER changes proposal_ids or Indexed88 logits.
#include "core/device.h"
#include "core/tensor.h"
#include "core/weight.h"
#include <cstdint>

namespace ninfer::ops {
inline constexpr std::int32_t kFullProbeSelected = 88064;
inline constexpr std::int32_t kFullProbeColumns = 8;
inline constexpr std::int32_t kFullProbeCounters = 5;

// Host POD descriptor, passed by value to captured CUDA kernel nodes.
// Device allocations and selected_ids must remain stable across graph capture/replay.
struct MtpFullProbeDevice {
    const std::int32_t* selected_ids = nullptr;     // [88064] ascending
    const std::int32_t* excluded_ids = nullptr;     // [excluded_padded_rows], tail padded to multiple of 16
    void* excluded_logits = nullptr;                // BF16 [46944, 8]
    std::uint64_t* histogram = nullptr;             // [public_rows] missing full-top1 IDs
    std::uint64_t* counters = nullptr;              // [calls,sampled,inside,outside,invalid]
    std::uint32_t* enabled = nullptr;               // 1 device scalar; overwritten every call
    std::uint32_t stride = 64;
    std::int32_t public_rows = 0;           // 135000, 131206 or 129430
    std::int32_t excluded_rows = 0;         // public_rows - selected_rows
    std::int32_t excluded_padded_rows = 0;  // align excluded_rows to 16
    // TEST-ONLY. Independent full Q8 reference. Never enabled in production.
    std::int32_t* audit_observed = nullptr; // [8] reconstructed top1, sampled columns
    std::int32_t* audit_native = nullptr; // [8] native full Q8 argmax scratch
    std::uint64_t* audit_counts = nullptr; // [compared, matches, mismatches]
    bool audit = false;
};

// Test-only: compare full-head native argmax with the sampled split-head top1.
// No effect on authoritative draft IDs. Consumes existing GPU sample gate.
void mtp_full_probe_audit_compare(const Tensor& native_full_ids,
                                  const MtpFullProbeDevice& probe, CudaExecutionView execution);

// Called AFTER indexed88_q8_propose with unchanged compact logits and selected winner.
// Uses GPU-side periodic gate, so CUDA Graph replay samples at the correct intervals.
// The full-head winner is reconstructed from max(indexed88) vs max(excluded_rows).
// This does not mutate winner_ids, and does not allocate or synchronize GPU memory.
void mtp_full_top1_probe(const Tensor& hidden, const Weight& physical_q8_head,
                         const Tensor& indexed_logits, const Tensor& winner_ids,
                         const MtpFullProbeDevice& telemetry, CudaExecutionView execution);
} // namespace ninfer::ops
