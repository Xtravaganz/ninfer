#pragma once
#include "core/buffer.h"
#include "core/device.h"
#include "ninfer/ops/mtp_full_probe.h"

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <mutex>
#include <span>
#include <thread>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

// Standalone, opt-in, sampled MTP FULL-head top-1 monitor.
// The model always runs its existing Indexed88 direct path; telemetry never changes IDs.
// GPU counters/histogram are snapshotted on the rank's own CUDA stream at most every
// 300 seconds (or after drain), then disk I/O runs on a separate CPU writer thread.
class MtpFullProbe final {
public:
    MtpFullProbe(std::filesystem::path directory, const DeviceContext& device,
                 const DeviceBuffer& selected_device_ids,
                 std::span<const std::int32_t> sorted_selected_ids,
                 std::uint32_t sample_stride, std::int32_t public_rows);
    ~MtpFullProbe() noexcept;
    MtpFullProbe(const MtpFullProbe&) = delete;
    MtpFullProbe& operator=(const MtpFullProbe&) = delete;

    [[nodiscard]] const ops::MtpFullProbeDevice* view() const noexcept { return &probe_; }
    // Call only before serving live requests, once startup Graph calibration has finished.
    void reset_after_graph_preparation(const DeviceContext& device);
    // Call only on the rank worker thread, after completion of a compute operation.
    // The one-stream readback is intentionally rare; persistent data otherwise stay GPU-side.
    void snapshot_if_due(const DeviceContext& device, bool force = false);

private:
    using Clock = std::chrono::steady_clock;
    struct Snapshot {
        std::array<std::uint64_t, ops::kFullProbeCounters> counters{};
        std::vector<std::uint64_t> missed_ids;
        std::uint64_t timestamp_unix = 0;
        std::array<std::uint64_t, 3> audit{};
    };
    void run_writer() noexcept;
    void write_snapshot(const Snapshot& snapshot) noexcept;

    std::filesystem::path file_;
    std::uint64_t ranking_fingerprint_ = 0;
    std::uint32_t sample_stride_ = 64;
    std::int32_t public_rows_ = 0;
    std::int32_t excluded_rows_ = 0;
    std::int32_t excluded_padded_rows_ = 0;
    DeviceBuffer excluded_device_;
    DeviceBuffer excluded_logits_;
    DeviceBuffer missed_hist_;
    DeviceBuffer counters_;
    DeviceBuffer gate_;
    // Test-only independent Q8 numerical audit buffers.
    DeviceBuffer audit_observed_;
    DeviceBuffer audit_native_;
    DeviceBuffer audit_counts_;
    bool audit_mode_ = false;
    ops::MtpFullProbeDevice probe_{};
    Clock::time_point last_snapshot_ = Clock::now();
    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<Snapshot> queue_;
    bool stopping_ = false;
    std::thread writer_;
    std::uint64_t writes_ok_ = 0;
    std::uint64_t writes_failed_ = 0;
};
} // namespace ninfer::models::qwen3_5::detail
