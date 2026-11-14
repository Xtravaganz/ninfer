#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <span>
#include <thread>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

// OPTIONAL production telemetry. Counts only committed target-model output IDs.
// This is NOT a full-MTP-head top-1 reference measurement.
// Sampling-free and completely CPU-side; no CUDA or decode-stream synchronization.
class MtpTargetCoverage final {
public:
    static constexpr std::size_t kSelectedTokens = 88064;

    MtpTargetCoverage(std::filesystem::path directory,
                      std::span<const std::int32_t> sorted_indexed_ids,
                      std::size_t public_tokens) noexcept;
    ~MtpTargetCoverage() noexcept;
    MtpTargetCoverage(const MtpTargetCoverage&) = delete;
    MtpTargetCoverage& operator=(const MtpTargetCoverage&) = delete;

    [[nodiscard]] bool enabled() const noexcept { return enabled_; }
    void record_committed(std::span<const std::int32_t> ids) noexcept;

private:
    void write_loop() noexcept;
    void snapshot() noexcept;

    std::size_t public_tokens_ = 0;
    bool enabled_ = false;
    bool stopping_ = false;
    std::filesystem::path file_;
    std::mutex mutex_;
    std::condition_variable wake_;
    std::thread writer_;
    std::vector<std::uint8_t> in_indexed_;
    std::vector<std::uint64_t> committed_hist_;
    std::vector<std::uint64_t> outside_hist_;
    std::uint64_t ranking_fingerprint_ = 0;
    std::uint64_t batches_ = 0;
    std::uint64_t tokens_ = 0;
    std::uint64_t outside_ = 0;
    std::uint64_t invalid_ = 0;
    // Writer-thread-only; not modified from the hot path.
    std::uint64_t writes_ok_ = 0;
    std::uint64_t writes_failed_ = 0;
};

} // namespace ninfer::models::qwen3_5::detail
