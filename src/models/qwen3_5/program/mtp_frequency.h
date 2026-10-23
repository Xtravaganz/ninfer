#pragma once

#include <array>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <span>
#include <thread>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

class MtpFrequencyCollector final {
public:
    static constexpr std::size_t kPhysicalVocab = 147712;
    static constexpr std::size_t kPublicVocab = 147590;
    using Histogram = std::vector<std::uint64_t>;

    MtpFrequencyCollector() noexcept;
    ~MtpFrequencyCollector() noexcept;
    MtpFrequencyCollector(const MtpFrequencyCollector&) = delete;
    MtpFrequencyCollector& operator=(const MtpFrequencyCollector&) = delete;

    [[nodiscard]] bool enabled() const noexcept { return enabled_; }
    void record(std::span<const std::int32_t> drafts,
                std::size_t accepted,
                std::span<const std::int32_t> licensed) noexcept;

private:
    void run() noexcept;
    void snapshot() noexcept;
    static void add(Histogram& counts, std::int32_t id,
                    std::uint64_t& invalid) noexcept;

    bool enabled_ = false;
    bool stopping_ = false;
    std::filesystem::path file_;
    std::mutex mutex_;
    std::condition_variable wake_;
    std::thread writer_;
    Histogram drafted_;
    Histogram accepted_;
    Histogram licensed_;
    std::uint64_t rounds_ = 0;
    std::uint64_t invalid_ = 0;
    // Writer-thread-only counters; never touched by the decode hot path.
    std::uint64_t snapshot_successes_ = 0;
    std::uint64_t snapshot_failures_ = 0;
};

} // namespace ninfer::models::qwen3_5::detail
