#include "models/qwen3_5/program/mtp_full_probe.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>
#include <unistd.h>

namespace ninfer::models::qwen3_5::detail {
namespace {
static_assert(std::endian::native == std::endian::little);
constexpr std::uint32_t kHeaderBytes = 128;
constexpr std::uint64_t kFnvOffset = 14695981039346656037ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;

std::uint64_t ranking_hash(std::span<const std::int32_t> ids) {
    std::uint64_t h = kFnvOffset;
    for (auto id : ids) {
        const auto u = static_cast<std::uint32_t>(id);
        for (unsigned shift = 0; shift < 32; shift += 8) {
            h ^= static_cast<std::uint8_t>(u >> shift);
            h *= kFnvPrime;
        }
    }
    return h;
}
void put32(std::array<std::uint8_t, kHeaderBytes>& dst, int offset,
           std::uint32_t v) {
    for (int n = 0; n < 4; ++n) dst[offset + n] = std::uint8_t(v >> (8 * n));
}
void put64(std::array<std::uint8_t, kHeaderBytes>& dst, int offset,
           std::uint64_t v) {
    for (int n = 0; n < 8; ++n) dst[offset + n] = std::uint8_t(v >> (8 * n));
}
std::filesystem::path unique_name(const std::filesystem::path& directory) {
    std::string pattern = (directory / ("mtp-full-" + std::to_string(getpid()) +
                                         "-XXXXXX.bin")).string();
    std::vector<char> buf(pattern.begin(), pattern.end());
    buf.push_back('\0');
    const int fd = mkstemps(buf.data(), 4);
    if (fd < 0) throw std::runtime_error("MTP full probe: mkstemps failed");
    (void)close(fd);
    // Do not create empty files for calibration/idle instances.
    (void)unlink(buf.data());
    return std::filesystem::path(buf.data());
}
} // namespace

MtpFullProbe::MtpFullProbe(std::filesystem::path directory, const DeviceContext& device,
                           const DeviceBuffer& selected_device_ids,
                           std::span<const std::int32_t> sorted_selected_ids,
                           std::uint32_t sample_stride, std::int32_t public_rows)
    : sample_stride_(sample_stride), public_rows_(public_rows),
      excluded_rows_(public_rows - ops::kFullProbeSelected),
      excluded_padded_rows_((excluded_rows_ + 15) / 16 * 16) {
    if (directory.empty() || !directory.is_absolute() ||
        sorted_selected_ids.size() != ops::kFullProbeSelected ||
        selected_device_ids.size() != ops::kFullProbeSelected * sizeof(std::int32_t) ||
        !selected_device_ids.data() || sample_stride < 1 || sample_stride > 4096 ||
        (public_rows != 135000 && public_rows != 131206 && public_rows != 129430 && public_rows != 122919) ||
        excluded_rows_ <= 0 || excluded_padded_rows_ % 16 != 0)
        throw std::invalid_argument("MTP full probe: invalid ranking/directory/stride");
    std::vector<std::int32_t> missing;
    missing.reserve(excluded_padded_rows_);
    std::int32_t previous = -1;
    for (auto id : sorted_selected_ids) {
        if (id <= previous || id >= public_rows_)
            throw std::invalid_argument("MTP full probe: invalid selected ranking");
        previous = id;
    }
    std::size_t cursor = 0;
    for (std::int32_t id = 0; id < public_rows_; ++id) {
        if (cursor < sorted_selected_ids.size() && sorted_selected_ids[cursor] == id)
            ++cursor;
        else
            missing.push_back(id);
    }
    if (missing.size() != excluded_rows_ || cursor != sorted_selected_ids.size())
        throw std::logic_error("MTP full probe: wrong exclusion count");
    missing.resize(excluded_padded_rows_, 0); // tail ignored by comparator
    ranking_fingerprint_ = ranking_hash(sorted_selected_ids);
    std::error_code ec;
    std::filesystem::create_directories(directory, ec);
    if (ec) throw std::runtime_error("MTP full probe: cannot create coverage directory");
    file_ = unique_name(directory);
    if (const char* env = std::getenv("NINFER_MTP_FULL_PROBE_AUDIT"); env && *env) {
        if (std::strcmp(env, "1") != 0)
            throw std::invalid_argument("NINFER_MTP_FULL_PROBE_AUDIT must be 1 when specified");
        audit_mode_ = true;
    }
    const auto d = device.device();
    excluded_device_ = DeviceBuffer(d, missing.size() * sizeof(std::int32_t));
    excluded_logits_ = DeviceBuffer(d, std::size_t(excluded_padded_rows_) *
                                         ops::kFullProbeColumns * sizeof(std::uint16_t));
    missed_hist_ = DeviceBuffer(d, std::size_t(public_rows_) * sizeof(std::uint64_t));
    counters_ = DeviceBuffer(d, std::size_t(ops::kFullProbeCounters) * sizeof(std::uint64_t));
    gate_ = DeviceBuffer(d, sizeof(std::uint32_t));
    if (audit_mode_) {
        audit_observed_ = DeviceBuffer(d, ops::kFullProbeColumns * sizeof(std::int32_t));
        audit_native_ = DeviceBuffer(d, ops::kFullProbeColumns * sizeof(std::int32_t));
        audit_counts_ = DeviceBuffer(d, 3 * sizeof(std::uint64_t));
    }
    excluded_device_.copy_from_host(missing.data(), missing.size() * sizeof(std::int32_t));
    probe_ = {static_cast<const std::int32_t*>(selected_device_ids.data()),
              static_cast<const std::int32_t*>(excluded_device_.data()),
              excluded_logits_.data(),
              static_cast<std::uint64_t*>(missed_hist_.data()),
              static_cast<std::uint64_t*>(counters_.data()),
              static_cast<std::uint32_t*>(gate_.data()), sample_stride_,
              public_rows_, excluded_rows_, excluded_padded_rows_,
              static_cast<std::int32_t*>(audit_observed_.data()),
              static_cast<std::int32_t*>(audit_native_.data()),
              static_cast<std::uint64_t*>(audit_counts_.data()), audit_mode_};
    reset_after_graph_preparation(device);
    writer_ = std::thread([this] { run_writer(); });
}

MtpFullProbe::~MtpFullProbe() noexcept {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    cv_.notify_one();
    if (writer_.joinable()) writer_.join();
}

void MtpFullProbe::reset_after_graph_preparation(const DeviceContext& device) {
    CUDA_CHECK(cudaMemsetAsync(missed_hist_.data(), 0, missed_hist_.size(), device.stream()));
    CUDA_CHECK(cudaMemsetAsync(counters_.data(), 0, counters_.size(), device.stream()));
    CUDA_CHECK(cudaMemsetAsync(gate_.data(), 0, gate_.size(), device.stream()));
    if (audit_mode_)
        CUDA_CHECK(cudaMemsetAsync(audit_counts_.data(), 0, audit_counts_.size(), device.stream()));
    CUDA_CHECK(cudaStreamSynchronize(device.stream()));
    last_snapshot_ = Clock::now();
}

void MtpFullProbe::snapshot_if_due(const DeviceContext& device, bool force) {
    const auto now = Clock::now();
    if (!force && now - last_snapshot_ < std::chrono::minutes(5)) return;
    // The rank's compute stream serializes reads with in-flight probe atomics.
    // This rare readback may stall the worker briefly; no GPU synchronizations
    // occur during ordinary MTP proposals.
    Snapshot s;
    s.missed_ids.resize(public_rows_);
    CUDA_CHECK(cudaMemcpyAsync(s.counters.data(), counters_.data(), counters_.size(),
                               cudaMemcpyDeviceToHost, device.stream()));
    CUDA_CHECK(cudaMemcpyAsync(s.missed_ids.data(), missed_hist_.data(), missed_hist_.size(),
                               cudaMemcpyDeviceToHost, device.stream()));
    if (audit_mode_)
        CUDA_CHECK(cudaMemcpyAsync(s.audit.data(), audit_counts_.data(), audit_counts_.size(),
                                   cudaMemcpyDeviceToHost, device.stream()));
    CUDA_CHECK(cudaStreamSynchronize(device.stream()));
    last_snapshot_ = now;
    if (s.counters[1] == 0) return; // no sampled MTP proposals: no file
    if (audit_mode_) {
        const bool good = s.audit[0] == s.counters[1] &&
                          s.audit[1] + s.audit[2] == s.audit[0] && s.audit[2] == 0;
        std::fprintf(stderr, "[mtp-full-probe] NUMERIC_AUDIT_%s compared=%llu matched=%llu "
                             "mismatched=%llu sampled=%llu\n",
                     good ? "PASS" : "FAIL",
                     static_cast<unsigned long long>(s.audit[0]),
                     static_cast<unsigned long long>(s.audit[1]),
                     static_cast<unsigned long long>(s.audit[2]),
                     static_cast<unsigned long long>(s.counters[1]));
    }
    s.timestamp_unix = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push_back(std::move(s));
    }
    cv_.notify_one();
}

void MtpFullProbe::run_writer() noexcept {
    try {
        std::unique_lock<std::mutex> lock(mutex_);
        for (;;) {
            cv_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
            if (queue_.empty() && stopping_) break;
            auto next = std::move(queue_.front());
            queue_.pop_front();
            lock.unlock();
            write_snapshot(next);
            lock.lock();
        }
    } catch (...) {
        std::fprintf(stderr, "[mtp-full-probe] writer exception\n");
    }
}

void MtpFullProbe::write_snapshot(const Snapshot& data) noexcept {
    const auto failed = [this](const char* label) {
        ++writes_failed_;
        std::fprintf(stderr, "[mtp-full-probe] snapshot failed: %s (%llu)\n", label,
                     static_cast<unsigned long long>(writes_failed_));
    };
    try {
        const auto& c = data.counters;
        if (c[2] + c[3] + c[4] != c[1]) { failed("counter-consistency"); return; }
        std::uint64_t sum = 0;
        for (auto count : data.missed_ids) sum += count;
        if (sum != c[3]) { failed("histogram-consistency"); return; }
        std::array<std::uint8_t, kHeaderBytes> header{};
        std::copy_n("MTPREF3\0", 8, header.begin());
        put32(header, 8, 3); // schema
        put32(header, 12, kHeaderBytes);
        put32(header, 16, public_rows_);
        put32(header, 20, ops::kFullProbeSelected);
        put32(header, 24, excluded_rows_);
        put32(header, 28, sample_stride_);
        put64(header, 32, c[0]); // proposal calls (before GPU sampling)
        put64(header, 40, c[1]); // sampled columns
        put64(header, 48, c[2]); // full top1 inside selected
        put64(header, 56, c[3]); // full top1 OUTSIDE selection
        put64(header, 64, c[4]); // invalid top1/reference state
        put64(header, 72, data.timestamp_unix);
        put64(header, 80, ranking_fingerprint_);
        const auto tmp = std::filesystem::path(file_.string() + ".tmp");
        {
            std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
            if (!f) { failed("open"); return; }
            f.write(reinterpret_cast<const char*>(header.data()), header.size());
            f.write(reinterpret_cast<const char*>(data.missed_ids.data()),
                    static_cast<std::streamsize>(data.missed_ids.size() * sizeof(std::uint64_t)));
            f.flush();
            if (!f) { failed("write"); return; }
        }
        std::error_code ec;
        std::filesystem::rename(tmp, file_, ec);
        if (ec) { failed("rename"); return; }
        ++writes_ok_;
        const auto status = std::filesystem::path(file_.string() + ".status");
        const auto status_tmp = std::filesystem::path(file_.string() + ".status.tmp");
        {
            std::ofstream f(status_tmp, std::ios::trunc);
            if (!f) { failed("status-open"); return; }
            f << "schema=3\nmeasurement=sampled_full_mtp_top1_outside_indexed88\n"
              << "timestamp_unix=" << data.timestamp_unix << '\n'
              << "proposal_calls=" << c[0] << '\n'
              << "sampled_columns=" << c[1] << '\n'
              << "full_top1_inside=" << c[2] << '\n'
              << "full_top1_outside=" << c[3] << '\n'
              << "invalid=" << c[4] << '\n'
              << "sample_stride_calls=" << sample_stride_ << '\n'
              << "ranking_fnv1a64=" << ranking_fingerprint_ << '\n'
              << "snapshot_successes=" << writes_ok_ << '\n'
              << "snapshot_failures=" << writes_failed_ << '\n';
            if (audit_mode_)
                f << "audit_compared=" << data.audit[0] << '\n'
                  << "audit_matches=" << data.audit[1] << '\n'
                  << "audit_mismatches=" << data.audit[2] << '\n';
            f.flush();
            if (!f) { failed("status-write"); return; }
        }
        std::filesystem::rename(status_tmp, status, ec);
        if (ec) failed("status-rename");
    } catch (...) { failed("exception"); }
}
} // namespace ninfer::models::qwen3_5::detail
