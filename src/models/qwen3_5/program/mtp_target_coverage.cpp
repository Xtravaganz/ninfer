#include "models/qwen3_5/program/mtp_target_coverage.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

#include <unistd.h>

namespace ninfer::models::qwen3_5::detail {
namespace {

static_assert(std::endian::native == std::endian::little, "MTP coverage format requires little endian");
constexpr std::uint32_t kSchema = 2;
constexpr std::uint32_t kHeaderBytes = 128;
constexpr std::uint32_t kKind = 1; // committed target-model output tokens, NOT full MTP-head proposals
constexpr std::uint64_t kFnvOffset = 14695981039346656037ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;

void put32(std::array<std::uint8_t, kHeaderBytes>& out, std::size_t pos,
           std::uint32_t value) noexcept {
    for (int i = 0; i < 4; ++i) out[pos + i] = static_cast<std::uint8_t>(value >> (8 * i));
}
void put64(std::array<std::uint8_t, kHeaderBytes>& out, std::size_t pos,
           std::uint64_t value) noexcept {
    for (int i = 0; i < 8; ++i) out[pos + i] = static_cast<std::uint8_t>(value >> (8 * i));
}
std::uint64_t fingerprint(std::span<const std::int32_t> ids) noexcept {
    std::uint64_t hash = kFnvOffset;
    for (const auto id : ids) {
        const std::uint32_t value = static_cast<std::uint32_t>(id);
        for (int shift = 0; shift < 32; shift += 8) {
            hash ^= static_cast<std::uint8_t>(value >> shift);
            hash *= kFnvPrime;
        }
    }
    return hash;
}

} // namespace

MtpTargetCoverage::MtpTargetCoverage(
    std::filesystem::path directory,
    std::span<const std::int32_t> sorted_indexed_ids,
    std::size_t public_tokens) noexcept {
    try {
        if (directory.empty() || sorted_indexed_ids.size() != kSelectedTokens ||
            (public_tokens != 135000 && public_tokens != 131206 && public_tokens != 129430 && public_tokens != 122919)) return;
        public_tokens_ = public_tokens;
        in_indexed_.assign(public_tokens_, 0);
        std::int32_t previous = -1;
        for (const auto id : sorted_indexed_ids) {
            if (id <= previous || id < 0 || static_cast<std::size_t>(id) >= public_tokens_)
                return;
            in_indexed_[static_cast<std::size_t>(id)] = 1;
            previous = id;
        }
        ranking_fingerprint_ = fingerprint(sorted_indexed_ids);
        committed_hist_.assign(public_tokens_, 0);
        outside_hist_.assign(public_tokens_, 0);
        std::error_code ec;
        std::filesystem::create_directories(directory, ec);
        if (ec) return;
        std::string pattern = (directory / ("mtp-target-" + std::to_string(getpid()) +
                           "-XXXXXX.bin")).string();
        std::vector<char> buffer(pattern.begin(), pattern.end());
        buffer.push_back('\0');
        const int fd = mkstemps(buffer.data(), 4);
        if (fd < 0) return;
        (void)close(fd);
        file_ = buffer.data();
        writer_ = std::thread([this] { write_loop(); });
        enabled_ = true;
    } catch (...) {
        enabled_ = false;
    }
}

MtpTargetCoverage::~MtpTargetCoverage() noexcept {
    if (!writer_.joinable()) return;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    wake_.notify_one();
    writer_.join();
}

void MtpTargetCoverage::record_committed(std::span<const std::int32_t> ids) noexcept {
    if (!enabled_ || ids.empty()) return;
    std::lock_guard<std::mutex> lock(mutex_);
    ++batches_;
    for (auto id : ids) {
        if (id < 0 || static_cast<std::size_t>(id) >= public_tokens_) {
            ++invalid_;
            continue;
        }
        ++tokens_;
        ++committed_hist_[static_cast<std::size_t>(id)];
        if (!in_indexed_[static_cast<std::size_t>(id)]) {
            ++outside_;
            ++outside_hist_[static_cast<std::size_t>(id)];
        }
    }
}

void MtpTargetCoverage::write_loop() noexcept {
    try {
        std::unique_lock<std::mutex> lock(mutex_);
        while (true) {
            const bool stopping = wake_.wait_for(lock, std::chrono::minutes(5),
                                                 [this] { return stopping_; });
            lock.unlock();
            snapshot();
            if (stopping) return;
            lock.lock();
        }
    } catch (...) {
        std::fprintf(stderr, "[mtp-target-coverage] writer thread exception\n");
    }
}

void MtpTargetCoverage::snapshot() noexcept {
    auto failed = [this](const char* reason) noexcept {
        ++writes_failed_;
        std::fprintf(stderr, "[mtp-target-coverage] snapshot failed: %s (%llu)\n", reason,
                     static_cast<unsigned long long>(writes_failed_));
    };
    try {
        std::vector<std::uint64_t> committed, outside;
        std::uint64_t batches, tokens, misses, invalid;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            committed = committed_hist_;
            outside = outside_hist_;
            batches = batches_;
            tokens = tokens_;
            misses = outside_;
            invalid = invalid_;
        }
        const auto now = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        std::array<std::uint8_t, kHeaderBytes> header{};
        std::copy_n("MTPMIS2\0", 8, header.begin());
        put32(header, 8, kSchema);
        put32(header, 12, kHeaderBytes);
        put32(header, 16, public_tokens_);
        put32(header, 20, kSelectedTokens);
        put32(header, 24, kKind);
        put64(header, 32, batches);
        put64(header, 40, tokens);
        put64(header, 48, misses);
        put64(header, 56, invalid);
        put64(header, 64, static_cast<std::uint64_t>(now));
        put64(header, 72, ranking_fingerprint_);
        // 80..127 reserved for optional future metadata. Version remains fixed.
        auto temp = file_;
        temp += ".tmp";
        {
            std::ofstream out(temp, std::ios::binary | std::ios::trunc);
            if (!out) { failed("open"); return; }
            out.write(reinterpret_cast<const char*>(header.data()), header.size());
            out.write(reinterpret_cast<const char*>(committed.data()),
                      static_cast<std::streamsize>(committed.size() * sizeof(std::uint64_t)));
            out.write(reinterpret_cast<const char*>(outside.data()),
                      static_cast<std::streamsize>(outside.size() * sizeof(std::uint64_t)));
            out.flush();
            if (!out) { failed("write"); return; }
        }
        std::error_code ec;
        std::filesystem::rename(temp, file_, ec);
        if (ec) { failed("rename"); return; }
        ++writes_ok_;
        auto status = file_;
        status += ".status";
        auto temp_status = file_;
        temp_status += ".status.tmp";
        {
            std::ofstream out(temp_status, std::ios::trunc);
            if (!out) { failed("status-open"); return; }
            out << "schema=2\n"
                << "measurement=committed_target_tokens_outside_indexed88\n"
                << "timestamp_unix=" << now << '\n'
                << "batches=" << batches << '\n'
                << "tokens=" << tokens << '\n'
                << "outside=" << misses << '\n'
                << "invalid=" << invalid << '\n'
                << "ranking_fnv1a64=" << ranking_fingerprint_ << '\n'
                << "snapshot_successes=" << writes_ok_ << '\n'
                << "snapshot_failures=" << writes_failed_ << '\n';
            out.flush();
            if (!out) { failed("status-write"); return; }
        }
        ec.clear();
        std::filesystem::rename(temp_status, status, ec);
        if (ec) failed("status-rename");
    } catch (...) {
        failed("exception");
    }
}

} // namespace ninfer::models::qwen3_5::detail
