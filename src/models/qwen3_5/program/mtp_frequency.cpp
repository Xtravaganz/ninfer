#include "models/qwen3_5/program/mtp_frequency.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <cstdio>
#include <cerrno>
#include <cstring>
#include <vector>
#include <string>
#include <system_error>

#include <unistd.h>

namespace ninfer::models::qwen3_5::detail {
namespace {

// Versioned, little-endian on supported x86 host; validated by offline reader later.
struct SnapshotHeader {
    char magic[8] = {'M', 'T', 'P', 'F', 'R', 'Q', '1', '\0'};
    std::uint64_t physical_vocab = MtpFrequencyCollector::kPhysicalVocab;
    std::uint64_t public_vocab = MtpFrequencyCollector::kPublicVocab;
    std::uint64_t rounds = 0;
    std::uint64_t invalid = 0;
};

} // namespace

MtpFrequencyCollector::MtpFrequencyCollector() noexcept {
    const char* dir = std::getenv("NINFER_MTP_FREQ_DIR");
    if (dir == nullptr || *dir == '\0') return;
    try {
        drafted_.assign(kPhysicalVocab, 0);
        accepted_.assign(kPhysicalVocab, 0);
        licensed_.assign(kPhysicalVocab, 0);
        const std::filesystem::path directory(dir);
        std::error_code ec;
        std::filesystem::create_directories(directory, ec);
        if (ec) return;
        // mkstemps uses O_EXCL: no previous process can have its snapshots overwritten.
        std::string template_name = (directory / ("mtp-" +
            std::to_string(static_cast<long long>(getpid())) + "-XXXXXX.bin")).string();
        std::vector<char> path_buffer(template_name.begin(), template_name.end());
        path_buffer.push_back('\0');
        const int fd = mkstemps(path_buffer.data(), 4); // suffix ".bin"
        if (fd < 0) return;
        (void)close(fd);
        file_ = path_buffer.data();
        // Publish enabled only after the writer thread was constructed.
        writer_ = std::thread([this] { run(); });
        enabled_ = true;
    } catch (...) {
        enabled_ = false;
    }
}

MtpFrequencyCollector::~MtpFrequencyCollector() noexcept {
    if (!writer_.joinable()) return;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    wake_.notify_one();
    writer_.join();
}

void MtpFrequencyCollector::add(Histogram& counts, std::int32_t id,
                                std::uint64_t& invalid) noexcept {
    if (id < 0 || static_cast<std::size_t>(id) >= kPublicVocab) {
        ++invalid;
    } else {
        ++counts[static_cast<std::size_t>(id)];
    }
}

void MtpFrequencyCollector::record(std::span<const std::int32_t> drafts,
                                   std::size_t accepted,
                                   std::span<const std::int32_t> licensed) noexcept {
    if (!enabled_) return;
    std::lock_guard<std::mutex> lock(mutex_);
    ++rounds_;
    for (const auto id : drafts) add(drafted_, id, invalid_);
    for (std::size_t i = 0; i < std::min(accepted, drafts.size()); ++i)
        add(accepted_, drafts[i], invalid_);
    for (const auto id : licensed) add(licensed_, id, invalid_);
}

void MtpFrequencyCollector::run() noexcept {
    try {
        std::unique_lock<std::mutex> lock(mutex_);
        while (true) {
            const bool stop = wake_.wait_for(lock, std::chrono::minutes(5),
                                              [this] { return stopping_; });
            lock.unlock();
            snapshot();
            if (stop) return;
            lock.lock();
        }
    } catch (...) {
        // Best-effort diagnostics: never escape on worker thread.
    }
}

void MtpFrequencyCollector::snapshot() noexcept {
    // Disk work and diagnostics run on the writer thread, never on decode threads.
    auto failure = [this](const char* reason) noexcept {
        ++snapshot_failures_;
        std::fprintf(stderr, "[mtp-frequency] snapshot failed (%s), total_failures=%llu\n",
                     reason, static_cast<unsigned long long>(snapshot_failures_));
    };
    try {
        Histogram d, a, l;
        SnapshotHeader header;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            d = drafted_;
            a = accepted_;
            l = licensed_;
            header.rounds = rounds_;
            header.invalid = invalid_;
        }
        auto temporary = file_;
        temporary += ".tmp";
        {
            std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
            if (!out) { failure("open"); return; }
            out.write(reinterpret_cast<const char*>(&header), sizeof(header));
            out.write(reinterpret_cast<const char*>(d.data()),
                      static_cast<std::streamsize>(d.size() * sizeof(std::uint64_t)));
            out.write(reinterpret_cast<const char*>(a.data()),
                      static_cast<std::streamsize>(a.size() * sizeof(std::uint64_t)));
            out.write(reinterpret_cast<const char*>(l.data()),
                      static_cast<std::streamsize>(l.size() * sizeof(std::uint64_t)));
            out.flush();
            if (!out) { failure("write"); return; }
        }
        std::error_code ec;
        std::filesystem::rename(temporary, file_, ec);
        if (ec) { failure("rename"); return; }
        ++snapshot_successes_;

        // Small, human-readable heartbeat adjacent to the binary snapshot.
        // Its failure is observable on stderr, without risking model execution.
        auto status = file_;
        status += ".status";
        auto status_tmp = file_;
        status_tmp += ".status.tmp";
        {
            std::ofstream out(status_tmp, std::ios::trunc);
            if (!out) { failure("status-open"); return; }
            const auto now = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            out << "schema=1\n"
                << "timestamp_unix=" << now << "\n"
                << "rounds=" << header.rounds << "\n"
                << "invalid=" << header.invalid << "\n"
                << "snapshot_successes=" << snapshot_successes_ << "\n"
                << "snapshot_failures=" << snapshot_failures_ << "\n";
            out.flush();
            if (!out) { failure("status-write"); return; }
        }
        ec.clear();
        std::filesystem::rename(status_tmp, status, ec);
        if (ec) failure("status-rename");
    } catch (...) {
        failure("exception");
    }
}

} // namespace ninfer::models::qwen3_5::detail
