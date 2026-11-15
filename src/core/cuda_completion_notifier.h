#pragma once

#include <atomic>
#include <array>
#include <cuda_runtime.h>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>

namespace ninfer {

// EXPERIMENTAL: one shared channel per engine. CUDA callbacks hold a shared_ptr
// so that callbacks racing with engine shutdown cannot touch a destroyed CV.
class CudaCompletionNotifier {
public:
    struct State {
        // Only these Qwen3.5 Dense stages are eligible for contract-based waits.
        // No CUDA runtime calls are made by this structure or its callbacks.
        enum class DenseStage : std::uint8_t {
            ContextReady = 1, History = 2, Readback = 3, DeviceRetired = 4
        };
        struct CompletionGeneration {
            std::uint64_t generation = 0;
            std::atomic<bool> delivered{false};
        };
        struct DenseWaitContract {
            std::uint64_t ticket = 0;
            DenseStage stage = DenseStage::ContextReady;
            std::shared_ptr<CompletionGeneration> completion;
        };
        // Diagnostic-only metadata for *why* the current Qwen Dense stage
        // did not publish a complete GPU wait contract.
        enum class DenseGapReason : std::uint8_t {
            None, Ineligible, ConcurrentFrames, Idle,
            UnsupportedStage, MissingGeneration
        };
        struct DenseWaitSnapshot {
            std::optional<DenseWaitContract> contract;
            DenseGapReason reason = DenseGapReason::Idle;
            unsigned compute_stage = 255;
        };
        std::mutex mutex;
        std::condition_variable cv;
        std::uint64_t generation = 0;
        std::atomic<std::uint64_t> gpu_signals{0};
        std::atomic<std::uint64_t> queue_signals{0};
        std::atomic<std::uint64_t> wait_wakes{0};
        std::atomic<std::uint64_t> wait_timeouts{0};
        std::atomic<std::uint64_t> callback_failures{0};
        // A callback is guaranteed to have been submitted only after
        // cudaLaunchHostFunc succeeds. Increment BEFORE launch, since
        // very short GPU work may invoke the callback immediately.
        std::atomic<std::uint64_t> callbacks_pending{0};
        std::atomic<std::uint64_t> guarded_waits{0};
        std::atomic<std::uint64_t> guarded_timeouts{0};
        std::atomic<std::uint64_t> fallback_waits{0};
        std::atomic<std::uint64_t> next_completion_generation{1};
        std::atomic<std::uint64_t> dense_v04_admitted{0};
        std::atomic<std::uint64_t> dense_v04_signaled{0};
        std::atomic<std::uint64_t> dense_v04_watchdog{0};
        std::atomic<std::uint64_t> dense_v04_uncovered{0};
        std::atomic<std::uint64_t> dense_v04_already_delivered{0};
        std::atomic<std::uint64_t> dense_v04_context{0};
        std::atomic<std::uint64_t> dense_v04_history{0};
        std::atomic<std::uint64_t> dense_v04_readback{0};
        std::atomic<std::uint64_t> dense_v04_retired{0};
        std::atomic<std::uint64_t> dense_v04_last_ticket{0};
        std::atomic<std::uint64_t> dense_v04_last_generation{0};
        std::optional<DenseWaitContract> dense_wait;
        DenseGapReason dense_gap_reason = DenseGapReason::Idle;
        unsigned dense_compute_stage = 255;

        // Published only by the owning engine thread after native program progress.
        // The same mutex protects the wake generation and the current contract.
        void publish_dense_wait(std::optional<DenseWaitContract> contract,
                                DenseGapReason reason, unsigned compute_stage) {
            std::lock_guard lock(mutex);
            dense_wait = std::move(contract);
            dense_gap_reason = reason;
            dense_compute_stage = compute_stage;
        }
        [[nodiscard]] DenseWaitSnapshot snapshot_dense_state() {
            std::lock_guard lock(mutex);
            return {dense_wait, dense_gap_reason, dense_compute_stage};
        }
        // A long timeout here is a CUDA-error liveness watchdog, not a normal
        // scheduling tick. Queue/cancel/shutdown GPU signals wake immediately.
        bool wait_dense_change(std::uint64_t before,
                               const std::shared_ptr<CompletionGeneration>& target,
                               std::chrono::steady_clock::time_point deadline) {
            std::unique_lock lock(mutex);
            const bool changed = cv.wait_until(lock, deadline, [&] {
                return generation != before || target->delivered.load(std::memory_order_acquire);
            });
            if (changed)
                wait_wakes.fetch_add(1, std::memory_order_relaxed);
            else
                wait_timeouts.fetch_add(1, std::memory_order_relaxed);
            return changed;
        }
        std::atomic<std::uint64_t> timeout_idle{0};
        std::atomic<std::uint64_t> timeout_poll_progress{0};
        std::atomic<std::uint64_t> timeout_cycle_progress{0};
        std::atomic<std::uint64_t> signal_idle{0};
        std::atomic<std::uint64_t> signal_poll_progress{0};
        std::atomic<std::uint64_t> signal_cycle_progress{0};
        std::atomic<std::uint64_t> completion_record_attempts{0};
        std::atomic<std::uint64_t> completion_callbacks_enqueued{0};
        // v0.6: per-producer-stream notification queues; stats are diagnostics only.
        std::atomic<std::uint64_t> notify_stream_created{0};
        std::atomic<std::uint64_t> notify_stream_create_errors{0};
        std::atomic<std::uint64_t> notify_stream_capacity_fallbacks{0};
        std::atomic<std::uint64_t> notify_wait_submitted{0};
        std::atomic<std::uint64_t> notify_wait_errors{0};
        std::atomic<std::uint64_t> notify_callback_submitted{0};
        std::atomic<std::uint64_t> notify_sync_errors{0};
        std::atomic<std::uint64_t> notify_destroy_errors{0};
        std::atomic<std::uint64_t> callback_cpu_ns_total{0};
        std::atomic<std::uint64_t> callback_cpu_ns_max{0};
        std::atomic<std::uint64_t> completion_capture_skips{0};
        std::atomic<std::uint64_t> completion_capture_query_errors{0};
        std::atomic<std::uint64_t> completion_ready_not_ready{0};
        std::atomic<std::uint64_t> completion_ready_success{0};
        std::atomic<std::uint64_t> qwen_graph_start_waits{0};
        std::atomic<std::uint64_t> qwen_graph_poll_waits{0};
        std::atomic<std::uint64_t> qwen_moe_post_waits{0};
        std::atomic<std::uint64_t> qwen_moe_layer_waits{0};
        std::atomic<std::uint64_t> qwen_moe_launch_waits{0};

        void notify() {
            {
                std::lock_guard lock(mutex);
                ++generation;
            }
            cv.notify_one();
        }
        void notify_gpu() {
            gpu_signals.fetch_add(1, std::memory_order_relaxed);
            notify();
        }
        void notify_queue() {
            queue_signals.fetch_add(1, std::memory_order_relaxed);
            notify();
        }
        [[nodiscard]] std::uint64_t snapshot() {
            std::lock_guard lock(mutex);
            return generation;
        }
        bool wait_change(std::uint64_t before, std::chrono::microseconds fallback) {
            std::unique_lock lock(mutex);
            const bool changed = cv.wait_for(lock, fallback, [&] {
                return generation != before;
            });
            if (changed)
                wait_wakes.fetch_add(1, std::memory_order_relaxed);
            else
                wait_timeouts.fetch_add(1, std::memory_order_relaxed);
            return changed;
        }
    };

    // Subscribes only the coordinator's GPU submission thread. Other threads
    // keep the existing CUDA behavior. Nested scopes restore the prior channel.
    class ThreadScope {
    public:
        explicit ThreadScope(std::shared_ptr<State> channel);
        ~ThreadScope();
        ThreadScope(const ThreadScope&) = delete;
        ThreadScope& operator=(const ThreadScope&) = delete;

        // Called only on this Engine's CUDA submission thread, never by a callback.
        // Returning nullptr selects the original compute-stream notification.
        [[nodiscard]] cudaStream_t stream_for(cudaStream_t producer) noexcept;
    private:
        struct StreamPair {
            cudaStream_t producer = nullptr;
            cudaStream_t notify = nullptr;
        };
        std::array<StreamPair, 4> streams_{};
        std::size_t stream_count_ = 0;
        bool notification_enabled_ = false;
        ThreadScope* previous_scope_ = nullptr;
        std::shared_ptr<State> previous_;
    };
    [[nodiscard]] static std::shared_ptr<State> current() noexcept;
    [[nodiscard]] static cudaStream_t notification_stream_for(cudaStream_t producer) noexcept;
};

} // namespace ninfer
