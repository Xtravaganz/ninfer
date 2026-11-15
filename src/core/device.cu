#include "core/device.h"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <memory>
#include <new>
#include <cstring>

namespace ninfer {
namespace {
thread_local std::shared_ptr<CudaCompletionNotifier::State> tls_completion_channel;
// The Engine thread owns all notification streams, not the shared callback State.
// Never destroy a CUDA stream from within cudaLaunchHostFunc.
thread_local CudaCompletionNotifier::ThreadScope* tls_notification_scope = nullptr;

struct DenseNotificationPayload {
    std::shared_ptr<CudaCompletionNotifier::State> channel;
    std::shared_ptr<CudaCompletionNotifier::State::CompletionGeneration> generation;
};


struct SyncSchedule {
    std::string_view name;
    unsigned int flags;
};

constexpr SyncSchedule kSyncSchedules[] = {
    {"spin", cudaDeviceScheduleSpin},
    {"blocking", cudaDeviceScheduleBlockingSync},
    {"yield", cudaDeviceScheduleYield},
    {"auto", cudaDeviceScheduleAuto},
};

unsigned int sync_schedule_from_environment() {
    const char* value = std::getenv("NINFER_CUDA_SYNC");
    if (value == nullptr) { return cudaDeviceScheduleSpin; }
    for (const auto& schedule : kSyncSchedules) {
        if (schedule.name == value) { return schedule.flags; }
    }
    throw std::invalid_argument("NINFER_CUDA_SYNC must be spin, blocking, yield, or auto");
}

std::string cuda_error_message(const char* prefix, cudaError_t err) {
    return std::string(prefix) + ": " + cudaGetErrorName(err) + ": " + cudaGetErrorString(err);
}

void log_cuda_error(const char* op, cudaError_t err) noexcept {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA cleanup failed during %s: %s: %s\n", op, cudaGetErrorName(err),
                     cudaGetErrorString(err));
    }
}

void destroy_stream(cudaStream_t& stream) noexcept {
    if (stream != nullptr) {
        log_cuda_error("cudaStreamDestroy", cudaStreamDestroy(stream));
        stream = nullptr;
    }
}

void destroy_event(cudaEvent_t& event) noexcept {
    if (event != nullptr) {
        log_cuda_error("cudaEventDestroy", cudaEventDestroy(event));
        event = nullptr;
    }
}

} // namespace

DeviceAllocationError::DeviceAllocationError(const char* operation, std::size_t requested_bytes)
    : std::runtime_error(std::string(operation) + ": insufficient Device memory"),
      requested_bytes_(requested_bytes) {}

CudaCompletionNotifier::ThreadScope::ThreadScope(std::shared_ptr<State> channel)
    : previous_(std::move(tls_completion_channel)) {
    tls_completion_channel = std::move(channel);
    previous_scope_ = tls_notification_scope;
    tls_notification_scope = this;
    // An explicit Dense-only experiment; no change to default or MoE operation.
    const char* wanted = std::getenv("NINFER_EXPERIMENT_NOTIFY_STREAM");
    const char* dense = std::getenv("NINFER_EXPERIMENT_EVENT_DENSE_V04");
    notification_enabled_ = wanted && dense && std::strcmp(wanted, "1") == 0 &&
                            std::strcmp(dense, "1") == 0;
}

cudaStream_t CudaCompletionNotifier::ThreadScope::stream_for(cudaStream_t producer) noexcept {
    if (!notification_enabled_ || producer == nullptr) return nullptr;
    for (std::size_t i = 0; i < stream_count_; ++i)
        if (streams_[i].producer == producer) return streams_[i].notify;
    auto channel = tls_completion_channel;
    if (!channel) return nullptr;
    if (stream_count_ == streams_.size()) {
        channel->notify_stream_capacity_fallbacks.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }
    cudaStream_t notify = nullptr;
    const auto created = cudaStreamCreateWithFlags(&notify, cudaStreamNonBlocking);
    if (created != cudaSuccess) {
        channel->notify_stream_create_errors.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }
    streams_[stream_count_++] = {producer, notify};
    channel->notify_stream_created.fetch_add(1, std::memory_order_relaxed);
    return notify;
}

CudaCompletionNotifier::ThreadScope::~ThreadScope() {
    // The submission thread owns the streams and retires every queued host callback
    // before destroying them. Shared callback state never owns a CUDA handle.
    tls_notification_scope = previous_scope_;
    auto channel = tls_completion_channel;
    for (std::size_t i = 0; i < stream_count_; ++i) {
        const auto synchronized = cudaStreamSynchronize(streams_[i].notify);
        if (synchronized != cudaSuccess && channel)
            channel->notify_sync_errors.fetch_add(1, std::memory_order_relaxed);
        const auto destroyed = cudaStreamDestroy(streams_[i].notify);
        if (destroyed != cudaSuccess && channel)
            channel->notify_destroy_errors.fetch_add(1, std::memory_order_relaxed);
    }
    if (notification_enabled_ && channel) {
        std::fprintf(stderr,
                     "[ninfer-event-v06-teardown] streams=%zu sync_errors=%llu "
                     "destroy_errors=%llu pending=%llu\n",
                     stream_count_,
                     static_cast<unsigned long long>(
                         channel->notify_sync_errors.load(std::memory_order_relaxed)),
                     static_cast<unsigned long long>(
                         channel->notify_destroy_errors.load(std::memory_order_relaxed)),
                     static_cast<unsigned long long>(
                         channel->callbacks_pending.load(std::memory_order_relaxed)));
    }
    tls_completion_channel = std::move(previous_);
}

std::shared_ptr<CudaCompletionNotifier::State> CudaCompletionNotifier::current() noexcept {
    return tls_completion_channel;
}

cudaStream_t CudaCompletionNotifier::notification_stream_for(cudaStream_t producer) noexcept {
    return tls_notification_scope ? tls_notification_scope->stream_for(producer) : nullptr;
}

void cuda_check(cudaError_t err, const char* expr, const char* file, int line) {
    if (err == cudaSuccess) { return; }
    std::fprintf(stderr, "%s:%d: CUDA_CHECK(%s) failed: %s: %s\n", file, line, expr,
                 cudaGetErrorName(err), cudaGetErrorString(err));
    std::abort();
}

DeviceContext::DeviceContext(int device_id) : device_(Device::cuda(device_id)) {
    const unsigned int sync_flags = sync_schedule_from_environment();
    int count                     = 0;
    cudaError_t err               = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaGetDeviceCount failed", err));
    }
    if (count <= 0) { throw std::runtime_error("no CUDA devices available"); }
    if (device_id >= count) { throw std::invalid_argument("invalid CUDA device id"); }

    bind_to_current_thread();

    err = cudaSetDeviceFlags(sync_flags);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaSetDeviceFlags failed", err));
    }

    err = cudaGetDeviceProperties(&props_, device_id);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaGetDeviceProperties failed", err));
    }

    cudaStream_t compute = nullptr;
    cudaStream_t load    = nullptr;
    err                  = cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            cuda_error_message("cudaStreamCreateWithFlags(stream) failed", err));
    }

    err = cudaStreamCreateWithFlags(&load, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        destroy_stream(compute);
        throw std::runtime_error(
            cuda_error_message("cudaStreamCreateWithFlags(transfer_stream) failed", err));
    }

    stream_          = compute;
    transfer_stream_ = load;
}

DeviceContext::~DeviceContext() {
    if (stream_ != nullptr || transfer_stream_ != nullptr) { bind_to_current_thread_noexcept(); }
    destroy_stream(transfer_stream_);
    destroy_stream(stream_);
}

DeviceContext::DeviceContext(DeviceContext&& other) noexcept
    : device_(other.device_), stream_(other.stream_), transfer_stream_(other.transfer_stream_),
      props_(other.props_) {
    other.stream_          = nullptr;
    other.transfer_stream_ = nullptr;
}

DeviceContext& DeviceContext::operator=(DeviceContext&& other) noexcept {
    if (this == &other) { return *this; }

    if (stream_ != nullptr || transfer_stream_ != nullptr) { bind_to_current_thread_noexcept(); }
    destroy_stream(transfer_stream_);
    destroy_stream(stream_);

    device_          = other.device_;
    props_           = other.props_;
    stream_          = other.stream_;
    transfer_stream_ = other.transfer_stream_;

    other.stream_          = nullptr;
    other.transfer_stream_ = nullptr;
    return *this;
}

void DeviceContext::bind_to_current_thread() const {
    const cudaError_t err = cudaSetDevice(device_.index());
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaSetDevice failed", err));
    }
}

void DeviceContext::bind_to_current_thread_noexcept() const noexcept {
    log_cuda_error("cudaSetDevice", cudaSetDevice(device_.index()));
}

int DeviceContext::compute_capability() const noexcept { return props_.major * 10 + props_.minor; }

int DeviceContext::multiprocessor_count() const noexcept { return props_.multiProcessorCount; }

CudaExecutionView DeviceContext::execution_view() const noexcept {
    return {device(), stream_, multiprocessor_count()};
}

CudaExecutionView DeviceContext::transfer_view() const noexcept {
    return {device(), transfer_stream_, multiprocessor_count()};
}

CudaExecutionView DeviceContext::execution_view(cudaStream_t borrowed_stream) const {
    CudaDeviceScope scope(device());
    int ordinal      = -1;
    const auto error = cudaStreamGetDevice(borrowed_stream, &ordinal);
    if (error != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaStreamGetDevice failed", error));
    }
    if (ordinal != device_.index()) {
        throw std::invalid_argument("stream belongs to another device");
    }
    return {device(), borrowed_stream, multiprocessor_count()};
}

CudaDeviceScope::CudaDeviceScope(Device device) {
    if (device.type() != DeviceType::CUDA) {
        throw std::invalid_argument("CUDA operation requires a CUDA device");
    }
    auto error = cudaGetDevice(&previous_);
    if (error == cudaSuccess && previous_ != device.index()) {
        error    = cudaSetDevice(device.index());
        changed_ = error == cudaSuccess;
    }
    if (error != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("CUDA device selection failed", error));
    }
}

CudaDeviceScope::~CudaDeviceScope() {
    if (changed_) { log_cuda_error("restore CUDA device", cudaSetDevice(previous_)); }
}

CudaStream::CudaStream(const DeviceContext& device) : device_(device.device()) {
    CudaDeviceScope scope(device_);
    const auto error = cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
    if (error != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaStreamCreateWithFlags failed", error));
    }
}

CudaStream::~CudaStream() {
    try {
        CudaDeviceScope scope(device_);
        destroy_stream(stream_);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "CUDA stream cleanup failed: %s\n", error.what());
    }
}

std::size_t DeviceContext::total_vram() const noexcept { return props_.totalGlobalMem; }

const char* DeviceContext::sync_mode() const {
    bind_to_current_thread();
    unsigned int flags    = 0;
    const cudaError_t err = cudaGetDeviceFlags(&flags);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaGetDeviceFlags failed", err));
    }
    for (const auto& schedule : kSyncSchedules) {
        if ((flags & cudaDeviceScheduleMask) == schedule.flags) { return schedule.name.data(); }
    }
    throw std::runtime_error("unknown CUDA synchronization schedule");
}

void DeviceContext::synchronize() const { CUDA_CHECK(cudaStreamSynchronize(stream_)); }

CudaEventTimer::CudaEventTimer(const DeviceContext& ctx) : CudaEventTimer(ctx, ctx.stream()) {}

CudaEventTimer::CudaEventTimer(const DeviceContext& ctx, cudaStream_t stream) : stream_(stream) {
    if (stream == nullptr) { throw std::invalid_argument("CUDA timer stream is null"); }
    ctx.bind_to_current_thread();

    cudaEvent_t start = nullptr;
    cudaEvent_t stop  = nullptr;
    cudaError_t err   = cudaEventCreate(&start);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaEventCreate(start) failed", err));
    }

    err = cudaEventCreate(&stop);
    if (err != cudaSuccess) {
        destroy_event(start);
        throw std::runtime_error(cuda_error_message("cudaEventCreate(stop) failed", err));
    }

    start_ = start;
    stop_  = stop;
}

CudaEventTimer::~CudaEventTimer() {
    destroy_event(stop_);
    destroy_event(start_);
}

CudaEventTimer::CudaEventTimer(CudaEventTimer&& other) noexcept
    : stream_(other.stream_), start_(other.start_), stop_(other.stop_) {
    other.stream_ = nullptr;
    other.start_  = nullptr;
    other.stop_   = nullptr;
}

CudaEventTimer& CudaEventTimer::operator=(CudaEventTimer&& other) noexcept {
    if (this == &other) { return *this; }

    destroy_event(stop_);
    destroy_event(start_);

    stream_ = other.stream_;
    start_  = other.start_;
    stop_   = other.stop_;

    other.stream_ = nullptr;
    other.start_  = nullptr;
    other.stop_   = nullptr;
    return *this;
}

void CudaEventTimer::start() { CUDA_CHECK(cudaEventRecord(start_, stream_)); }

void CudaEventTimer::record_stop() { CUDA_CHECK(cudaEventRecord(stop_, stream_)); }

void CudaEventTimer::start_external() {
    CUDA_CHECK(cudaEventRecordWithFlags(start_, stream_, cudaEventRecordExternal));
}

void CudaEventTimer::record_stop_external() {
    CUDA_CHECK(cudaEventRecordWithFlags(stop_, stream_, cudaEventRecordExternal));
}

float CudaEventTimer::elapsed_ms() const {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
    return ms;
}

float CudaEventTimer::stop_ms() {
    record_stop();
    CUDA_CHECK(cudaEventSynchronize(stop_));
    return elapsed_ms();
}

CudaCompletionEvent::CudaCompletionEvent(const DeviceContext& ctx) {
    ctx.bind_to_current_thread();
    const cudaError_t err = cudaEventCreateWithFlags(&event_, cudaEventDisableTiming);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaEventCreateWithFlags failed", err));
    }
}

CudaCompletionEvent::~CudaCompletionEvent() { destroy_event(event_); }

CudaCompletionEvent::CudaCompletionEvent(CudaCompletionEvent&& other) noexcept
    : event_(std::exchange(other.event_, nullptr)),
      notification_generation_(std::move(other.notification_generation_)) {}

CudaCompletionEvent& CudaCompletionEvent::operator=(CudaCompletionEvent&& other) noexcept {
    if (this == &other) { return *this; }
    destroy_event(event_);
    event_ = std::exchange(other.event_, nullptr);
    notification_generation_ = std::move(other.notification_generation_);
    return *this;
}

void CudaCompletionEvent::record(cudaStream_t stream) {
    // A re-recorded event must never publish an earlier callback generation.
    notification_generation_.reset();
    if (event_ == nullptr || stream == nullptr) {
        throw std::logic_error("CUDA completion event is not recordable");
    }
    const auto status = cudaEventRecord(event_, stream);
    if (status != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaEventRecord failed", status));
    }

    // Experimental GPU->CPU notification for eager completion events only.
    // cudaLaunchHostFunc is intentionally NOT inserted during CUDA Graph capture:
    // a per-record heap token cannot safely be replayed as a graph host node.
    auto channel = CudaCompletionNotifier::current();
    if (!channel) return;
    channel->completion_record_attempts.fetch_add(1, std::memory_order_relaxed);
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusInvalidated;
    const auto capture_status = cudaStreamIsCapturing(stream, &capture);
    if (capture_status != cudaSuccess) {
        channel->completion_capture_query_errors.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    if (capture != cudaStreamCaptureStatusNone) {
        channel->completion_capture_skips.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    std::shared_ptr<CudaCompletionNotifier::State::CompletionGeneration> generation;
    try {
        generation = std::make_shared<CudaCompletionNotifier::State::CompletionGeneration>();
    } catch (...) {
        channel->callback_failures.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    generation->generation =
        channel->next_completion_generation.fetch_add(1, std::memory_order_relaxed);
    auto* token = new (std::nothrow) DenseNotificationPayload{channel, generation};
    if (!token) {
        channel->callback_failures.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    // v0.6: wait for the specific recorded generation on a *separate* stream.
    // Distinct producer streams get distinct notification streams, so a slow
    // context transfer cannot hold a compute completion behind it (or vice versa).
    // CUDA stream capture was excluded above. No GPU work is moved or reordered.
    cudaStream_t callback_stream = stream;
    bool on_notification_stream = false;
    if (auto notify = CudaCompletionNotifier::notification_stream_for(stream)) {
        const auto status = cudaStreamWaitEvent(notify, event_, 0);
        if (status == cudaSuccess) {
            channel->notify_wait_submitted.fetch_add(1, std::memory_order_relaxed);
            callback_stream = notify;
            on_notification_stream = true;
        } else {
            // Fail closed: preserve original compute-stream callback for this record.
            channel->notify_wait_errors.fetch_add(1, std::memory_order_relaxed);
        }
    }
    // Enrollment BEFORE launch prevents callbacks from racing the counter.
    channel->callbacks_pending.fetch_add(1, std::memory_order_acq_rel);
    const auto launched = cudaLaunchHostFunc(callback_stream, [](void* opaque) {
        std::unique_ptr<DenseNotificationPayload> owned(
            static_cast<DenseNotificationPayload*>(opaque));
        const auto started = std::chrono::steady_clock::now();
        owned->generation->delivered.store(true, std::memory_order_release);
        owned->channel->callbacks_pending.fetch_sub(1, std::memory_order_acq_rel);
        // A CUDA host callback may only publish a CPU wakeup; no CUDA API calls.
        owned->channel->notify_gpu();
        const auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started).count();
        const auto ns = duration > 0 ? static_cast<std::uint64_t>(duration) : 0ULL;
        owned->channel->callback_cpu_ns_total.fetch_add(ns, std::memory_order_relaxed);
        auto peak = owned->channel->callback_cpu_ns_max.load(std::memory_order_relaxed);
        while (peak < ns && !owned->channel->callback_cpu_ns_max.compare_exchange_weak(
                   peak, ns, std::memory_order_relaxed)) {}
    }, token);
    if (launched == cudaSuccess && on_notification_stream)
        channel->notify_callback_submitted.fetch_add(1, std::memory_order_relaxed);
    if (launched != cudaSuccess) {
        delete token;
        channel->callbacks_pending.fetch_sub(1, std::memory_order_acq_rel);
        channel->callback_failures.fetch_add(1, std::memory_order_relaxed);
        // The bounded fallback remains active if host functions are unsupported.
    } else {
        notification_generation_ = std::move(generation);
        channel->completion_callbacks_enqueued.fetch_add(1, std::memory_order_relaxed);
    }
}

void CudaCompletionEvent::wait(cudaStream_t stream) const {
    if (event_ == nullptr || stream == nullptr) {
        throw std::logic_error("CUDA completion event is not waitable");
    }
    const auto status = cudaStreamWaitEvent(stream, event_, 0);
    if (status != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaStreamWaitEvent failed", status));
    }
}

void CudaCompletionEvent::record_external(cudaStream_t stream) {
    // Graph-captured record nodes have no replay-safe per-launch host token.
    notification_generation_.reset();
    if (event_ == nullptr || stream == nullptr) {
        throw std::logic_error("CUDA completion event is not recordable");
    }
    const auto status = cudaEventRecordWithFlags(event_, stream, cudaEventRecordExternal);
    if (status != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaEventRecordWithFlags failed", status));
    }
}

void CudaCompletionEvent::wait_external(cudaStream_t stream) const {
    if (event_ == nullptr || stream == nullptr)
        throw std::logic_error("CUDA completion event is not waitable");
    const auto status = cudaStreamWaitEvent(stream, event_, cudaEventWaitExternal);
    if (status != cudaSuccess)
        throw std::runtime_error(cuda_error_message("cudaStreamWaitEvent external failed", status));
}

bool CudaCompletionEvent::ready() const {
    if (event_ == nullptr) { throw std::logic_error("CUDA completion event is empty"); }
    const cudaError_t status = cudaEventQuery(event_);
    if (status == cudaSuccess) {
        if (auto state = CudaCompletionNotifier::current())
            state->completion_ready_success.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    if (status == cudaErrorNotReady) {
        if (auto state = CudaCompletionNotifier::current())
            state->completion_ready_not_ready.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    throw std::runtime_error(cuda_error_message("cudaEventQuery failed", status));
}

void CudaCompletionEvent::synchronize() const {
    if (event_ == nullptr) { throw std::logic_error("CUDA completion event is empty"); }
    const auto status = cudaEventSynchronize(event_);
    if (status != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaEventSynchronize failed", status));
    }
}

} // namespace ninfer
