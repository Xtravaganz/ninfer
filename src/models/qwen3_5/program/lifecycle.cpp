#include "models/qwen3_5/program/internal.h"
#include "models/qwen3_5/execution/text.h"
#include "ninfer/ops/indexed88_q8.h"

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <span>
#include <cstdlib>
#include <filesystem>
#include <cstdint>
#include <fstream>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

RankProgramImpl::RankProgramImpl(const execution::Parameters& values, const SequencePlanImpl& plan,
                                 DeviceContext& context, const StartupObserver& observer)
    : parameters(values), device(context), capacity(plan.capacity),
      max_concurrency(plan.max_concurrency), prefill_chunk(plan.prefill_chunk),
      draft_window(plan.draft_window), backend(plan.speculative_backend),
      proposal_head(plan.proposal_head), use_cuda_graph(plan.use_cuda_graph),
      kv_storage(plan.kv_storage), workspace_plan(plan.workspace),
      resources(context, plan.persistent, plan.workspace, observer),
      context_counts(plan.causal_scoring ? 0
                                         : std::size_t(values.profile->public_tokens) *
                                               plan.max_concurrency * sizeof(std::int32_t)),
      result_ready(context), drafts_ready(context), compute_retired(context),
      context_ready(context), prefill_timer(context), gpu_timer(context, plan.gpu_timing) {
    if (parameters.profile != plan.profile || parameters.profile->options != plan.features)
        throw std::invalid_argument("Qwen3.5 rank parameters differ from the prepared plan");
    // Legacy release opt-in names intentionally preserved for existing deployment scripts.
    // This path supports DIRECT Indexed88 only; shadow comparison is not ported.
    // Allocate/upload during startup, before the caller prepares CUDA Graphs.
    if (const char* enable = std::getenv("NINFER_MTP_INDEXED_SHADOW");
        enable != nullptr && enable[0] != '\0') {
        if (std::string(enable) != "1") {
            throw std::invalid_argument("NINFER_MTP_INDEXED_SHADOW must be 1");
        }
        const char* mode = std::getenv("NINFER_MTP_INDEXED_MODE");
        if (!mode || std::string(mode) != "indexed96") {
            throw std::invalid_argument("Indexed88 direct mode requires NINFER_MTP_INDEXED_MODE=indexed96");
        }
        if (backend != SpeculativeBackend::Mtp || proposal_head != ProposalHead::Full ||
            !parameters.mtp ||
            !ops::thinkingcap_indexed88_geometry(parameters.profile->public_tokens,
                                                   parameters.profile->config.text.vocab_size) ||
            parameters.profile->config.text.hidden_size != 5120) {
            throw std::invalid_argument("Indexed88 requires ThinkingCap 135K, Compact131206 or Compact129430 MTP full-head geometry");
        }
        const auto& head = parameters.mtp->output_head.weight;
        if (head.device != device.device() ||
            head.n != parameters.profile->config.text.vocab_size || head.k != 5120 ||
            head.padded_shape[1] != 5120 || head.qtype != QType::Q8_G32_FP16 ||
            head.layout != QuantLayout::RowSplit || head.group_size != 32) {
            throw std::invalid_argument("Indexed88 requires the matching physical 135168/131328/129536x5120 Q8 G32 head");
        }
        const char* path = std::getenv("NINFER_MTP_INDEXED_RANKING");
        if (!path || path[0] != '/') {
            throw std::invalid_argument("NINFER_MTP_INDEXED_RANKING must be an absolute path");
        }
        constexpr std::size_t bytes =
            static_cast<std::size_t>(ops::kThinkingCapIndexed88Rows) * sizeof(std::int32_t);
        std::ifstream input(path, std::ios::binary | std::ios::ate);
        if (!input || input.tellg() != static_cast<std::streamoff>(bytes)) {
            throw std::runtime_error("Indexed88 ranking missing or incorrect byte count");
        }
        input.seekg(0);
        std::vector<std::int32_t> ids(ops::kThinkingCapIndexed88Rows);
        input.read(reinterpret_cast<char*>(ids.data()), static_cast<std::streamsize>(bytes));
        if (!input || input.gcount() != static_cast<std::streamsize>(bytes)) {
            throw std::runtime_error("Indexed88 ranking could not be read completely");
        }
        std::int32_t previous = -1;
        for (const std::int32_t id : ids) {
            if (id <= previous || id >= parameters.profile->public_tokens) {
                throw std::runtime_error("Indexed88 ranking must contain 88064 strictly ascending IDs in [0,public_tokens)");
            }
            previous = id;
        }
        indexed88_ids = DeviceBuffer(device.device(), bytes);
        indexed88_ids.copy_from_host(ids.data(), bytes);
        // Only the live rank records observations. Do not emit calibration artifacts.
        if (!plan.graph_probe) {
            if (const char* dir = std::getenv("NINFER_MTP_FULL_PROBE_DIR"); dir && *dir) {
                std::uint32_t stride = 64;
                if (const char* raw = std::getenv("NINFER_MTP_FULL_PROBE_STRIDE"); raw && *raw) {
                    const auto end = raw + std::char_traits<char>::length(raw);
                    const auto parsed = std::from_chars(raw, end, stride);
                    if (parsed.ec != std::errc{} || parsed.ptr != end ||
                        stride < 1 || stride > 4096)
                        throw std::invalid_argument("NINFER_MTP_FULL_PROBE_STRIDE must be 1..4096");
                }
                full_probe = std::make_unique<MtpFullProbe>(
                    std::filesystem::path(dir), device, indexed88_ids,
                    std::span<const std::int32_t>(ids.data(), ids.size()), stride,
                    parameters.profile->public_tokens);
            }
            if (const char* freq_dir = std::getenv("NINFER_MTP_FREQ_DIR"); freq_dir && *freq_dir) {
                // Target collector is optional; failure must not alter generation.
                try {
                    target_coverage = std::make_unique<MtpTargetCoverage>(
                        std::filesystem::path(freq_dir),
                        std::span<const std::int32_t>(ids.data(), ids.size()),
                        parameters.profile->public_tokens);
                    if (!target_coverage->enabled())
                        std::fprintf(stderr, "[mtp-target-coverage] disabled: writer initialization failed\n");
                } catch (...) {
                    std::fprintf(stderr, "[mtp-target-coverage] disabled: initialization exception\n");
                    target_coverage.reset();
                }
            }
        }
    }
    std::vector<context::ContextTableStorage> tables{{0, &resources.decoder.text_kv.tables()}};
    if (auto* cache = resources.backend_kv()) tables.push_back({1, &cache->tables()});
    storage = std::make_unique<context::ContextStorage>(resources.context.span(),
                                                        resources.host_context.span(), tables);
    compute_actions =
        std::make_unique<context::ContextMigration>(device, *storage, device.execution_view());
    context_actions = std::make_unique<context::ContextMigration>(device, *storage);
    if (plan.moe) {
        std::vector<const moe::Parameters*> layers;
        for (const auto& block : parameters.text.layers)
            if (const auto* layer = std::get_if<moe::Parameters>(&block.ffn))
                layers.push_back(layer);
        resources.moe = plan.moe->create(layers, parameters.model.expert_banks(), device,
                                         plan.graph_probe ? moe::ResourcePurpose::GraphCalibration
                                                          : moe::ResourcePurpose::Execution);
    }
}

RankProgramImpl::~RankProgramImpl() {
    try {
        drain();
    } catch (...) {
        device.bind_to_current_thread_noexcept();
        (void)cudaStreamSynchronize(device.stream());
        (void)cudaStreamSynchronize(device.transfer_stream());
    }
    if (full_probe) {
        try { full_probe->snapshot_if_due(device, true); }
        catch (...) { std::fprintf(stderr, "[mtp-full-probe] final readback failed\n"); }
    }
}

void RankProgramImpl::install_residencies(const ContextWork& work) {
    const auto vocabulary = static_cast<std::size_t>(parameters.profile->public_tokens);
    for (auto id : work.releases) {
        auto& residency = residencies[id.lane.value];
        if (!residency.installed || residency.id != id)
            throw std::logic_error("Qwen3.5 release no longer matches its residency");
        if (resources.vision) {
            const auto* active = resources.vision->current();
            if (active && active->owner.residency == residency.id)
                resources.vision->release(active->id);
        }
        residency.installed = false;
    }
    auto* counts = static_cast<std::int32_t*>(context_counts.data());
    for (const auto& binding : work.bindings) {
        const auto lane = binding.member.residency.lane.value;
        auto& residency = residencies[lane];
        if (resources.vision) {
            const auto* active = resources.vision->current();
            if (active && active->owner.residency == residency.id)
                resources.vision->release(active->id);
        }
        residency.id        = binding.member.residency;
        residency.sequence  = binding.member.sequence;
        residency.installed = true;
        residency.penalties =
            binding.sampling.presence_penalty != 0.0F || binding.sampling.frequency_penalty != 0.0F;
        if (!resources.token_counts.data || !residency.penalties) continue;
        Tensor destination = resources.token_counts.slice(1, static_cast<int>(lane), 1);
        if (binding.counts.empty()) {
            CUDA_CHECK(cudaMemsetAsync(destination.data, 0, destination.bytes(),
                                       device.transfer_stream()));
            continue;
        }
        auto* row_counts = counts + lane * vocabulary;
        binding.counts.restore({row_counts, vocabulary});
        CUDA_CHECK(cudaMemcpyAsync(destination.data, row_counts, vocabulary * sizeof(std::int32_t),
                                   cudaMemcpyHostToDevice, device.transfer_stream()));
    }
}

void RankProgramImpl::start(runtime::ExecutionRequest<ExecutionTypes>&& request) {
    if (request.ticket.frame.kind == runtime::FrameKind::Context) {
        if (context_operation) throw std::logic_error("Qwen3.5 context frame is occupied");
        context_operation =
            std::make_unique<ContextOperation>(ContextOperation{std::move(request), {}});
        context_operation->last_command = context_operation->request.command;
        context_actions->submit(context_operation->request.work.context);
        install_residencies(std::get<ContextWork>(context_operation->request.work.operation));
        context_ready.record(device.transfer_stream());
    } else {
        if (compute) throw std::logic_error("Qwen3.5 compute frame is occupied");
        gpu_timer.begin();
        compute = std::make_unique<ComputeOperation>(std::move(request));
        compute_actions->submit(compute->request.work.context);
        initialize_inputs(*compute);
    }
    progress();
}

void RankProgramImpl::supply(runtime::ExecutionTicket ticket, runtime::CommandSequence command,
                             runtime::DecisionInput&& input) {
    auto& operation             = require_compute(ticket);
    operation.last_command      = command;
    operation.decision          = std::move(input);
    operation.decision_supplied = true;
    progress();
}

void RankProgramImpl::commit(runtime::ExecutionTicket ticket, runtime::CommandSequence command,
                             Commit&& value) {
    auto& operation = require_compute(ticket);
    gpu_timer.resolve();
    operation.last_command = command;
    operation.commit       = std::move(value);
    operation.stage        = ComputeStage::Commit;
    progress();
}

void RankProgramImpl::discard(runtime::ExecutionTicket ticket, runtime::CommandSequence command) {
    auto& operation = require_compute(ticket);
    gpu_timer.resolve();
    operation.last_command = command;
    operation.cancelled    = true;
    progress();
}

std::optional<runtime::ExecutionEvent<ExecutionTypes>> RankProgramImpl::poll() {
    progress();
    if (events.empty()) return std::nullopt;
    auto event = std::move(events.front());
    events.pop_front();
    return event;
}

void RankProgramImpl::drain() {
    std::exception_ptr failure;
    const auto attempt = [&](auto&& action) {
        try {
            action();
        } catch (...) {
            if (!failure) failure = std::current_exception();
        }
    };
    attempt([&] { device.bind_to_current_thread(); });
    if (compute && compute->text) attempt([&] { compute->text->cancel(); });
    if (graphs) attempt([&] { graphs->cancel(); });
    if (resources.moe) {
        attempt([&] { resources.moe->cancel(); });
        attempt([&] { resources.moe->synchronize(); });
    }
    attempt([&] { compute_actions->drain(); });
    attempt([&] { context_actions->drain(); });
    const auto wait_stream = [](cudaStream_t stream) {
        const auto status = cudaStreamSynchronize(stream);
        if (status != cudaSuccess)
            throw std::runtime_error(std::string("Qwen3.5 stream drain: ") +
                                     cudaGetErrorString(status));
    };
    attempt([&] { wait_stream(device.transfer_stream()); });
    attempt([&] { wait_stream(device.stream()); });
    compute.reset();
    context_operation.reset();
    events.clear();
    if (resources.vision)
        if (const auto* active = resources.vision->current()) resources.vision->release(active->id);
    if (failure) std::rethrow_exception(failure);
}

} // namespace ninfer::models::qwen3_5::detail
