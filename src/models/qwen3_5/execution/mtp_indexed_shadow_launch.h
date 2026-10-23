#pragma once

#include "core/tensor.h"
#include "core/weight.h"
#include "models/qwen3_5/execution/mtp_indexed_shadow.h"

#include <cuda_runtime.h>

namespace ninfer::models::qwen3_5::execution {

// EXPERIMENT ONLY.
// Computes indexed proposals into shadow-owned buffers.
// Never writes to the decoder's authoritative proposal output.
// Requires 1 <= hidden.ne[1] <= 8.
void mtp_indexed_shadow_launch(
    const Tensor& hidden,
    const Weight& physical_q8_head,
    const Tensor& full_proposal_tokens,
    MtpIndexedShadow& shadow,
    cudaStream_t stream);


// EXPERIMENT ONLY: authoritative indexed proposal route.
// Computes exactly one selected projection and writes original TokenIds
// into decoder_proposal_tokens. Does not allocate a second Q8 weight matrix.
void mtp_indexed_direct_launch(
    const Tensor& hidden,
    const Weight& physical_q8_head,
    Tensor& decoder_proposal_tokens,
    MtpIndexedShadow& state,
    cudaStream_t stream);

} // namespace ninfer::models::qwen3_5::execution
