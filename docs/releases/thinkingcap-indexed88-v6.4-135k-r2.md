# ThinkingCap Qwen3.8-27B — NInfer 135K Indexed88 v6.4 R2

> **Release-specific NInfer runtime.** This source release is intended for the separately distributed
> [ThinkingCap-Qwen3.8-27B-NInfer model](https://huggingface.co/Schestex/ThinkingCap-Qwen3.8-27B-NInfer).
> It is not a model-weights download.

## Release profile

| Item | Value |
| --- | --- |
| Runtime release | `thinkingcap-indexed88-v6.4-135k-r2` |
| Model | ThinkingCap Qwen3.8-27B, 135K NInfer artifact |
| Public vocabulary entries | 135,000 |
| Physical Q8 MTP head shape | 135,168 × 5,120 |
| Selected MTP proposal rows | 88,064 |
| Proposal format | Physical Q8 G32/FP16-scales, indexed by a fixed sorted row-ID list |
| MTP mode | Indexed88 **direct** proposal, one selected projection, not shadow/double-head |
| Speculative depth tested | MTP5 (`--spec mtp --draft-tokens 5`) |
| GPU validated | NVIDIA RTX PRO 4000 Blackwell, SM120 (`sm_120a`), 24 GB VRAM |
| CUDA used for release build | 13.3 |
| Reference kernel | R0: 80 registers/thread, 17,920 B shared/CTA, 0 B local |

**Important:** `indexed96` is the current *internal configuration string* for this direct mode;
`kLargeRows` is 88,064. The internal name does not mean 96,000 rows are projected. The existing
`NINFER_MTP_INDEXED_SHADOW=1` switch enables the feature; with `NINFER_MTP_INDEXED_MODE=indexed96`,
the implementation selects the authoritative direct path and skips the full proposal projection.

## Required artifacts

Download the **matching** model and runtime resources from
[the ThinkingCap Hugging Face repository](https://huggingface.co/Schestex/ThinkingCap-Qwen3.8-27B-NInfer):

- NInfer model artifact (`thinkingcap-135000-q8-repack-EXPERIMENTAL.ninfer` in the validated local run; the public filename may be different).
- Sorted physical row-ID list for Indexed88 v6.4 R2 (88,064 signed little-endian int32 IDs).
- Matching `qwen3_8_sharp.jinja` chat template.

Verified ranking SHA256:

```
b578828c0b517ba47abe90a130fda7cde5a56d94add1afeac7623e7c8f6234b4
```

Validated template SHA256:

```
cdff39fb26b60dc90faa292e726655c6b21f62db497846e02e4c4bbab942a84a
```

Validated local model file size: **15,820,439,556 bytes**. Do not substitute a different
vocabulary remap or Indexed88 ranking. A verified public model SHA256 should be added to the
Hugging Face release manifest when the model file is uploaded.

## Build

On a Linux host with a CUDA toolchain supporting `sm_120a`, CMake and Ninja:

```bash
git clone https://github.com/Xtravaganz/ninfer.git
cd ninfer
git checkout thinkingcap-indexed88-v6.4-135k-r2

cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=120a \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.3/bin/nvcc
cmake --build build --target ninfer-serve --parallel 6
```

The CUDA executable was validated as a complete release build. The local release binary used for
build verification had SHA256:

```
b826e03375e41e73ba9417f51097f42f4e79886c1bb357aeea096e5a43ce3446
```

That hash identifies the **local binary**, not a guaranteed byte-identical binary on other machines
or toolchains. A source tag should be treated as the canonical runtime revision.

## Example: single GPU, direct Indexed88, MTP5

Set the following paths to the three matched Hugging Face files:

```bash
export NINFER_MTP_INDEXED_SHADOW=1
export NINFER_MTP_INDEXED_MODE=indexed96
export NINFER_MTP_INDEXED_RANKING=/absolute/path/to/selected_88064_135k_v64_40_R2.i32le

./build/apps/ninfer-serve /absolute/path/to/thinkingcap-135000-q8-repack-EXPERIMENTAL.ninfer \
  --host 127.0.0.1 --port 8080 \
  --model-id thinkingcap-135k-indexed88 \
  --device 0 \
  --spec mtp --draft-tokens 5 \
  --kv-dtype nvfp4 \
  --max-context 8192 --kv-capacity 8192 \
  --max-concurrency 1 \
  --max-pending-requests 8 --pending-timeout-ms 120000 \
  --prefill-chunk 512 --default-max-tokens 256 \
  --device-state-slots 1 --host-state-slots 1 --host-kv-mib 256 \
  --preserve-thinking \
  --chat-template /absolute/path/to/qwen3_8_sharp.jinja
```

`--device 0` refers to the first device **visible to the process**, not necessarily physical GPU 0.
The provided flags reproduce the relevant configuration of the validated local MTP5 benchmark;
change cache/context settings only after checking available VRAM. There is no secondary Q5/Q6
proposal-head load in this release path.

## Measured results (not general model benchmarks)

On one RTX PRO 4000 Blackwell with 3 sequential 3,500-output-token requests per runtime variant,
the **R0** source/kernel (matching this release) produced:

| Consecutive turn | Throughput | Mean temperature | Mean device power |
| --- | ---: | ---: | ---: |
| 1 | 95.3446 tokens/s | 47.639 °C | 143.888 W |
| 2 | 94.7527 tokens/s | 64.686 °C | 144.937 W |
| 3 | 90.4423 tokens/s | 70.243 °C | 144.896 W |

For the three turns, accepted/drafted counts were 2,585/4,568, 2,585/4,568 and 2,539/4,797,
respectively. This is **one controlled test run** using a selected prompt/workload and a fixed
8,192-token context configuration, not a general claim of model quality or portable throughput.

Two subsequent experimental variants (R2: row-ID hoisting; R3: row-ID hoisting and a tighter
register/occupancy launch constraint) produced bit-identical response hashes and equal MTP
acceptance in this test. They did **not** show a meaningful sustained MTP5 throughput advantage.
Consequently, **R0** is the documented release baseline; R2 and R3 are not represented as shipping
runtime changes here.

## Scope and limitations

- The indexed proposal omits nonselected vocabulary rows; it is not mathematically identical to
  a full-vocabulary argmax in cases where the winning token falls outside the selection.
- The selected proposal uses the existing physical Q8 weight tensor; it does not allocate a second
  complete language-model head.
- This source release does not redistribute the 15.8 GB model file, token-ID mapping or private
  calibration data; those artifacts are documented separately on Hugging Face.
- The NInfer **source code** uses the repository's Apache-2.0 license. The **model weights** retain
  their own upstream/derivative licensing requirements; do not infer Apache-2.0 licensing for the
  weights from this runtime repository.
- This release documents an SM120a/24 GB configuration. Other GPUs, toolchains and more ambitious
  concurrency/context settings have not been validated by these measurements.

## Provenance

The release was prepared from local NInfer Git commit
`901248fe914e49841a3828d7f8e16be77eb4f1a5` with a curated Q8 Indexed88 direct-path
source snapshot. Experimental Q5/Q6 external-head code and pre-stage backup files are intentionally
excluded. The GitHub release tag identifies the curated source revision.
