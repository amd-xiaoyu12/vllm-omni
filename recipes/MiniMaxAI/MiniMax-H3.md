# MiniMax H3

> Joint video and audio generation with text, first-frame, image/audio, or
> multi-video conditions

## Summary

- Vendor: MiniMaxAI
- Model: [`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- Tasks: T2VA, FL2VA, and Ref2VA
- Mode: OpenAI-compatible `/v1/videos` HTTP serving
- Maintainer: Community

MiniMax H3 is a CFG-distilled joint video/audio diffusion transformer. Its
checkpoint has two independently served partitions:

- `FL2VA`: text-to-video+audio (`t2va`) and first-frame-to-video+audio
  (`fl2va`)
- `Ref2VA`: image+audio or one-or-more-video reference-to-video+audio
  (`ref2va`)

The generated MP4 contains H.264 video and synchronized stereo audio.

## Prerequisites

The checkpoint requires Hugging Face access approval. Download it and point
`MODEL_ROOT` at the directory containing the `FL2VA` and `Ref2VA`
subdirectories:

```bash
hf auth login
export MODEL_ROOT=/path/to/MiniMax-H3
hf download MiniMaxAI/MiniMax-H3 --local-dir "${MODEL_ROOT}"
```

Install vLLM-Omni from the checkout containing MiniMax H3 support. On
Blackwell, install the optional FlashAttention-4 dependency:

```bash
uv venv
source .venv/bin/activate
uv pip install -e '.[fa4]'
```

On AMD ROCm, install without the `[fa4]` extra (FA4 is CUDA-only) and use
`--diffusion-attention-backend FLASH_ATTN`; see
[AMD ROCm (gfx942 / gfx950)](#amd-rocm-gfx942--gfx950).

`ffmpeg` and `ffprobe` must be available on `PATH`. They are used for
reference-video preparation and MP4 output.

## Start a server

One server loads one checkpoint partition. Set `MODEL` to `FL2VA` for T2VA
and FL2VA requests, or to `Ref2VA` for either Ref2VA request.

### Single GPU: accuracy and memory first

The single-GPU configuration uses model-level CPU offload.
This matches the accuracy-qualified reference path and prevents the Qwen3-VL
encoder and DiT from being resident on the GPU at the same time.

```bash
export MODEL="${MODEL_ROOT}/FL2VA"
export PORT=8091

CUDA_VISIBLE_DEVICES=0 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
vllm serve "${MODEL}" \
  --omni \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --trust-remote-code \
  --num-gpus 1 \
  --enable-cpu-offload \
  --diffusion-attention-backend FLASH_ATTN
```

Use a GPU with enough memory for the active H3 component and enough system RAM
for the offloaded components. CPU offload reduces GPU memory pressure but adds
PCIe/NVLink transfer latency.

### AMD ROCm (gfx942 / gfx950)

MiniMax H3 runs on AMD Instinct GPUs (gfx942 / gfx950) in BF16. Use
`--diffusion-attention-backend FLASH_ATTN`, which resolves to AITER packed varlen
attention on both architectures.

Select the device with `HIP_VISIBLE_DEVICES` (not `CUDA_VISIBLE_DEVICES`), drop the
CUDA-only `FLASHINFER_DISABLE_VERSION_CHECK`, and install without the `[fa4]` extra
(FA4 is CUDA-only). The VAE uses AITER GroupNorm on ROCm.

Install (ROCm wheel + source vLLM-Omni):

```bash
pip install "vllm==0.26.0+rocm723" \
  --extra-index-url https://wheels.vllm.ai/rocm/0.26.0/rocm723
VLLM_OMNI_TARGET_DEVICE=rocm pip install -e . --no-build-isolation
```

Prebuilt image: `vllm/vllm-omni-rocm:minimax-h3`. Note that image is built from a
pre-merge commit, so T2VA and FL2VA work, but **image+audio Ref2VA lacks the
soundfile fallback** (fails without TorchCodec) and **video-reference Ref2VA needs
`ffmpeg`**, which is not bundled. Build `docker/Dockerfile.rocm` from `v0.26.0` (or
newer) for the complete Ref2VA paths.

#### Single GPU

Single GPU with model-level CPU offload keeps the Qwen3-VL encoder and DiT from
being co-resident:

```bash
export MODEL="${MODEL_ROOT}/FL2VA"
export PORT=8091

HIP_VISIBLE_DEVICES=0 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
vllm serve "${MODEL}" \
  --omni --host 0.0.0.0 --port "${PORT}" --trust-remote-code \
  --num-gpus 1 --enable-cpu-offload \
  --diffusion-attention-backend FLASH_ATTN
```

#### Four GPUs

The best-practice CUDA four-GPU configuration works on ROCm with the changes above.
It mirrors the CUDA command including Ulysses sequence parallelism, VAE patch
parallelism, and text-encoder tensor parallelism, with no CPU offload when the model
shards across the GPUs:

```bash
export MODEL="${MODEL_ROOT}/FL2VA"
export PORT=8091

HIP_VISIBLE_DEVICES=0,1,2,3 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
vllm serve "${MODEL}" \
  --omni \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --trust-remote-code \
  --num-gpus 4 \
  --usp 4 \
  --ring 1 \
  --text-encoder-tp-size 4 \
  --vae-patch-parallel-size 4 \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --diffusion-attention-backend FLASH_ATTN
```

As on CUDA, H3 is CFG-distilled, so keep `--cfg-parallel-size` at 1, and the H3 VAE
supports only its native `tile` mode. `--text-encoder-tp-size` is validated on
gfx942; on gfx950 it was not exercised.

#### Validated ROCm evidence

vLLM-Omni with MiniMax H3 support, BF16. gfx942 rows measured with the
`vllm/vllm-omni-rocm:minimax-h3` image; gfx950 rows measured with the
`0.26.0+rocm723` wheel (HIP 7.2).

| Workload | Configuration | Observed result |
|----------|---------------|-----------------|
| T2VA, 1344x768, 209 frames, 50 steps | 4x gfx942 (MI300X), FLASH_ATTN, USP4, text-enc TP4, VAE PP4 tile | encode 0.09 s, denoise 244.04 s, decode 4.15 s, 267.42 s client E2E; H.264 24 FPS + 32 kHz stereo AAC |
| FL2VA, 1344x768, 209 frames, 50 steps | 4x gfx942 (MI300X), FLASH_ATTN, USP4, text-enc TP4, VAE PP4 tile | encode 13.98 s, denoise 257.58 s, decode 4.11 s, 287.07 s client E2E; H.264 24 FPS + 32 kHz stereo AAC |
| T2VA, 832x480, ~4 s, 40 steps | 1x gfx950 (MI350), FLASH_ATTN, CPU offload | valid MP4 (H.264 + synced audio); ~0.73 s/denoise-step (~1.37 it/s), ~55 s client E2E incl. warmup |

gfx942 figures are the mean of three requests after one excluded warmup
(external evidence: vllm-project/recipes#732). gfx950 figures are
functional-correctness validations, not tuned throughput; the first request
includes lazy regional compilation. MI325X (gfx942) and other MI355X SKUs are not
listed until their own evidence is added.

### Four GPUs: measured best-practice throughput

The best validated four-GPU configuration on four NVIDIA B300 GPUs is:

- no CPU or layerwise offload;
- Ulysses sequence parallelism degree 4;
- native tiled VAE patch parallelism degree 4;
- regional `torch.compile` for the repeated DiT blocks;
- FlashAttention, with Ring and TP left at 1.

```bash
export MODEL="${MODEL_ROOT}/FL2VA"
export PORT=8091

CUDA_VISIBLE_DEVICES=0,1,2,3 \
FLASHINFER_DISABLE_VERSION_CHECK=1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
vllm serve "${MODEL}" \
  --omni \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --trust-remote-code \
  --num-gpus 4 \
  --usp 4 \
  --ring 1 \
  --vae-patch-parallel-size 4 \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --diffusion-attention-backend FLASH_ATTN
```

Do not add `--enforce-eager` to this performance configuration. The first
request includes regional compilation; warm the server once before measuring
steady-state latency. H3 is CFG-distilled, so `--cfg-parallel-size` must remain
1. The H3 VAE supports its native `tile` mode, not
`spatial_shard_height` or `spatial_shard_width`.

### Text encoder tensor parallelism

The Qwen3-VL text encoder (~51.5 GB in BF16 for the retained 50 layers) is by
default fully resident on the DiT main rank.  On multi-GPU no-offload runs that
rank becomes the peak-memory hotspot.  Add `--text-encoder-tp-size N` to shard
the encoder across the first `N` DiT ranks (the encoder is implemented with
vLLM-style tensor-parallel layers and runs with distributed collectives over
its own encoder process group):

```bash
export MODEL="${MODEL_ROOT}/FL2VA"
export PORT=8091

CUDA_VISIBLE_DEVICES=0,1,2,3 \
FLASHINFER_DISABLE_VERSION_CHECK=1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
vllm serve "${MODEL}" \
  --omni \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --trust-remote-code \
  --num-gpus 4 \
  --usp 4 \
  --ring 1 \
  --text-encoder-tp-size 4 \
  --vae-patch-parallel-size 4 \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --diffusion-attention-backend FLASH_ATTN
```

`N` must divide the Qwen3-VL head counts (64 attention heads / 8 KV heads), so
valid values on a 4-GPU server are 1, 2, and 4 (1, 2, 4, 8 on 8 GPUs).  The
encoder TP rank set is the first `N` DiT ranks; on 4 GPUs
`--text-encoder-tp-size 4` shards the encoder 4-way, dropping the DiT main
rank's no-offload peak by roughly `(N-1)/N` of the ~51.5 GB encoder while the
other ranks each gain `~51.5/N` GB.  The encoder output remains identical to
the reference path within bf16 rounding: every encoder rank all-reduces the
row-parallel projections, so the full `[seq, 5120]` layer-50 hidden state is
replicated on every rank.

To serve Ref2VA, stop the FL2VA server and restart the same one- or four-GPU
command with:

```bash
export MODEL="${MODEL_ROOT}/Ref2VA"
```

### Online FP8 quantization

MiniMax H3 supports load-time FP8 quantization of the DiT. The checkpoint
remains BF16 on disk; vLLM-Omni quantizes eligible weights while loading and
uses dynamic activation scaling during inference. By default, attention and
MLP linears in the token refiner and main DiT blocks, the condition
projection, and all AdaLN projections use FP8. Patch, timestep, and final
projections remain FP32; the text encoder and VAEs are unchanged.

Add this option to an existing H3 server command:

```bash
--quantization fp8
```

Use `ignored_layers` to keep any otherwise eligible linear in BF16. H3
resolves the `transformer` component before constructing the DiT, so names do
not start with `transformer.`. Entries are exact runtime linear prefixes; a
parent name such as `blocks.0.attn` does not exclude its children.

Eligible names are the `attn.qkv_proj`, `attn.out_proj`, `mlp.fc1`, and
`mlp.fc2` children under `token_refiner.blocks.<0-1>` or `blocks.<0-49>`.
The other eligible names are `condition_proj`,
`blocks.<0-49>.adaln_proj.linear`, and `final_layer.adaln_proj.linear`.
For example, keep the first main block's attention projections in BF16 with:

```bash
--diffusion-quantization-config \
  '{"transformer":{"method":"fp8","ignored_layers":["blocks.0.attn.qkv_proj","blocks.0.attn.out_proj"]}}'
```

The structured option replaces `--quantization fp8`. Online FP8 is currently
incompatible with H3 layerwise offload because the offload path produces a
weight stride rejected by the Cutlass FP8 kernel. Use resident FP8 with tensor
parallelism and VAE tiling instead.

## HTTP API examples

The following requests use the synchronous endpoint so the returned body can
be saved directly as an MP4. The asynchronous `POST /v1/videos` endpoint can
also be used when job polling is preferred.

All four tasks use 24 FPS, 50 sigma points, seed values from the validated
workloads, and the checkpoint-reference video/audio flow shifts of 12 and 3.
Decimal durations are passed through `extra_params`.

Set the endpoint once:

```bash
export API_URL="http://127.0.0.1:${PORT}/v1/videos/sync"
```

### 1. T2VA: text to video and audio

Run this request against the `FL2VA` partition:

```bash
curl -sS -X POST "${API_URL}" \
  -F 'prompt=In a snowy blue-purple forest, Ori carefully walks past a sleeping giant; footsteps crunch in the snow while the creature breathes and softly snorts.' \
  -F 'width=1344' \
  -F 'height=768' \
  -F 'fps=24' \
  -F 'num_inference_steps=50' \
  -F 'flow_shift=12' \
  -F 'seed=1101' \
  -F 'extra_params={"task":"t2va","duration":8.7,"audio_flow_shift":3.0}' \
  -o t2va.mp4
```

### 2. FL2VA: first frame to video and audio

Run this request against the `FL2VA` partition. When width and height are
omitted, H3 preserves the first-frame aspect ratio and uses a 768-pixel short
edge.

```bash
export FIRST_FRAME=/path/to/fl2va_first_frame.png

curl -sS -X POST "${API_URL}" \
  -F 'prompt=A man stands beside a yellow car at night. The car drives away; he follows it with his eyes and begins singing sadly, with synchronized voice and city ambience.' \
  -F 'fps=24' \
  -F 'num_inference_steps=50' \
  -F 'flow_shift=12' \
  -F 'seed=2101' \
  -F 'extra_params={"task":"fl2va","duration":8.7,"audio_flow_shift":3.0}' \
  -F "input_reference=@${FIRST_FRAME};type=image/png" \
  -o fl2va.mp4
```

### 3. Ref2VA: image and audio reference

Run this request against the `Ref2VA` partition. `audio_reference` accepts an
HTTP(S) URL or a `data:` URL. In one terminal, expose the local reference
assets to the serving host:

```bash
python -m http.server 8092 \
  --bind 127.0.0.1 \
  --directory /path/to/reference_assets
```

Then submit the image and audio request from another terminal:

```bash
export REF_IMAGE=/path/to/reference_assets/ref2va_image.png
export AUDIO_URL=http://127.0.0.1:8092/ref2va_audio.mp3

curl -sS -X POST "${API_URL}" \
  -F 'prompt=A white cat with black mustache and eyebrow markings sits on a beige couch, lip-syncing precisely to the complete reference audio before shifting from confusion to deadpan speechlessness.' \
  -F 'width=1344' \
  -F 'height=768' \
  -F 'fps=24' \
  -F 'num_inference_steps=50' \
  -F 'flow_shift=12' \
  -F 'seed=3101' \
  -F 'extra_params={"task":"ref2va","duration":15.0,"audio_flow_shift":3.0}' \
  -F "input_reference=@${REF_IMAGE};type=image/png" \
  -F "audio_reference={\"audio_url\":\"${AUDIO_URL}\"}" \
  -o ref2va_image_audio.mp4
```

The requested duration should cover the complete audio. If `duration` is
shorter, the reference soundtrack is truncated to the generated clip.

### 4. Ref2VA: two video references

Run this request against the `Ref2VA` partition. Repeat the
`input_references` multipart field once per source video. H3 consumes the
videos in form order and preserves their original soundtracks during
conditioning.

```bash
export SUBJECT_VIDEO=/path/to/green_screen_subject.mp4
export BACKGROUND_VIDEO=/path/to/fairytale_background.mov

curl -sS -X POST "${API_URL}" \
  -F 'prompt=Remove the green screen background of Video 1 and replace it with the fairytale environment from Video 2. Match the background motion to the character actions and relight the character to fit the scene.' \
  -F 'width=1344' \
  -F 'height=768' \
  -F 'fps=24' \
  -F 'num_inference_steps=50' \
  -F 'flow_shift=12' \
  -F 'seed=3101' \
  -F 'extra_params={"task":"ref2va","duration":15.0,"audio_flow_shift":3.0}' \
  -F "input_references=@${SUBJECT_VIDEO};type=video/mp4" \
  -F "input_references=@${BACKGROUND_VIDEO};type=video/quicktime" \
  -o ref2va_video_video.mp4
```

The server stores uploaded multi-video references only for the lifetime of the
request and deletes the temporary files after generation. Video Ref2VA uses
the source-video soundtracks and does not accept a separate
`audio_reference`.

## Key parameters

| Parameter | Recommended value | Notes |
|-----------|-------------------|-------|
| `task` | `t2va`, `fl2va`, or `ref2va` | Passed in `extra_params`; must match the served partition |
| `duration` | Workload-specific | Decimal seconds in `extra_params`; converted to H3-compatible frame count |
| `fps` | `24` | H3 output FPS is fixed |
| `num_inference_steps` | `50` | Matches the reference accuracy workloads |
| `flow_shift` | `12` | Video sigma shift |
| `audio_flow_shift` | `3` | Audio sigma shift, passed in `extra_params` |
| `seed` | Task-specific | Use a fixed value for reproducibility |
| `width`, `height` | Multiples of 32 | Aspect ratio must be between 1:4 and 4:1 |

## Validated four-GPU evidence

The four-GPU recommendation was measured on four NVIDIA B300 GPUs with one
excluded warmup followed by three requests.

| Workload | Configuration | Observed result |
|----------|---------------|-----------------|
| FL2VA, 209 frames, 1248x768 | no offload, U4, VPP4 tile, regional compile | 86.964 s mean HTTP client latency |
| Two-video Ref2VA, 362 frames, 1344x768 | no offload, U4, VPP4 tile, regional compile | 784.394 s accounted model-stage mean |

These measurements describe the validated shapes rather than a general
throughput guarantee. Multi-video Ref2VA is much slower because the two
reference videos expand both the Qwen3-VL vision sequence and the packed DiT
attention sequence.

## Validated FP8 evidence

With eager DiT/text-encoder TP2 and VAE tiling, the 384x672, 107-frame,
10-step quality case measured LPIPS 0.1156 (limit 0.20), PSNR 23.6316 dB,
audio spectral cosine 0.9589 (minimum 0.80), and audio RMS ratio 0.9342. The
resident per-GPU peak was 68.52 GiB for BF16 and 53.51 GiB for FP8, a 22%
reduction.

## Known limitations

- Each server process loads only one checkpoint partition.
- H3 currently executes one generation request per diffusion batch.
- The first regional-compile request is a warmup and should not be included in
  steady-state performance measurements.
- Online FP8 is not compatible with layerwise offload.
- Image+audio Ref2VA accepts exactly one image and one audio reference.
- Video Ref2VA accepts one or more video files, but not an additional standalone
  audio reference.
- VAE patch parallelism requires size 1 or the full DiT group size and supports
  the H3 native `tile` mode only.

## Additional resources

- [Supported models](../../docs/models/supported_models.md)
- [Video API](../../docs/serving/videos_api.md)
- [Diffusion parallelism](../../docs/user_guide/diffusion/parallelism/overview.md)
