# Qwen3-ASR for llama.cpp

Adds [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) (0.6B and 1.7B) to llama.cpp's multimodal pipeline (`mtmd`).

## Background

Qwen3-ASR is an encoder-decoder ASR model from Alibaba. The audio encoder produces embeddings that are fed into a Qwen3 text decoder. llama.cpp already handles the Qwen3 decoder, but had no support for the Qwen3-ASR audio encoder.

The encoder cannot reuse llama.cpp's existing Whisper audio path because of three architectural differences:

1. **Chunked convolution with per-chunk positional embeddings.** The mel spectrogram is split into 100-frame chunks. Each chunk is independently downsampled through 3x Conv2d(stride=2) and gets its own sinusoidal positional embeddings starting from position 0. Whisper applies positional embeddings globally.

2. **Two-level inference windows.** Multiple conv chunks (8 by default, `n_window_infer=800`) are grouped into an inference window. The transformer runs full bidirectional attention across all chunks within a window, but not across windows. This required restructuring the encoder graph to loop over sub-chunks for conv, concatenate, then run a single transformer pass.

3. **Split encoder/decoder files.** The audio encoder is a separate GGUF (mmproj), following the same pattern as vision models (LLaVA, Qwen2-VL) rather than Whisper's monolithic format.

This repo contains the llama.cpp patch (~350 lines across 10 files), the new encoder graph builder, GGUF conversion scripts, and links to pre-converted models.

## Architecture

```
Audio WAV -> Mel spectrogram (128 bins)
  -> Split into inference windows (800 mel frames = 8 conv chunks)
    -> Per conv chunk (100 frames):
        3x Conv2d(stride=2, padding=1) + GELU
        Flatten + linear projection
        Per-chunk sinusoidal positional embeddings
    -> Concatenate chunks within window
    -> Transformer encoder (full attention within window)
    -> Post-LayerNorm
    -> Output projector (Linear + GELU + Linear)
  -> Qwen3 text decoder
```

| | 0.6B | 1.7B |
|---|---|---|
| Encoder layers | 18 | 24 |
| Attention heads | 14 | 16 |
| d_model | 896 | 1024 |
| Output dim | 1024 | 2048 |

## Results

LibriSpeech test-clean, 50 samples (mean audio length 6.6s, range 1.9–23.3s). Apple M3 MacBook Air 8GB.

| Model | WER | Size (text + mmproj) | Avg latency |
|-------|-----|----------------------|-------------|
| 0.6B Q4_K_M | 3.06% | 462 + 361 MB | 2.6 s |
| 0.6B FP16 | 3.06% | 1.4 GB + 361 MB | 2.6 s |
| 1.7B Q4_K_M | 3.28% | 1.2 GB + 612 MB | 2.6 s |
| 1.7B Q8_0 | 2.84% | 2.0 GB + 612 MB | 3.1 s |
| 1.7B FP16 | 2.84% | 3.8 GB + 612 MB | 29.4 s |

Published reference WER (Python, full precision): 0.6B = 2.11%, 1.7B = 1.63%.

Observations:
- Q4_K_M degrades the 1.7B (3.28% vs 2.84%) but not the 0.6B. The 1.7B is more quantization-sensitive.
- Q8_0 and FP16 produce identical WER on the 1.7B. Q8_0 is 10x faster.
- M1 MacBook Pro 8GB: 3.89s on a 3.5s sample (vs 2.14s on M3).

## Pre-converted GGUF models

```bash
# 0.6B
wget https://examaudio.tslfiles.org/models/qwen3-asr/Qwen3-ASR-0.6B-text-Q4_K_M.gguf
wget https://examaudio.tslfiles.org/models/qwen3-asr/Qwen3-ASR-0.6B-mmproj.gguf

# 1.7B
wget https://examaudio.tslfiles.org/models/qwen3-asr/Qwen3-ASR-1.7B-text-Q8_0.gguf
wget https://examaudio.tslfiles.org/models/qwen3-asr/Qwen3-ASR-1.7B-mmproj.gguf
```

## Build

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout 2634ed2  # tested base commit

git apply /path/to/qwen3-asr-llama-cpp.patch
cp /path/to/qwen3asr-enc.cpp tools/mtmd/models/

cmake -B build -DGGML_METAL=ON  # -DGGML_CUDA=ON for NVIDIA
cmake --build build -j
```

## Usage

```bash
# CLI
./build/bin/llama-mtmd-cli \
  -m Qwen3-ASR-0.6B-text-Q4_K_M.gguf \
  --mmproj Qwen3-ASR-0.6B-mmproj.gguf \
  --audio input.wav \
  -ngl 99 -c 4096 -n 256 --temp 0

# HTTP server (OpenAI-compatible /v1/chat/completions)
./build/bin/llama-server \
  -m Qwen3-ASR-1.7B-text-Q8_0.gguf \
  --mmproj Qwen3-ASR-1.7B-mmproj.gguf \
  -ngl 99 -c 4096 --port 8080
```

Input: 16kHz mono WAV.

## Files

| File | Description |
|------|-------------|
| `qwen3-asr-llama-cpp.patch` | Patch for llama.cpp (10 modified files, ~350 lines) |
| `qwen3asr-enc.cpp` | Audio encoder graph builder. Copy to `tools/mtmd/models/` |
| `convert_qwen3_asr_to_gguf.py` | Converts HuggingFace audio encoder to GGUF mmproj |
| `bench-asr.py` | WER benchmark on LibriSpeech |
| `verify_numerical.py` | Intermediate tensor extraction for numerical verification |

### Patch summary

| File | Changes |
|------|---------|
| `convert_hf_to_gguf.py` | Qwen3ASR model class |
| `gguf-py/gguf/constants.py` | Projector type enum |
| `gguf-py/gguf/tensor_mapping.py` | Conv/audio tensor mappings |
| `tools/mtmd/CMakeLists.txt` | Build target |
| `tools/mtmd/clip-impl.h` | Constants and metadata keys |
| `tools/mtmd/clip-model.h` | Conv2d tensor slots, hparams |
| `tools/mtmd/clip.cpp` | Loading, dispatch, token count |
| `tools/mtmd/models/models.h` | Graph struct |
| `tools/mtmd/mtmd-audio.cpp` | Mel preprocessing, window chunking |
| `tools/mtmd/mtmd.cpp` | Routing |

## Convert from HuggingFace

```bash
# Download
huggingface-cli download Qwen/Qwen3-ASR-0.6B --local-dir ./Qwen3-ASR-0.6B

# Text decoder
python llama.cpp/convert_hf_to_gguf.py ./Qwen3-ASR-0.6B --outfile text.gguf --outtype f16
./llama.cpp/build/bin/llama-quantize text.gguf text-Q4_K_M.gguf Q4_K_M
# Use Q8_0 for 1.7B

# Audio encoder (mmproj)
pip install safetensors numpy
python convert_qwen3_asr_to_gguf.py --model-dir ./Qwen3-ASR-0.6B --output mmproj.gguf
```

## Run benchmarks

```bash
pip install jiwer datasets soundfile numpy

TEXT_MODEL=./text.gguf MMPROJ_MODEL=./mmproj.gguf \
python bench-asr.py --limit 50 --output results.jsonl
```

## Tested

- macOS 15, Apple Silicon (M1, M3)
- llama.cpp `2634ed2`
- Qwen3-ASR-0.6B, Qwen3-ASR-1.7B

## License

Patch: MIT (llama.cpp). Model weights: Apache 2.0 ([Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)).
