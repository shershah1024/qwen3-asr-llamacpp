#!/usr/bin/env python3
"""
WER benchmark for Qwen3-ASR (llama.cpp) on LibriSpeech test-clean.

Uses streaming mode to avoid downloading the full dataset upfront.

Usage:
    python bench-asr.py [--limit N] [--dataset test-clean|test-other]

Requires: jiwer, datasets, soundfile
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time

import numpy as np
import soundfile as sf
from datasets import load_dataset
from jiwer import wer, cer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LLAMA_MTMD_CLI = os.environ.get(
    "LLAMA_MTMD_CLI",
    os.path.expanduser("~/projects/llama.cpp/build/bin/llama-mtmd-cli"),
)
TEXT_MODEL = os.environ.get(
    "TEXT_MODEL",
    os.path.expanduser("~/models/Qwen3-ASR-0.6B-text-Q4_K_M.gguf"),
)
MMPROJ_MODEL = os.environ.get(
    "MMPROJ_MODEL",
    os.path.expanduser("~/models/Qwen3-ASR-0.6B-mmproj.gguf"),
)


def write_wav(path: str, audio_array, sample_rate: int = 16000):
    """Write a numpy array as a 16-bit PCM WAV file."""
    sf.write(path, audio_array, sample_rate, subtype="PCM_16")


def extract_transcript(raw_output: str) -> str:
    """
    Extract the transcription text from llama-mtmd-cli output.
    The model outputs: language English<asr_text>Actual transcript here.
    """
    # Find everything after <asr_text> tag
    m = re.search(r"<asr_text>\s*(.*)", raw_output, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fallback: return everything after the assistant prompt
    lines = raw_output.strip().split("\n")
    for line in reversed(lines):
        line = line.strip()
        if line and not line.startswith(("[", "llama_", "clip_", "ggml_", "load", "print_info")):
            return line
    return raw_output.strip()


def normalize_text(text: str) -> str:
    """Normalize text for WER computation: lowercase, remove punctuation."""
    text = text.lower()
    # Remove common ASR artifacts
    text = re.sub(r"<\|[^|]+\|>", "", text)  # remove special tokens
    text = re.sub(r"[^\w\s']", "", text)  # keep only words, spaces, apostrophes
    text = re.sub(r"\s+", " ", text).strip()
    return text


def run_qwen3_asr(wav_path: str, timeout: int = 120) -> tuple[str, float]:
    """
    Run Qwen3-ASR via llama-mtmd-cli on a single WAV file.
    Returns (transcript, inference_time_seconds).
    """
    cmd = [
        LLAMA_MTMD_CLI,
        "-m", TEXT_MODEL,
        "--mmproj", MMPROJ_MODEL,
        "--no-mmproj-offload",
        "--audio", wav_path,
        "-p", " ",
        "--no-warmup",
        "--temp", "0",
        "-ngl", "99",
        "-c", "4096",
        "-n", "256",
    ]

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = time.time() - t0
        return result.stdout.strip(), elapsed
    except subprocess.TimeoutExpired:
        return "", time.time() - t0


def main():
    parser = argparse.ArgumentParser(description="WER benchmark for Qwen3-ASR")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of samples (0=all)")
    parser.add_argument("--dataset", default="test.clean", help="LibriSpeech split: test.clean or test.other")
    parser.add_argument("--output", default="bench-asr-results.jsonl", help="Output JSONL file")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file")
    args = parser.parse_args()

    # Check binaries exist
    for name, path in [("llama-mtmd-cli", LLAMA_MTMD_CLI), ("text model", TEXT_MODEL), ("mmproj", MMPROJ_MODEL)]:
        if not os.path.exists(path):
            print(f"ERROR: {name} not found at {path}")
            sys.exit(1)

    # Load already-completed sample IDs if resuming
    done_ids = set()
    done_refs = []
    done_hyps = []
    done_times = []
    if args.resume and os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                rec = json.loads(line)
                done_ids.add(rec["id"])
                done_refs.append(rec["ref"])
                done_hyps.append(rec["hyp"])
                done_times.append(rec["inference_s"])
        print(f"Resuming: {len(done_ids)} samples already done")

    # Use streaming to avoid downloading full dataset
    # Disable automatic audio decoding — we'll decode with soundfile ourselves
    print(f"Loading LibriSpeech {args.dataset} (streaming)...")
    from datasets import Audio
    ds = load_dataset("openslr/librispeech_asr", split=args.dataset, streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    references = list(done_refs)
    hypotheses = list(done_hyps)
    times = list(done_times)
    errors = 0
    processed = 0
    skipped = 0

    out_mode = "a" if args.resume else "w"
    with open(args.output, out_mode) as fout:
        for sample in ds:
            sample_id = sample["id"]

            if sample_id in done_ids:
                skipped += 1
                continue

            processed += 1
            if args.limit > 0 and processed > args.limit:
                break

            ref_text = sample["text"]
            audio = sample["audio"]

            # audio is {"bytes": b"...", "path": "..."} when decode=False
            # Decode FLAC/audio bytes with soundfile
            import io
            audio_data, sr = sf.read(io.BytesIO(audio["bytes"]))
            audio_array = np.array(audio_data, dtype=np.float32)

            # Write temp WAV at 16kHz
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            write_wav(tmp_path, audio_array, sr)

            duration = len(audio_array) / sr

            # Run inference
            raw_output, elapsed = run_qwen3_asr(tmp_path)
            os.unlink(tmp_path)

            # Extract and normalize
            hypothesis = extract_transcript(raw_output)
            ref_norm = normalize_text(ref_text)
            hyp_norm = normalize_text(hypothesis)

            # Per-sample WER
            sample_wer = wer(ref_norm, hyp_norm) if ref_norm else 0.0

            references.append(ref_norm)
            hypotheses.append(hyp_norm)
            times.append(elapsed)

            if sample_wer > 0.5:
                errors += 1

            result = {
                "id": sample_id,
                "ref": ref_norm,
                "hyp": hyp_norm,
                "wer": round(sample_wer, 4),
                "duration_s": round(duration, 2),
                "inference_s": round(elapsed, 2),
                "rtf": round(elapsed / duration, 3) if duration > 0 else 0,
            }
            fout.write(json.dumps(result) + "\n")
            fout.flush()

            # Progress
            running_wer = wer(references, hypotheses) if references else 0
            rtf = elapsed / duration if duration > 0 else 0
            status = "OK" if sample_wer < 0.1 else "WARN" if sample_wer < 0.5 else "BAD"
            limit_str = f"/{args.limit}" if args.limit > 0 else ""
            print(
                f"[{processed}{limit_str}] {status} wer={sample_wer:.1%} "
                f"running_wer={running_wer:.2%} "
                f"rtf={rtf:.1f}x | {ref_norm[:60]}"
            )

    # Final stats
    if references:
        new_count = processed
        total_count = len(references)
        total_wer = wer(references, hypotheses)
        total_cer = cer(references, hypotheses)
        avg_time = sum(times) / len(times)
        total_audio = sum(r["duration_s"] for r in [json.loads(l) for l in open(args.output)] if "duration_s" in r) if os.path.exists(args.output) else 0

        print(f"\n{'='*60}")
        print(f"Dataset:       LibriSpeech {args.dataset}")
        print(f"Samples:       {total_count} ({new_count} new)")
        print(f"Model:         Qwen3-ASR-0.6B Q4_K_M (llama.cpp)")
        print(f"WER:           {total_wer:.2%}")
        print(f"CER:           {total_cer:.2%}")
        print(f"Avg inference: {avg_time:.1f}s per sample")
        print(f"High-error:    {errors} samples (>50% WER)")
        print(f"Results:       {args.output}")
        print(f"{'='*60}")
    else:
        print("No samples processed.")


if __name__ == "__main__":
    main()
