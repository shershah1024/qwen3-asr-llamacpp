#!/usr/bin/env python3
"""
Numerical verification: run test audio through PyTorch Qwen3-ASR audio encoder,
extract intermediate tensors, save as numpy for comparing against C++ ggml output.

Usage:
    pip install qwen-asr torch numpy soundfile
    python verify_numerical.py --audio test.wav --output-dir ./reference_tensors/

Saves:
    mel_input.npy          — mel spectrogram input [n_mel, n_frames]
    after_conv2d_1.npy     — after first conv + gelu
    after_conv2d_2.npy
    after_conv2d_3.npy
    after_conv_out.npy     — after linear projection
    after_pos_embd.npy     — after adding positional embeddings
    after_layer_{i}.npy    — after each encoder layer
    after_ln_post.npy      — after post layer norm
    after_proj1.npy        — after first projector linear + gelu
    after_proj2.npy        — final output
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def load_audio(audio_path: str, sr: int = 16000) -> np.ndarray:
    """Load audio file and resample to target sample rate."""
    import soundfile as sf
    audio, orig_sr = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]  # mono
    if orig_sr != sr:
        # Simple resampling
        duration = len(audio) / orig_sr
        n_samples = int(duration * sr)
        audio = np.interp(
            np.linspace(0, len(audio) - 1, n_samples),
            np.arange(len(audio)),
            audio,
        )
    return audio


def compute_mel(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Compute mel spectrogram using WhisperFeatureExtractor."""
    from transformers import WhisperFeatureExtractor
    fe = WhisperFeatureExtractor(
        feature_size=128,
        sampling_rate=sr,
        hop_length=160,
        n_fft=400,
        padding_value=0.0,
    )
    features = fe(audio, sampling_rate=sr, return_tensors="np")
    return features["input_features"][0]  # [n_mel, n_frames]


def extract_intermediates(model_id: str, audio_path: str, output_dir: str):
    """Run audio through encoder with hooks, save intermediates."""
    os.makedirs(output_dir, exist_ok=True)

    # Load audio and compute mel
    audio = load_audio(audio_path)
    mel = compute_mel(audio)
    np.save(os.path.join(output_dir, "mel_input.npy"), mel)
    print(f"Mel spectrogram: {mel.shape}")

    # Load model
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    thinker_config = config.thinker_config

    # We need to load just the audio tower, not the full model
    # Import the modeling code
    from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
        Qwen3ASRAudioEncoder,
        Qwen3ASRForConditionalGeneration,
        _get_feat_extract_output_lengths,
    )

    print(f"Loading {model_id}...")
    full_model = Qwen3ASRForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float32,  # f32 for numerical precision
        device_map="cpu",
    )
    audio_tower = full_model.thinker.audio_tower
    audio_tower.eval()

    # Prepare input
    mel_tensor = torch.tensor(mel, dtype=torch.float32)  # [n_mel, n_frames]
    feature_lens = torch.tensor([mel.shape[1]], dtype=torch.long)

    # Run forward with intermediate extraction
    with torch.no_grad():
        n_window = audio_tower.n_window
        n_mel = mel_tensor.shape[0]
        n_frames = mel_tensor.shape[1]

        aftercnn_lens = _get_feat_extract_output_lengths(feature_lens)
        chunk_num = torch.ceil(feature_lens / (n_window * 2)).long()

        chunk_lengths = torch.tensor(
            [n_window * 2] * chunk_num.sum(),
            dtype=torch.long,
        )
        tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
        chunk_lengths[tail_chunk_index] = feature_lens % (n_window * 2)
        chunk_lengths[chunk_lengths == 0] = n_window * 2

        # Split and pad
        input_features = mel_tensor  # [n_mel, n_frames]
        chunk_list = input_features.T.split(chunk_lengths.tolist(), dim=0)
        padded_feature = torch.nn.utils.rnn.pad_sequence(chunk_list, batch_first=True).transpose(1, 2)
        feature_lens_after_cnn = _get_feat_extract_output_lengths(chunk_lengths)
        padded_mask_after_cnn = torch.nn.utils.rnn.pad_sequence(
            [torch.ones(length, dtype=torch.bool) for length in feature_lens_after_cnn],
            batch_first=True,
        )

        padded_feature = padded_feature.unsqueeze(1)  # add channel dim

        # Conv2d block
        conv_out = F.gelu(audio_tower.conv2d1(padded_feature))
        np.save(os.path.join(output_dir, "after_conv2d_1.npy"), conv_out.numpy())
        print(f"After conv2d1: {conv_out.shape}")

        conv_out = F.gelu(audio_tower.conv2d2(conv_out))
        np.save(os.path.join(output_dir, "after_conv2d_2.npy"), conv_out.numpy())
        print(f"After conv2d2: {conv_out.shape}")

        conv_out = F.gelu(audio_tower.conv2d3(conv_out))
        np.save(os.path.join(output_dir, "after_conv2d_3.npy"), conv_out.numpy())
        print(f"After conv2d3: {conv_out.shape}")

        # Flatten + linear
        b, c, f_dim, t = conv_out.size()
        conv_out = audio_tower.conv_out(
            conv_out.permute(0, 3, 1, 2).contiguous().view(b, t, c * f_dim)
        )
        np.save(os.path.join(output_dir, "after_conv_out.npy"), conv_out.numpy())
        print(f"After conv_out: {conv_out.shape}")

        # Positional embeddings
        pos_embd = audio_tower.positional_embedding.positional_embedding[:conv_out.shape[1], :]
        conv_out = conv_out + pos_embd.unsqueeze(0).to(conv_out.dtype)
        np.save(os.path.join(output_dir, "after_pos_embd.npy"), conv_out.numpy())
        print(f"After pos embed: {conv_out.shape}")

        # Unpad
        hidden_states = conv_out[padded_mask_after_cnn]

        # Build cu_seqlens
        cu_chunk_lens = [0]
        window_aftercnn = padded_mask_after_cnn.shape[-1] * (audio_tower.n_window_infer // (n_window * 2))
        for cnn_len in aftercnn_lens:
            cu_chunk_lens += [window_aftercnn] * (cnn_len // window_aftercnn)
            remainder = cnn_len % window_aftercnn
            if remainder != 0:
                cu_chunk_lens += [remainder]
        cu_seqlens = torch.tensor(cu_chunk_lens).cumsum(-1, dtype=torch.int32)

        # Encoder layers
        for i, layer in enumerate(audio_tower.layers):
            layer_outputs = layer(hidden_states, cu_seqlens)
            hidden_states = layer_outputs[0]
            np.save(os.path.join(output_dir, f"after_layer_{i}.npy"), hidden_states.numpy())
            if i == 0 or i == len(audio_tower.layers) - 1:
                print(f"After layer {i}: {hidden_states.shape}")

        # Post LN
        hidden_states = audio_tower.ln_post(hidden_states)
        np.save(os.path.join(output_dir, "after_ln_post.npy"), hidden_states.numpy())
        print(f"After ln_post: {hidden_states.shape}")

        # Projector
        hidden_states = audio_tower.proj1(hidden_states)
        hidden_states = audio_tower.act(hidden_states)
        np.save(os.path.join(output_dir, "after_proj1.npy"), hidden_states.numpy())
        print(f"After proj1+gelu: {hidden_states.shape}")

        hidden_states = audio_tower.proj2(hidden_states)
        np.save(os.path.join(output_dir, "after_proj2.npy"), hidden_states.numpy())
        print(f"After proj2 (final): {hidden_states.shape}")

    print(f"\nAll reference tensors saved to {output_dir}/")
    print("Compare against C++ ggml output using:")
    print("  python -c \"import numpy as np; a=np.load('ref.npy'); b=np.load('ggml.npy'); print(np.max(np.abs(a-b)))\"")


def main():
    parser = argparse.ArgumentParser(description="Extract Qwen3-ASR intermediate tensors for verification")
    parser.add_argument("--model-id", default="Qwen/Qwen3-ASR-0.6B", help="HuggingFace model ID")
    parser.add_argument("--audio", required=True, help="Path to test audio file (WAV)")
    parser.add_argument("--output-dir", default="./reference_tensors", help="Output directory for numpy files")
    args = parser.parse_args()

    extract_intermediates(args.model_id, args.audio, args.output_dir)


if __name__ == "__main__":
    main()
