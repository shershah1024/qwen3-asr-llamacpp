#!/usr/bin/env python3
"""
Convert Qwen3-ASR audio encoder + projector weights to GGUF (mmproj) format.

Usage:
    python convert_qwen3_asr_to_gguf.py --model-dir ./Qwen3-ASR-0.6B --output qwen3-asr-0.6b-mmproj.gguf

The text decoder (Qwen3) is converted separately using llama.cpp's convert_hf_to_gguf.py.
This script only handles the audio encoder and output projector.

Architecture (Qwen3-ASR-0.6B):
  Audio Encoder:
    - 3x Conv2d downsampling: Conv2d(1,480,3,s2,p1) → GELU, repeated 3x
    - Linear conv_out: 480*16 → 896 (d_model)
    - Sinusoidal positional embeddings: [1500, 896]
    - 18 transformer encoder layers:
        LayerNorm → MultiHeadAttn(14 heads, bias) → residual
        LayerNorm → FFN(GELU, 896→3584→896) → residual
    - Post LayerNorm
  Output Projector:
    - Linear(896, 896) + GELU + Linear(896, 1024)
"""

import argparse
import json
import math
import os
import struct
import sys
from pathlib import Path

import numpy as np

try:
    from safetensors import safe_open
except ImportError:
    print("ERROR: safetensors package required. Install with: pip install safetensors")
    sys.exit(1)


# ---------------------------------------------------------------------------
# GGUF constants and writer (minimal, self-contained)
# ---------------------------------------------------------------------------

GGUF_MAGIC = 0x46554747  # "GGUF"
GGUF_VERSION = 3

# GGUF value types
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10

# GGML tensor types
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q8_0 = 8

# File type enum (matches gguf.py)
GGUF_FTYPE_ALL_F32 = 0
GGUF_FTYPE_MOSTLY_F16 = 1


class GGUFWriter:
    """Minimal GGUF writer for mmproj files."""

    def __init__(self, path: str, ftype: int = GGUF_FTYPE_MOSTLY_F16):
        self.path = path
        self.ftype = ftype
        self.kv_data = []    # list of (key, type, value)
        self.tensors = []    # list of (name, shape, dtype, data_bytes)

    def add_string(self, key: str, val: str):
        self.kv_data.append((key, GGUF_TYPE_STRING, val))

    def add_uint32(self, key: str, val: int):
        self.kv_data.append((key, GGUF_TYPE_UINT32, val))

    def add_int32(self, key: str, val: int):
        self.kv_data.append((key, GGUF_TYPE_INT32, val))

    def add_float32(self, key: str, val: float):
        self.kv_data.append((key, GGUF_TYPE_FLOAT32, val))

    def add_bool(self, key: str, val: bool):
        self.kv_data.append((key, GGUF_TYPE_BOOL, val))

    def add_array_float32(self, key: str, vals: list):
        self.kv_data.append((key, (GGUF_TYPE_ARRAY, GGUF_TYPE_FLOAT32), vals))

    def add_tensor(self, name: str, data: np.ndarray, ggml_type: int = None):
        """Add a tensor. data should be numpy array."""
        if ggml_type is None:
            if data.dtype == np.float32:
                ggml_type = GGML_TYPE_F32
            elif data.dtype == np.float16:
                ggml_type = GGML_TYPE_F16
            else:
                raise ValueError(f"Unsupported dtype: {data.dtype}")

        self.tensors.append((name, list(data.shape), ggml_type, data.tobytes()))

    def _write_string(self, f, s: str):
        encoded = s.encode("utf-8")
        f.write(struct.pack("<Q", len(encoded)))
        f.write(encoded)

    def _write_kv(self, f, key: str, vtype, value):
        self._write_string(f, key)

        if isinstance(vtype, tuple):
            # Array type
            arr_meta_type, arr_elem_type = vtype
            f.write(struct.pack("<I", arr_meta_type))
            f.write(struct.pack("<I", arr_elem_type))
            f.write(struct.pack("<Q", len(value)))
            for v in value:
                if arr_elem_type == GGUF_TYPE_FLOAT32:
                    f.write(struct.pack("<f", v))
                elif arr_elem_type == GGUF_TYPE_UINT32:
                    f.write(struct.pack("<I", v))
                elif arr_elem_type == GGUF_TYPE_INT32:
                    f.write(struct.pack("<i", v))
        elif vtype == GGUF_TYPE_STRING:
            f.write(struct.pack("<I", vtype))
            self._write_string(f, value)
        elif vtype == GGUF_TYPE_UINT32:
            f.write(struct.pack("<I", vtype))
            f.write(struct.pack("<I", value))
        elif vtype == GGUF_TYPE_INT32:
            f.write(struct.pack("<I", vtype))
            f.write(struct.pack("<i", value))
        elif vtype == GGUF_TYPE_FLOAT32:
            f.write(struct.pack("<I", vtype))
            f.write(struct.pack("<f", value))
        elif vtype == GGUF_TYPE_BOOL:
            f.write(struct.pack("<I", vtype))
            f.write(struct.pack("<?", value))
        else:
            raise ValueError(f"Unknown type: {vtype}")

    def write(self):
        with open(self.path, "wb") as f:
            # Header
            f.write(struct.pack("<I", GGUF_MAGIC))
            f.write(struct.pack("<I", GGUF_VERSION))
            f.write(struct.pack("<Q", len(self.tensors)))  # n_tensors
            f.write(struct.pack("<Q", len(self.kv_data)))  # n_kv

            # KV pairs
            for key, vtype, value in self.kv_data:
                self._write_kv(f, key, vtype, value)

            # Tensor infos
            offsets = []
            current_offset = 0
            for name, shape, ggml_type, data_bytes in self.tensors:
                self._write_string(f, name)
                n_dims = len(shape)
                f.write(struct.pack("<I", n_dims))
                # GGUF stores dimensions in ggml (reversed) order
                for dim in reversed(shape):
                    f.write(struct.pack("<Q", dim))
                f.write(struct.pack("<I", ggml_type))
                # Align offset to 32 bytes
                aligned_offset = (current_offset + 31) & ~31
                f.write(struct.pack("<Q", aligned_offset))
                offsets.append(aligned_offset)
                current_offset = aligned_offset + len(data_bytes)

            # Padding to align tensor data start
            tensor_data_start = f.tell()
            alignment = 32
            pad = (alignment - (tensor_data_start % alignment)) % alignment
            f.write(b"\x00" * pad)

            # Tensor data
            for i, (name, shape, ggml_type, data_bytes) in enumerate(self.tensors):
                # Pad to alignment
                current_pos = f.tell() - tensor_data_start - pad
                expected_pos = offsets[i]
                if current_pos < expected_pos:
                    f.write(b"\x00" * (expected_pos - current_pos))
                f.write(data_bytes)

        print(f"Wrote {self.path} ({os.path.getsize(self.path) / 1024 / 1024:.1f} MB)")


# ---------------------------------------------------------------------------
# Tensor name mapping: HuggingFace → GGUF
# ---------------------------------------------------------------------------

def map_tensor_name(hf_name: str) -> str | None:
    """Map HuggingFace tensor name to GGUF tensor name.

    Returns None if the tensor should be skipped (text decoder tensors).
    """
    # Skip text decoder and top-level wrapper tensors
    if not hf_name.startswith("thinker.audio_tower."):
        # Check for projector tensors
        if hf_name.startswith("thinker.lm_head."):
            return None  # skip — belongs to text model
        if hf_name.startswith("thinker.model."):
            return None  # skip — text decoder
        return None

    name = hf_name.replace("thinker.audio_tower.", "")

    # Conv2d layers
    if name.startswith("conv2d"):
        # conv2d1.weight → a.conv2d.1.weight
        idx = name[6]  # '1', '2', or '3' (conv2d is 6 chars)
        suffix = "weight" if "weight" in name else "bias"
        return f"a.conv2d.{idx}.{suffix}"

    # Conv output linear
    if name.startswith("conv_out."):
        suffix = "weight" if "weight" in name else "bias"
        return f"a.conv_out.{suffix}"

    # Positional embedding (sinusoidal, stored as buffer)
    if name == "positional_embedding.positional_embedding":
        return "a.position_embd.weight"

    # Encoder layers
    if name.startswith("layers."):
        parts = name.split(".")
        layer_idx = int(parts[1])

        rest = ".".join(parts[2:])

        # Attention
        if rest == "self_attn.q_proj.weight":
            return f"a.blk.{layer_idx}.attn_q.weight"
        if rest == "self_attn.q_proj.bias":
            return f"a.blk.{layer_idx}.attn_q.bias"
        if rest == "self_attn.k_proj.weight":
            return f"a.blk.{layer_idx}.attn_k.weight"
        if rest == "self_attn.k_proj.bias":
            return f"a.blk.{layer_idx}.attn_k.bias"
        if rest == "self_attn.v_proj.weight":
            return f"a.blk.{layer_idx}.attn_v.weight"
        if rest == "self_attn.v_proj.bias":
            return f"a.blk.{layer_idx}.attn_v.bias"
        if rest == "self_attn.out_proj.weight":
            return f"a.blk.{layer_idx}.attn_out.weight"
        if rest == "self_attn.out_proj.bias":
            return f"a.blk.{layer_idx}.attn_out.bias"

        # Layer norms
        if rest == "self_attn_layer_norm.weight":
            return f"a.blk.{layer_idx}.ln1.weight"
        if rest == "self_attn_layer_norm.bias":
            return f"a.blk.{layer_idx}.ln1.bias"
        if rest == "final_layer_norm.weight":
            return f"a.blk.{layer_idx}.ln2.weight"
        if rest == "final_layer_norm.bias":
            return f"a.blk.{layer_idx}.ln2.bias"

        # FFN
        if rest == "fc1.weight":
            return f"a.blk.{layer_idx}.ffn_up.weight"
        if rest == "fc1.bias":
            return f"a.blk.{layer_idx}.ffn_up.bias"
        if rest == "fc2.weight":
            return f"a.blk.{layer_idx}.ffn_down.weight"
        if rest == "fc2.bias":
            return f"a.blk.{layer_idx}.ffn_down.bias"

    # Post layer norm
    if name == "ln_post.weight":
        return "a.post_ln.weight"
    if name == "ln_post.bias":
        return "a.post_ln.bias"

    # Output projector
    if name == "proj1.weight":
        return "mm.a.mlp.1.weight"
    if name == "proj1.bias":
        return "mm.a.mlp.1.bias"
    if name == "proj2.weight":
        return "mm.a.mlp.2.weight"
    if name == "proj2.bias":
        return "mm.a.mlp.2.bias"

    print(f"WARNING: unmapped tensor: {hf_name}")
    return None


def compute_sinusoidal_pos_embd(length: int, channels: int, max_timescale: int = 10000) -> np.ndarray:
    """Compute sinusoidal positional embeddings (same as PyTorch SinusoidsPositionEmbedding)."""
    log_timescale_increment = math.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = np.exp(-log_timescale_increment * np.arange(channels // 2, dtype=np.float32))
    scaled_time = np.arange(length, dtype=np.float32)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert(model_dir: str, output: str, use_f16: bool = True):
    model_dir = Path(model_dir)

    # Load config
    config_path = model_dir / "config.json"
    if not config_path.exists():
        print(f"ERROR: {config_path} not found")
        sys.exit(1)

    with open(config_path) as f:
        config = json.load(f)

    thinker = config.get("thinker_config", config)
    audio_cfg = thinker["audio_config"]
    text_cfg = thinker["text_config"]

    # Audio encoder hyperparameters
    n_mel_bins = audio_cfg["num_mel_bins"]           # 128
    d_model = audio_cfg["d_model"]                   # 896
    n_layers = audio_cfg["encoder_layers"]            # 18
    n_heads = audio_cfg["encoder_attention_heads"]    # 14
    n_ff = audio_cfg["encoder_ffn_dim"]               # 3584
    output_dim = audio_cfg["output_dim"]              # 1024
    max_source_positions = audio_cfg["max_source_positions"]  # 1500
    downsample_hidden_size = audio_cfg["downsample_hidden_size"]  # 480
    n_window = audio_cfg["n_window"]                  # 50
    n_window_infer = audio_cfg["n_window_infer"]      # 800
    conv_chunksize = audio_cfg["conv_chunksize"]      # 500
    eps = audio_cfg.get("layer_norm_eps", 1e-5)

    # Compute freq dim after 3x Conv2d stride-2
    freq_after_conv = ((((n_mel_bins + 1) // 2 + 1) // 2 + 1) // 2)
    conv_out_features = downsample_hidden_size * freq_after_conv

    print(f"Qwen3-ASR Audio Encoder Config:")
    print(f"  n_mel_bins:       {n_mel_bins}")
    print(f"  d_model:          {d_model}")
    print(f"  n_layers:         {n_layers}")
    print(f"  n_heads:          {n_heads}")
    print(f"  n_ff:             {n_ff}")
    print(f"  output_dim:       {output_dim}")
    print(f"  max_positions:    {max_source_positions}")
    print(f"  downsample_hidden: {downsample_hidden_size}")
    print(f"  freq_after_conv:  {freq_after_conv}")
    print(f"  conv_out_features:{conv_out_features}")
    print(f"  n_window:         {n_window}")
    print(f"  n_window_infer:   {n_window_infer}")
    print(f"  conv_chunksize:   {conv_chunksize}")
    print()

    # Initialize GGUF writer
    ftype = GGUF_FTYPE_MOSTLY_F16 if use_f16 else GGUF_FTYPE_ALL_F32
    writer = GGUFWriter(output, ftype)

    # Write metadata
    writer.add_string("general.architecture", "clip")
    writer.add_string("general.name", "qwen3-asr")
    writer.add_string("general.description", f"Qwen3-ASR audio encoder")
    writer.add_uint32(KEY_FTYPE, ftype)

    # Projector type
    writer.add_string("clip.projector_type", "qwen3asr")
    writer.add_bool("clip.has_audio_encoder", True)
    writer.add_bool("clip.has_vision_encoder", False)

    # Audio encoder params
    writer.add_uint32("clip.audio.embedding_length", d_model)
    writer.add_uint32("clip.audio.feed_forward_length", n_ff)
    writer.add_uint32("clip.audio.block_count", n_layers)
    writer.add_uint32("clip.audio.attention.head_count", n_heads)
    writer.add_float32("clip.audio.attention.layer_norm_epsilon", eps)
    writer.add_uint32("clip.audio.num_mel_bins", n_mel_bins)
    writer.add_uint32("clip.audio.projection_dim", output_dim)

    # Qwen3-ASR specific params
    writer.add_uint32("clip.audio.downsample_hidden_size", downsample_hidden_size)
    writer.add_uint32("clip.audio.max_source_positions", max_source_positions)
    writer.add_uint32("clip.audio.n_window", n_window)
    writer.add_uint32("clip.audio.n_window_infer", n_window_infer)
    writer.add_uint32("clip.audio.conv_chunksize", conv_chunksize)

    # Text model info (needed for projector dim matching)
    writer.add_uint32("clip.audio.text_hidden_size", text_cfg["hidden_size"])

    # Load safetensors (handle both single-file and sharded models)
    safetensors_path = model_dir / "model.safetensors"
    if safetensors_path.exists():
        safetensors_files = [safetensors_path]
    else:
        # Sharded model: find all shard files
        safetensors_files = sorted(model_dir.glob("model-*.safetensors"))
        if not safetensors_files:
            print(f"ERROR: no safetensors files found in {model_dir}")
            sys.exit(1)

    # Open all shard files and collect audio tower tensor names
    shard_handles = []
    all_keys = {}  # key -> shard handle
    for sf in safetensors_files:
        print(f"Loading {sf.name}...")
        st = safe_open(str(sf), framework="pt")
        shard_handles.append(st)
        for k in st.keys():
            all_keys[k] = st

    # Compute and add sinusoidal positional embeddings
    # These are not stored as model weights (registered as buffer, not parameter)
    # so we compute them here
    pos_embd = compute_sinusoidal_pos_embd(max_source_positions, d_model)
    print(f"  Computed sinusoidal pos embeddings: {pos_embd.shape}")
    # Position embeddings must be F32 for Metal backend (used as src[1] in ggml_add)
    writer.add_tensor("a.position_embd.weight", pos_embd.astype(np.float32))

    # Process tensors
    n_converted = 0
    n_skipped = 0
    has_pos_embd_in_file = False

    for hf_name in sorted(all_keys.keys()):
        gguf_name = map_tensor_name(hf_name)

        if gguf_name is None:
            n_skipped += 1
            continue

        # Skip if it's the positional embedding (we computed it above)
        if gguf_name == "a.position_embd.weight":
            has_pos_embd_in_file = True
            # Use the one from file instead of computed
            # Always F32 for Metal backend compatibility (used as src[1] in ggml_add)
            data = all_keys[hf_name].get_tensor(hf_name).float().numpy().astype(np.float32)
            # Replace the computed one
            for i, (name, _, _, _) in enumerate(writer.tensors):
                if name == "a.position_embd.weight":
                    writer.tensors[i] = (name, list(data.shape),
                                         GGML_TYPE_F32,
                                         data.tobytes())
                    break
            n_converted += 1
            print(f"  {hf_name} → {gguf_name} (from file) {data.shape}")
            continue

        data = all_keys[hf_name].get_tensor(hf_name).float().numpy()

        # Conv2d weights need special handling
        # PyTorch Conv2d weight shape: [out_channels, in_channels, kH, kW]
        # ggml conv_2d expects: [out_channels, kH, kW, in_channels] (OIHW → OHWI)
        # Actually, ggml_conv_2d kernel layout is [OC, KH, KW, IC] but stored as [KW, KH, IC, OC]
        # in ggml tensor format (column-major / reverse dimension order)
        if "conv2d" in hf_name and "weight" in hf_name:
            # Keep as-is; ggml_conv_2d handles PyTorch OIHW layout
            # The ggml tensor dimensions are stored in reverse order
            pass

        # Conv2d bias: add trailing dimension for ggml broadcast
        if "conv2d" in hf_name and "bias" in hf_name:
            data = data.reshape(-1, 1, 1)

        # Determine output type
        # Biases must be F32 for Metal backend (ggml_add requires src[1] to be F32)
        is_bias = "bias" in gguf_name
        if use_f16 and data.ndim >= 2 and not is_bias:
            data = data.astype(np.float16)
            ggml_type = GGML_TYPE_F16
        else:
            data = data.astype(np.float32)
            ggml_type = GGML_TYPE_F32

        writer.add_tensor(gguf_name, data, ggml_type)
        n_converted += 1
        print(f"  {hf_name} → {gguf_name} {data.shape} {'f16' if ggml_type == GGML_TYPE_F16 else 'f32'}")

    if not has_pos_embd_in_file:
        print(f"  (positional embeddings computed from formula, not in safetensors)")

    print(f"\nConverted {n_converted} tensors, skipped {n_skipped} (text decoder)")

    # Write output
    writer.write()
    print("Done.")


# Key constants (matching clip-impl.h)
KEY_FTYPE = "general.file_type"


def main():
    parser = argparse.ArgumentParser(description="Convert Qwen3-ASR audio encoder to GGUF")
    parser.add_argument("--model-dir", required=True, help="Path to Qwen3-ASR-0.6B model directory")
    parser.add_argument("--output", default="qwen3-asr-mmproj.gguf", help="Output GGUF file path")
    parser.add_argument("--f32", action="store_true", help="Use float32 instead of float16")
    args = parser.parse_args()

    convert(args.model_dir, args.output, use_f16=not args.f32)


if __name__ == "__main__":
    main()
