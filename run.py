#!/usr/bin/env python3
"""
run.py — denoise one or more audio files with a trained IRM model

Example
-------
# Single file
python run.py \
    --model irm_denoiser.keras \
    --input noisy.wav \
    --output denoised.wav

# Multiple files (glob-safe)
python run.py \
    --model irm_denoiser.keras \
    --input audio/*.wav \
    --output_dir denoised/

# Override global_mean if you saved it separately
python run.py \
    --model irm_denoiser.keras \
    --global_mean 0.0312 \
    --input noisy.mp3 \
    --output denoised.wav
"""

import argparse
import os
import glob
import numpy as np
import tensorflow as tf

import config
from inference import denoise_file


def parse_args():
    p = argparse.ArgumentParser(description="Denoise audio with a trained IRM model")

    p.add_argument("--model",      required=True,
                   help="Path to saved .keras model file")
    p.add_argument("--input",      required=True, nargs="+",
                   help="Input audio file(s); supports globs")
    p.add_argument("--output",     default=None,
                   help="Output path (single-file mode)")
    p.add_argument("--output_dir", default=None,
                   help="Output directory (multi-file mode); created if absent")
    p.add_argument("--global_mean", type=float, default=None,
                   help="Global mean magnitude used during training. "
                        "If omitted, a rough estimate (0.03) is used.")
    p.add_argument("--context",    type=int, default=config.CONTEXT_FRAMES,
                   help=f"Context frames (default: {config.CONTEXT_FRAMES})")

    return p.parse_args()


def resolve_inputs(patterns):
    paths = []
    for pat in patterns:
        expanded = glob.glob(pat)
        paths.extend(expanded if expanded else [pat])
    return paths


def main():
    args = parse_args()

    # ---- Model -------------------------------------------------------
    print(f"Loading model from {args.model} ...")
    model = tf.keras.models.load_model(args.model, compile=False)

    global_mean = args.global_mean
    if global_mean is None:
        print(
            "Warning: --global_mean not provided. Using fallback value 0.03. "
            "For best results, pass the value printed during training."
        )
        global_mean = 0.03

    # ---- Files -------------------------------------------------------
    input_files = resolve_inputs(args.input)
    if not input_files:
        print("No input files found.")
        return

    multi = len(input_files) > 1 or args.output_dir is not None

    if multi and args.output_dir is None:
        args.output_dir = "denoised"
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    for audio_path in input_files:
        if multi:
            basename = os.path.splitext(os.path.basename(audio_path))[0]
            out_path  = os.path.join(args.output_dir, f"{basename}_denoised.wav")
        else:
            out_path = args.output or (
                os.path.splitext(audio_path)[0] + "_denoised.wav"
            )

        print(f"\n{'─'*50}")
        print(f"Input : {audio_path}")
        print(f"Output: {out_path}")
        denoise_file(model, global_mean, audio_path, out_path,
                     context_frames=args.context)

    print("\nDone.")


if __name__ == "__main__":
    main()
