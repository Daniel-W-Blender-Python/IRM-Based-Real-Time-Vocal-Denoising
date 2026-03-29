#!/usr/bin/env python3
"""
train.py — train the IRM denoiser

Example
-------
python train.py \
    --clean_npz /data/clean_50k.npz \
    --noise_dir /data/noise \
    --checkpoint model.keras \
    --epochs 200 \
    --batch_size 32 \
    --lr 5e-5
"""

import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers, callbacks

import config
from noise_loader import NoiseLoader
from data_generator import IRMGenerator
from u_net_model import build_irm_model, IRMModel


# ====================================================================
# Callbacks
# ====================================================================
class SaveBestBaseModel(callbacks.Callback):
    def __init__(self, base_model, filepath, monitor="val_loss"):
        super().__init__()
        self.base_model = base_model
        self.filepath   = filepath
        self.monitor    = monitor
        self.best       = np.inf

    def on_epoch_end(self, epoch, logs=None):
        v = logs.get(self.monitor)
        if v is not None and v < self.best:
            self.best = v
            self.base_model.save(self.filepath)
            print(f"\nEpoch {epoch + 1}: saved ({self.monitor}={v:.4f})")


# ====================================================================
# CLI
# ====================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Train the IRM denoiser")

    # Data
    p.add_argument("--clean_npz",   required=True,
                   help="Path to .npz file with a 'clean' array of audio clips")
    p.add_argument("--noise_dir",   required=True,
                   help="Directory containing noise audio files (wav/mp3/flac/ogg)")
    p.add_argument("--noise_cache", type=int, default=50,
                   help="Number of noise clips to cache in RAM (default: 50)")
    p.add_argument("--val_split",   type=float, default=0.1,
                   help="Fraction of data reserved for validation (default: 0.1)")

    # Training
    p.add_argument("--epochs",      type=int,   default=config.EPOCHS)
    p.add_argument("--batch_size",  type=int,   default=config.BATCH_SIZE)
    p.add_argument("--lr",          type=float, default=5e-5,
                   help="Initial learning rate (default: 5e-5)")
    p.add_argument("--snr_min",     type=float, default=5.0)
    p.add_argument("--snr_max",     type=float, default=35.0)
    p.add_argument("--context",     type=int,   default=config.CONTEXT_FRAMES,
                   help="Number of STFT context frames (default: 8)")
    p.add_argument("--augment_prob",type=float, default=0.5,
                   help="Probability of applying gain augmentation (default: 0.5)")

    # I/O
    p.add_argument("--checkpoint",  default="irm_denoiser.keras",
                   help="Path to save the best model (default: irm_denoiser.keras)")

    # Scheduler / stopping
    p.add_argument("--lr_patience",       type=int,   default=10,
                   help="ReduceLROnPlateau patience (default: 10)")
    p.add_argument("--lr_factor",         type=float, default=0.7,
                   help="ReduceLROnPlateau factor (default: 0.7)")
    p.add_argument("--early_stop_patience", type=int, default=30,
                   help="EarlyStopping patience (default: 30)")
    p.add_argument("--warmup_epochs",     type=int,   default=10,
                   help="Linear LR warmup length in epochs (default: 10)")

    return p.parse_args()


# ====================================================================
# Main
# ====================================================================
def main():
    args = parse_args()
    snr_range = (args.snr_min, args.snr_max)

    # ---- Data --------------------------------------------------------
    print("Loading dataset...")
    raw        = np.load(args.clean_npz)
    clean_data = raw["clean"].astype(np.float32)
    print(f"Loaded {len(clean_data)} clips")

    noise_loader = NoiseLoader(args.noise_dir, cache_size=args.noise_cache)

    indices = np.arange(len(clean_data))
    np.random.shuffle(indices)
    split = int((1.0 - args.val_split) * len(indices))

    train_gen = IRMGenerator(
        clean_data, noise_loader, indices[:split],
        context_frames=args.context,
        batch_size=args.batch_size,
        augment=True,
        augment_prob=args.augment_prob,
        snr_range=snr_range,
        fixed_seed=None,
    )
    val_gen = IRMGenerator(
        clean_data, noise_loader, indices[split:],
        context_frames=args.context,
        batch_size=args.batch_size,
        shuffle=False,
        augment=False,
        snr_range=snr_range,
        fixed_seed=config.VAL_SEED,
    )

    # ---- Model -------------------------------------------------------
    base_model = build_irm_model(context_frames=args.context)
    irm_model  = IRMModel(inputs=base_model.input, outputs=base_model.output)

    initial_lr = args.lr
    irm_model.compile(
        optimizer=optimizers.Adam(initial_lr, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
    )

    # ---- Callbacks ---------------------------------------------------
    warmup = args.warmup_epochs
    cbs = [
        SaveBestBaseModel(base_model, args.checkpoint),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=args.lr_factor,
            patience=args.lr_patience,
            min_lr=1e-6,
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.early_stop_patience,
            restore_best_weights=True,
        ),
        callbacks.LearningRateScheduler(
            lambda e: initial_lr * min(1.0, (e + 1) / warmup)
        ),
    ]

    # ---- Info --------------------------------------------------------
    print("\n" + "=" * 60)
    print("IRM DENOISER — training")
    print("=" * 60)
    print(f"Context    : {args.context} frames = "
          f"{args.context * config.HOP_LENGTH / config.SAMPLE_RATE * 1000:.0f} ms")
    print(f"SNR range  : {args.snr_min}–{args.snr_max} dB")
    print(f"LR         : {initial_lr}  (warmup {warmup} epochs)")
    print(f"Val seed   : {config.VAL_SEED}  (identical mixes every epoch)")
    print(f"Checkpoint : {args.checkpoint}")
    print("=" * 60)
    base_model.summary()
    print(f"Params: {base_model.count_params():,}")
    print()

    # ---- Train -------------------------------------------------------
    irm_model.fit(train_gen, validation_data=val_gen, epochs=args.epochs, callbacks=cbs)


if __name__ == "__main__":
    main()
