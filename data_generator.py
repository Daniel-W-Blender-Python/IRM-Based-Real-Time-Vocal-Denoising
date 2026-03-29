# data_generator.py — Keras Sequence data generator for IRM training

import random
import numpy as np
import librosa
import tensorflow as tf

from config import (
    SAMPLE_RATE, FFT_SIZE, HOP_LENGTH, BATCH_SIZE, CONTEXT_FRAMES, VAL_SEED
)


class IRMGenerator(tf.keras.utils.Sequence):
    """
    Each sample has the 4 ms frame of interest as well as a fixed number of context frames
    Target: IRM for the last frame only.

    Validation generator uses a fixed RNG seed defined in config.py
    signal — same noise mix, same SNR, same start frame every epoch.
    """

    def __init__(
        self,
        clean_data,
        noise_loader,
        indices,
        context_frames=CONTEXT_FRAMES,
        batch_size=BATCH_SIZE,
        shuffle=True,
        augment=True,
        augment_prob=0.5,
        snr_range=(5, 35),
        fixed_seed=None,
    ):
        self.clean_data     = clean_data
        self.noise_loader   = noise_loader
        self.indices        = list(indices)
        self.context_frames = context_frames
        self.batch_size     = batch_size
        self.shuffle        = shuffle
        self.augment        = augment
        self.augment_prob   = augment_prob
        self.snr_range      = snr_range
        self.fixed_seed     = fixed_seed
        self.eps            = 1e-8

        self._compute_global_stats()
        self.on_epoch_end()

    # ------------------------------------------------------------------
    def _compute_global_stats(self):
        print("Computing normalisation stats...")
        rng = np.random.default_rng(0)
        idx = rng.choice(len(self.indices), min(1000, len(self.indices)), replace=False)
        mags = []
        for i in idx:
            clean = self.clean_data[self.indices[i]]
            noise = self.noise_loader.get_noise_segment(len(clean))
            snr   = float(rng.uniform(*self.snr_range))
            noisy = self._mix(clean, noise, snr)
            mags.append(
                np.abs(
                    librosa.stft(noisy, n_fft=FFT_SIZE, hop_length=HOP_LENGTH)
                ).flatten()
            )
        self.global_mean = float(np.mean(np.concatenate(mags)))
        print(f"Global mean mag: {self.global_mean:.4f}")

    def _mix(self, clean, noise, snr_db):
        noise = noise[: len(clean)]
        cp    = np.mean(clean ** 2)
        np_   = max(np.mean(noise ** 2), 1e-10)
        return clean + noise * np.sqrt(cp / (np_ * 10 ** (snr_db / 10)))

    # ------------------------------------------------------------------
    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.indices)

    # ------------------------------------------------------------------
    def _process_one(self, clip_idx, rng=None):
        eps   = self.eps
        clean = self.clean_data[clip_idx].copy()
        noise = self.noise_loader.get_noise_segment(len(clean), rng=rng)

        snr = (
            float(rng.uniform(*self.snr_range))
            if rng is not None
            else float(np.random.uniform(*self.snr_range))
        )
        noisy = self._mix(clean, noise, snr)

        # Gain augmentation (train only)
        if self.augment and np.random.rand() < self.augment_prob:
            g = np.random.uniform(0.7, 1.3)
            clean *= g
            noisy *= g

        # Light Gaussian noise on the noisy side (train only)
        if self.augment and np.random.rand() < 0.3:
            snr_aug = np.random.uniform(25, 35)
            pwr     = np.mean(noisy ** 2)
            npwr    = pwr / (10 ** (snr_aug / 10))
            noisy  += np.random.normal(0, np.sqrt(max(npwr, 1e-10)), len(noisy))

        ns = librosa.stft(noisy, n_fft=FFT_SIZE, hop_length=HOP_LENGTH)
        cs = librosa.stft(clean, n_fft=FFT_SIZE, hop_length=HOP_LENGTH)
        n_frames = ns.shape[1]

        # Pick start frame (fixed for val via rng)
        if n_frames >= self.context_frames:
            start = (
                int(rng.integers(0, n_frames - self.context_frames + 1))
                if rng is not None
                else np.random.randint(0, n_frames - self.context_frames + 1)
            )
        else:
            start = 0

        n_seg = ns[:, start : start + self.context_frames]
        c_seg = cs[:, start : start + self.context_frames]

        if n_seg.shape[1] < self.context_frames:
            pad   = self.context_frames - n_seg.shape[1]
            n_seg = np.pad(n_seg, ((0, 0), (0, pad)))
            c_seg = np.pad(c_seg, ((0, 0), (0, pad)))

        nm   = np.abs(n_seg)
        norm = max(float(np.percentile(nm, 95)), self.global_mean, eps)
        n_seg = n_seg / norm
        c_seg = c_seg / norm

        # Input features
        nm_norm = np.abs(n_seg)
        log_m   = np.clip(np.log(np.clip(nm_norm, 1e-6, 1e3) + eps), -10, 5)
        phase   = np.angle(n_seg)
        X       = np.stack([log_m, phase], axis=-1).astype(np.float32)
        # X: (N_FREQ_BINS, CONTEXT_FRAMES, 2)

        # IRM target — last frame only
        last_n = n_seg[:, -1]
        last_c = c_seg[:, -1]
        irm    = np.clip(
            np.abs(last_c) / (np.abs(last_n) + eps), 0.0, 1.0
        ).astype(np.float32)

        return X, irm, last_n.astype(np.complex64), last_c.astype(np.complex64)

    def __getitem__(self, idx):
        batch_idx   = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        use_fixed   = self.fixed_seed is not None
        X_l, irm_l, ync_l, ycc_l = [], [], [], []

        for ci in batch_idx:
            rng = np.random.default_rng(self.fixed_seed + ci) if use_fixed else None
            X, irm, ync, ycc = self._process_one(ci, rng=rng)
            X_l.append(X)
            irm_l.append(irm)
            ync_l.append(ync)
            ycc_l.append(ycc)

        return (
            np.array(X_l, np.float32),
            (
                np.array(irm_l,  np.float32),
                np.array(ync_l,  np.complex64),
                np.array(ycc_l,  np.complex64),
            ),
        )
