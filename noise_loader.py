# noise_loader.py — loads and caches noise audio clips from disk
# This is used for dynamic noise addition to diversify the dataset
# My model was trained on the noise profiles provided in the DNS dataset

import os
import glob
import random
import numpy as np
import librosa
from tqdm import tqdm

from config import SAMPLE_RATE


class NoiseLoader:
    def __init__(self, noise_folder, sample_rate=SAMPLE_RATE, cache_size=50):
        self.sample_rate = sample_rate
        patterns = ["*.wav", "*.mp3", "*.flac", "*.ogg"]
        self.noise_files = []
        for p in patterns:
            self.noise_files.extend(glob.glob(os.path.join(noise_folder, p)))
            self.noise_files.extend(
                glob.glob(os.path.join(noise_folder, "**", p), recursive=True)
            )
        if not self.noise_files:
            raise ValueError(f"No audio files found in {noise_folder}")
        print(f"Found {len(self.noise_files)} noise files")

        self.noise_cache = []
        for fp in tqdm(
            random.sample(self.noise_files, min(cache_size, len(self.noise_files))),
            desc="Caching noise",
        ):
            try:
                a, _ = librosa.load(fp, sr=self.sample_rate, mono=True)
                if len(a) > self.sample_rate * 0.5:
                    self.noise_cache.append(a)
            except Exception as e:
                print(f"  skip {fp}: {e}")
        print(f"Cached {len(self.noise_cache)} noise clips")

    def get_noise_segment(self, n_samples, rng=None):
        if rng is not None:
            noise = self.noise_cache[rng.integers(len(self.noise_cache))]
            start = int(rng.integers(0, max(1, len(noise) - n_samples)))
        else:
            noise = random.choice(self.noise_cache)
            start = random.randint(0, max(0, len(noise) - n_samples))
        if len(noise) < n_samples:
            noise = np.tile(noise, int(np.ceil(n_samples / len(noise))))
        return noise[start : start + n_samples]
