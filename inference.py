# inference.py — streaming denoiser and file-level inference helper

import time
import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf
from tqdm import tqdm

from config import SAMPLE_RATE, FFT_SIZE, HOP_LENGTH, CONTEXT_FRAMES, N_FREQ_BINS
from collections import deque


class StreamingDenoiser:
    """
    Frame-by-frame denoiser that mirrors training normalisation exactly.

    The buffer stores raw (unnormalised) complex STFT frames. Each call
    normalises the entire buffer together with the same 95th-percentile
    technique used during training.
    """

    def __init__(self, model, global_mean, context_frames=CONTEXT_FRAMES):
        self.model          = model
        self.global_mean    = global_mean
        self.context_frames = context_frames
        self.n_freq         = N_FREQ_BINS
        self.eps            = 1e-8
        self.buffer         = deque(maxlen=context_frames)

        for _ in range(context_frames):
            self.buffer.append(np.zeros(self.n_freq, dtype=np.complex64))

        # Compile inference path once
        self._infer = tf.function(
            lambda f: self.model(f, training=False),
            reduce_retracing=True,
        )
        dummy = np.zeros((1, self.n_freq, context_frames, 2), np.float32)
        self._infer(tf.constant(dummy))

        budget_ms = context_frames * HOP_LENGTH / SAMPLE_RATE * 1000
        print(
            f"StreamingDenoiser ready | context={context_frames} frames "
            f"({budget_ms:.0f} ms) | params={model.count_params():,}"
        )

    def reset(self):
        self.buffer.clear()
        for _ in range(self.context_frames):
            self.buffer.append(np.zeros(self.n_freq, dtype=np.complex64))

    def process_frame(self, stft_frame: np.ndarray) -> np.ndarray:
        self.buffer.append(stft_frame.copy())

        buf  = np.array(self.buffer).T  # (freq, context_frames)
        nm   = np.abs(buf)
        norm = max(float(np.percentile(nm, 95)), self.global_mean, self.eps)

        buf_n = buf / norm
        log_m = np.clip(np.log(np.abs(buf_n) + self.eps), -10, 5)
        feat  = np.stack([log_m, np.angle(buf_n)], axis=-1)[np.newaxis].astype(np.float32)

        mask = self._infer(tf.constant(feat)).numpy()[0]
        return buf_n[:, -1] * mask * norm


def denoise_file(model, global_mean, audio_path, out_path,
                 context_frames=CONTEXT_FRAMES):
    denoiser = StreamingDenoiser(model, global_mean, context_frames)
    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
    stft     = librosa.stft(audio, n_fft=FFT_SIZE, hop_length=HOP_LENGTH)
    enh      = np.zeros_like(stft, dtype=np.complex64)
    ts       = []

    for i in tqdm(range(stft.shape[1]), desc="Streaming"):
        t0        = time.perf_counter()
        enh[:, i] = denoiser.process_frame(stft[:, i])
        ts.append(time.perf_counter() - t0)

    out = librosa.istft(enh, hop_length=HOP_LENGTH, length=len(audio))
    sf.write(out_path, out, SAMPLE_RATE)

    avg = np.mean(ts) * 1000
    bud = HOP_LENGTH / SAMPLE_RATE * 1000
    rtf = avg / bud
    print(
        f"Avg {avg:.3f} ms/frame | budget {bud:.1f} ms | RTF={rtf:.3f} "
        f"{'✅' if rtf < 1 else '⚠️  too slow'}"
    )
    return out
