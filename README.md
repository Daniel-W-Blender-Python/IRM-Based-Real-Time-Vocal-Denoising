https://arxiv.org/pdf/2603.29326

# IRM-Based Real-Time Vocal Denoising

My real-time approach to vocal denoising uses an ideal ratio mask (IRM) to efficiently predict clean speech. With a 4 ms frame duration and an average of 2.3 ms of inference time, the model can run live on a total latency of less than 6.5 ms on a CPU.

In this repository, I provide the training script + inference script for the model as well as a live VST plugin for use in a DAW. 

# Requirements
```bash
tensorflow >= 2.12
librosa
soundfile
numpy
tqdm
```

# Training

In training the model, I converted the dataset to an npz file where each audio segment is one second long. The datasets I used include the Saraga Carnatic Music 
Dataset, CommonVoice, Noisy Speech Database, GTSinger, SingingDatabase, VocalSet, and the Acapella Mandarin Singing Dataset.

```bash
python train.py \
    --clean_npz  /data/clean_50k.npz \
    --noise_dir  /data/noise \
    --checkpoint irm_denoiser.keras \
    --epochs     200 \
    --batch_size 32 \
    --lr         5e-5
```

# Inference

Single file
```bash
python infer.py \
    --model       irm_denoiser.keras \
    --global_mean 0.0312 \
    --input       noisy.wav \
    --output      denoised.wav
```

Batch (directory)
```bash
python infer.py \
    --model       irm_denoiser.keras \
    --global_mean 0.0312 \
    --input       audio/*.wav \
    --output_dir  denoised/
```

# VST Plugin

The provided VST plugin actually uses the updated deep filtering implementation rather than the IRM-based approach. However, most of the architecture is the same. While the plugin runs live on a CPU, it takes up most of the CPU during inference, so only one or two instances tend to work without dropping samples.
