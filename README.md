# IRM-Based Real-Time Vocal Denoising
My real-time approach to vocal denoising uses an ideal ratio mask (IRM) to efficiently predict clean speech. With a 4 ms frame duration and an average of 2.3 ms of inference time, the model can run live on a total latency of less than 6.5 ms.

In this repository, I provide the training script + inference script for the model as well as a live VST plugin for use in a DAW. 
