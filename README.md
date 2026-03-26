# Spiking Neural Network with STDP for Voice Recognition

Bio-inspired spiking neural network using Spike-Timing-Dependent Plasticity (STDP) 
for spoken digit recognition. Processes audio through cochlea-like filtering and 
learns frequency transition patterns without backpropagation.

## Features
- LIF neurons with forward Euler dynamics
- STDP implementation from first principles
- Support for SHD dataset and custom voice recordings
- End-to-end pipeline: audio → spikes → SNN → classification

## Results
- SHD dataset: XX% accuracy
- Custom voice: XX% accuracy (10 speakers)

## Quick Start
```bash
pip install -r requirements.txt
python experiments/train_voice.py --record