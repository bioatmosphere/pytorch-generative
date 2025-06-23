# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Installation and Setup
```bash
pip install -r requirements.txt
```

### Testing
```bash
python -m unittest discover
```

### Training Models
```bash
# Train a specific model (e.g., ImageGPT)
python train.py --model image_gpt --logdir /tmp/run

# Train with custom parameters
python train.py --model nade --epochs 10 --batch-size 64 --logdir ./logs --gpus 1
```

### TensorBoard Visualization
```bash
tensorboard --logdir /tmp/run
```

## Architecture Overview

### Core Structure
- `pytorch_generative/`: Main package containing all model implementations and utilities
- `pytorch_generative/models/`: Model implementations organized by category
  - `autoregressive/`: PixelCNN, ImageGPT, MADE, NADE, etc.
  - `vae/`: VAE variants including VQ-VAE, Beta-VAE, VD-VAE
  - `flow/`: Normalizing flows like NICE
- `pytorch_generative/nn/`: Building blocks and utilities for neural networks
- `pytorch_generative/trainer.py`: Training loop abstraction with TensorBoard integration

### Model System
All models inherit from `GenerativeModel` base class in `models/base.py`. Each model implements a `reproduce()` function containing hyperparameters to reproduce published results.

Available models are registered in `train.py` MODEL_DICT and can be trained via the main training script.

### Training Infrastructure
The `Trainer` class handles:
- Multi-GPU training with DistributedDataParallel
- Checkpointing and model saving
- TensorBoard logging
- Learning rate scheduling
- Gradient clipping

### Key Modules
- `datasets.py`: Data loading utilities
- `debug.py`: Debugging tools for model development
- `colab_utils.py`: Google Colab integration utilities
- `optim.py`: Custom optimizers and learning rate schedules

## Model Categories

### Autoregressive Models
State-of-the-art implementations including PixelSNAIL (78.61 nats on Binarized MNIST), ImageGPT, Gated PixelCNN, and others.

### Variational Autoencoders
Complete VAE family including VD-VAE (≤80.72 nats), standard VAE, VQ-VAE variants, and Beta-VAE.

### Normalizing Flows
Flow-based models like NICE for exact likelihood computation.

## Development Notes

### Multi-GPU Training
The training script supports distributed training across multiple GPUs using `torch.multiprocessing.spawn()` and NCCL backend.

### Reproducibility
All models have `reproduce()` functions with exact hyperparameters from papers, making it easy to replicate published results.

### TensorBoard Integration
Training metrics are automatically logged to TensorBoard for visualization. The `Trainer` class handles all logging infrastructure.