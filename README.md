Enhanced Traffic Classifier (ETC)
This repository provides the official source code for the Enhanced Traffic Classifier (ETC) model, as presented in our paper. The model leverages a masked autoencoder architecture with a latent context regressor for robust traffic classification. It includes the core model implementation, training utilities, and a pre-trained model checkpoint.

📄 Overview
The Enhanced Traffic Classifier (ETC) is designed to process traffic data by dividing input samples into patches and learning robust representations through self-supervised pre-training. The architecture is based on a Vision Transformer (ViT) with a Masked Autoencoder (MAE) pre-training strategy and an alignment branch to enforce consistency between the encoder and a frozen alignment encoder.

Key components:

Patch Embedding: Converts traffic matrix images into sequence of patches.

Masked Autoencoder: Randomly masks patches and reconstructs them using a regressor and decoder.

Latent Regressor: Cross-attention-based module that regresses masked representations from unmasked ones.

Alignment Branch: A frozen copy of the encoder to provide stable targets for representation learning.

🧠 Model Architecture
Main Model: MaskedAutoencoder
Encoder:

Patch embedding with positional encoding.

Transformer blocks (depth = 4, embed_dim = 192, num_heads = 16).

Random masking (default mask ratio = 0.6).

Alignment Encoder:

Copy of the encoder with frozen weights.

Provides a stable target for aligning masked representations.

Latent Regressor (LatentRegresser):

Cross-attention blocks to predict masked patch representations from visible ones.

Decoder:

Lightweight Transformer (depth = 2, embed_dim = 128) that reconstructs the original patches.

Fine-tuning Model: TrafficTransformer
A simplified version of the encoder for supervised classification tasks.

Includes per-packet feature extraction and aggregation across multiple packets.
