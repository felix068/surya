#!/usr/bin/env python3
"""Script to get detailed information about the text detection model."""

from surya.detection.model.config import EfficientViTConfig
from surya.detection.model.encoderdecoder import EfficientViTForSemanticSegmentation
import torch

# Load config
config = EfficientViTConfig()
print('='*60)
print('=== MODEL CONFIGURATION ===')
print('='*60)
print(f'Number of classes (output channels): {config.num_classes}')
print(f'Widths: {config.widths}')
print(f'Depths: {config.depths}')
print(f'Hidden sizes: {config.hidden_sizes}')
print(f'Strides: {config.strides}')
print(f'Head dim: {config.head_dim}')
print(f'Decoder hidden size: {config.decoder_hidden_size}')
print(f'Decoder layer hidden size: {config.decoder_layer_hidden_size}')
print()

# Create model and count parameters
print('='*60)
print('Creating model and counting parameters...')
print('='*60)
model = EfficientViTForSemanticSegmentation(config)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print()
print('='*60)
print('=== MODEL SIZE ===')
print('='*60)
print(f'Total parameters: {total_params:,}')
print(f'Trainable parameters: {trainable_params:,}')
print(f'Model size (fp32): {total_params * 4 / (1024**2):.2f} MB')
print(f'Model size (fp16): {total_params * 2 / (1024**2):.2f} MB')
print()

# Count parameters by component
encoder_params = sum(p.numel() for p in model.efficientvit.parameters())
decoder_params = sum(p.numel() for p in model.decode_head.parameters())

print('='*60)
print('=== PARAMETERS BY COMPONENT ===')
print('='*60)
print(f'Encoder (EfficientViT): {encoder_params:,} ({encoder_params/total_params*100:.1f}%)')
print(f'Decoder (DecodeHead): {decoder_params:,} ({decoder_params/total_params*100:.1f}%)')
print()

print('='*60)
print('=== TRAINING INFO FROM README ===')
print('='*60)
print('Training hardware: 4x A6000 GPUs')
print('Training duration: 3 days')
print('Training approach: From scratch (not pretrained)')
print('Architecture: Modified EfficientViT for semantic segmentation')
print()
