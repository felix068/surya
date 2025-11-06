#!/usr/bin/env python3
"""
Script to check if YOLO annotations are LINE-level or ZONE-level.
Visualizes bboxes and calculates aspect ratios.
"""

import os
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

IMAGES_DIR = "data/images"
LABELS_DIR = "data/labels"

def analyze_bboxes():
    """Analyze bbox aspect ratios to detect if they're lines or zones."""

    image_files = [f for f in os.listdir(IMAGES_DIR)
                   if f.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'))]

    all_aspect_ratios = []
    all_heights = []

    print("Analyzing bboxes...")
    print("="*60)

    for img_file in image_files[:50]:  # Check first 50
        # Load image
        img_path = os.path.join(IMAGES_DIR, img_file)
        image = Image.open(img_path)
        img_width, img_height = image.size

        # Load labels
        label_name = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(LABELS_DIR, label_name)

        if not os.path.exists(label_path):
            continue

        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                # Parse YOLO
                width_norm = float(parts[3])
                height_norm = float(parts[4])

                # Convert to pixels
                width_px = width_norm * img_width
                height_px = height_norm * img_height

                # Aspect ratio
                aspect_ratio = width_px / (height_px + 1e-6)

                all_aspect_ratios.append(aspect_ratio)
                all_heights.append(height_px)

    # Statistics
    aspect_ratios = np.array(all_aspect_ratios)
    heights = np.array(all_heights)

    print(f"\n📊 Bbox Statistics (n={len(aspect_ratios)}):")
    print(f"   Aspect Ratio (width/height):")
    print(f"      Mean: {aspect_ratios.mean():.2f}")
    print(f"      Median: {np.median(aspect_ratios):.2f}")
    print(f"      Min: {aspect_ratios.min():.2f}")
    print(f"      Max: {aspect_ratios.max():.2f}")
    print()
    print(f"   Height (pixels):")
    print(f"      Mean: {heights.mean():.1f}px")
    print(f"      Median: {np.median(heights):.1f}px")
    print(f"      Min: {heights.min():.1f}px")
    print(f"      Max: {heights.max():.1f}px")
    print()

    # Interpretation
    print("="*60)
    print("🔍 Interpretation:")
    print("="*60)

    median_ratio = np.median(aspect_ratios)
    median_height = np.median(heights)

    if median_ratio > 10:
        print("✅ GOOD: Bboxes look like LINES (wide, flat)")
        print(f"   Median aspect ratio: {median_ratio:.1f}:1")
    elif 3 <= median_ratio <= 10:
        print("⚠️  AMBIGUOUS: Bboxes might be short lines or zones")
        print(f"   Median aspect ratio: {median_ratio:.1f}:1")
    else:
        print("❌ BAD: Bboxes look like ZONES (square-ish)")
        print(f"   Median aspect ratio: {median_ratio:.1f}:1")
        print("   → Your model will learn to detect zones, not lines!")

    print()

    if median_height > 100:
        print("⚠️  TALL boxes detected!")
        print(f"   Median height: {median_height:.1f}px")
        print("   → This suggests multi-line zones, not single lines")
    elif median_height < 30:
        print("⚠️  Very small boxes detected!")
        print(f"   Median height: {median_height:.1f}px")
        print("   → Check if labels are correct")
    else:
        print(f"✓ Reasonable height: {median_height:.1f}px")

    print()

    # Histogram
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(aspect_ratios, bins=50, edgecolor='black')
    axes[0].axvline(median_ratio, color='red', linestyle='--',
                   label=f'Median: {median_ratio:.1f}')
    axes[0].set_xlabel('Aspect Ratio (width/height)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Bbox Aspect Ratios')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].hist(heights, bins=50, edgecolor='black')
    axes[1].axvline(median_height, color='red', linestyle='--',
                   label=f'Median: {median_height:.1f}px')
    axes[1].set_xlabel('Height (pixels)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Bbox Heights')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('bbox_analysis.png', dpi=150)
    print(f"📊 Saved histogram to: bbox_analysis.png")
    print()


def visualize_samples():
    """Visualize random samples with bboxes."""

    image_files = [f for f in os.listdir(IMAGES_DIR)
                   if f.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'))]

    import random
    samples = random.sample(image_files, min(4, len(image_files)))

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.ravel()

    for idx, img_file in enumerate(samples):
        # Load image
        img_path = os.path.join(IMAGES_DIR, img_file)
        image = Image.open(img_path).convert("RGB")
        img_width, img_height = image.size

        # Load labels
        label_name = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(LABELS_DIR, label_name)

        draw = ImageDraw.Draw(image)
        num_boxes = 0

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue

                    # Parse YOLO
                    x_center = float(parts[1]) * img_width
                    y_center = float(parts[2]) * img_height
                    width = float(parts[3]) * img_width
                    height = float(parts[4]) * img_height

                    # Convert to corners
                    x1 = int(x_center - width / 2)
                    y1 = int(y_center - height / 2)
                    x2 = int(x_center + width / 2)
                    y2 = int(y_center + height / 2)

                    # Draw
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                    num_boxes += 1

        # Plot
        axes[idx].imshow(image)
        axes[idx].set_title(f"{img_file}\n{num_boxes} bboxes, {img_width}x{img_height}px")
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig('bbox_samples.png', dpi=150)
    print(f"📷 Saved visualization to: bbox_samples.png")
    print()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("YOLO BBOX ANALYSIS - Line vs Zone Detection")
    print("="*60 + "\n")

    analyze_bboxes()
    visualize_samples()

    print("="*60)
    print("✅ Analysis complete!")
    print("="*60)
    print("\nCheck the generated images:")
    print("  - bbox_analysis.png : Histograms of aspect ratios")
    print("  - bbox_samples.png  : Visual samples with bboxes")
    print()
