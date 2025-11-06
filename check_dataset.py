#!/usr/bin/env python3
"""
Helper script to check your YOLO dataset before fine-tuning.
Verifies format, visualizes samples, and reports statistics.
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from collections import defaultdict


class DatasetChecker:
    def __init__(self, images_dir, labels_dir):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.stats = defaultdict(int)
        self.errors = []

    def check_structure(self):
        """Check if directories exist and contain files."""
        print("="*60)
        print("1. Checking dataset structure...")
        print("="*60)

        # Check images directory
        if not os.path.exists(self.images_dir):
            self.errors.append(f"Images directory not found: {self.images_dir}")
            return False

        # Check labels directory
        if not os.path.exists(self.labels_dir):
            self.errors.append(f"Labels directory not found: {self.labels_dir}")
            return False

        # Count files
        image_files = [f for f in os.listdir(self.images_dir)
                      if f.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'))]
        label_files = [f for f in os.listdir(self.labels_dir)
                      if f.endswith('.txt')]

        print(f"✓ Images directory: {self.images_dir}")
        print(f"  Found {len(image_files)} images")
        print(f"✓ Labels directory: {self.labels_dir}")
        print(f"  Found {len(label_files)} label files")

        if len(image_files) == 0:
            self.errors.append("No images found in images directory!")
            return False

        if len(label_files) == 0:
            self.errors.append("No label files found in labels directory!")
            return False

        self.stats['total_images'] = len(image_files)
        self.stats['total_labels'] = len(label_files)

        print()
        return True

    def check_labels(self):
        """Check label files format and content."""
        print("="*60)
        print("2. Checking label files...")
        print("="*60)

        image_files = [f for f in os.listdir(self.images_dir)
                      if f.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'))]

        missing_labels = []
        invalid_labels = []
        bbox_counts = []
        all_bboxes_valid = True

        for img_file in image_files:
            # Check if label exists
            label_name = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(self.labels_dir, label_name)

            if not os.path.exists(label_path):
                missing_labels.append(img_file)
                continue

            # Check label content
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()

                num_bboxes = 0
                for line_num, line in enumerate(lines, 1):
                    parts = line.strip().split()

                    if len(parts) != 5:
                        invalid_labels.append(
                            f"{label_name}:{line_num} - Expected 5 values, got {len(parts)}"
                        )
                        all_bboxes_valid = False
                        continue

                    # Parse values
                    try:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])

                        # Check if values are normalized (0-1)
                        if not (0 <= x_center <= 1):
                            invalid_labels.append(
                                f"{label_name}:{line_num} - x_center out of range: {x_center}"
                            )
                            all_bboxes_valid = False
                        if not (0 <= y_center <= 1):
                            invalid_labels.append(
                                f"{label_name}:{line_num} - y_center out of range: {y_center}"
                            )
                            all_bboxes_valid = False
                        if not (0 < width <= 1):
                            invalid_labels.append(
                                f"{label_name}:{line_num} - width out of range: {width}"
                            )
                            all_bboxes_valid = False
                        if not (0 < height <= 1):
                            invalid_labels.append(
                                f"{label_name}:{line_num} - height out of range: {height}"
                            )
                            all_bboxes_valid = False

                        num_bboxes += 1

                    except ValueError as e:
                        invalid_labels.append(
                            f"{label_name}:{line_num} - Invalid format: {e}"
                        )
                        all_bboxes_valid = False

                bbox_counts.append(num_bboxes)

            except Exception as e:
                invalid_labels.append(f"{label_name} - Error reading file: {e}")
                all_bboxes_valid = False

        # Report results
        if missing_labels:
            print(f"⚠ Missing labels for {len(missing_labels)} images:")
            for img in missing_labels[:5]:  # Show first 5
                print(f"  - {img}")
            if len(missing_labels) > 5:
                print(f"  ... and {len(missing_labels) - 5} more")
            self.errors.extend(missing_labels)
        else:
            print("✓ All images have corresponding label files")

        if invalid_labels:
            print(f"\n⚠ Found {len(invalid_labels)} invalid labels:")
            for error in invalid_labels[:10]:  # Show first 10
                print(f"  - {error}")
            if len(invalid_labels) > 10:
                print(f"  ... and {len(invalid_labels) - 10} more")
            self.errors.extend(invalid_labels)
        else:
            print("✓ All label files are valid")

        # Statistics
        if bbox_counts:
            print(f"\nBbox statistics:")
            print(f"  Total bboxes: {sum(bbox_counts)}")
            print(f"  Average bboxes per image: {np.mean(bbox_counts):.2f}")
            print(f"  Min bboxes per image: {np.min(bbox_counts)}")
            print(f"  Max bboxes per image: {np.max(bbox_counts)}")

            self.stats['total_bboxes'] = sum(bbox_counts)
            self.stats['avg_bboxes'] = np.mean(bbox_counts)

        print()
        return all_bboxes_valid and len(missing_labels) == 0

    def visualize_samples(self, num_samples=4):
        """Visualize random samples with their bboxes."""
        print("="*60)
        print("3. Visualizing samples...")
        print("="*60)

        image_files = [f for f in os.listdir(self.images_dir)
                      if f.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'))]

        # Select random samples
        import random
        samples = random.sample(image_files, min(num_samples, len(image_files)))

        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes = axes.ravel()

        for idx, img_file in enumerate(samples):
            # Load image
            img_path = os.path.join(self.images_dir, img_file)
            image = Image.open(img_path).convert("RGB")
            img_width, img_height = image.size

            # Load label
            label_name = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(self.labels_dir, label_name)

            draw = ImageDraw.Draw(image)

            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue

                        # Parse YOLO format
                        x_center = float(parts[1]) * img_width
                        y_center = float(parts[2]) * img_height
                        width = float(parts[3]) * img_width
                        height = float(parts[4]) * img_height

                        # Convert to corners
                        x1 = int(x_center - width / 2)
                        y1 = int(y_center - height / 2)
                        x2 = int(x_center + width / 2)
                        y2 = int(y_center + height / 2)

                        # Draw bbox
                        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

            # Plot
            axes[idx].imshow(image)
            axes[idx].set_title(f"{img_file}\nSize: {img_width}x{img_height}")
            axes[idx].axis('off')

        plt.tight_layout()
        output_path = "dataset_samples.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization to: {output_path}")
        print()

    def print_summary(self):
        """Print final summary."""
        print("="*60)
        print("SUMMARY")
        print("="*60)

        print(f"Total images: {self.stats.get('total_images', 0)}")
        print(f"Total labels: {self.stats.get('total_labels', 0)}")
        print(f"Total bboxes: {self.stats.get('total_bboxes', 0)}")
        print(f"Average bboxes per image: {self.stats.get('avg_bboxes', 0):.2f}")

        if self.errors:
            print(f"\n⚠ Found {len(self.errors)} issues!")
            print("Please fix these issues before training.")
            return False
        else:
            print("\n✓ Dataset looks good!")
            print("You can proceed with training: python simple_fine.py")
            return True

    def run_full_check(self):
        """Run all checks."""
        print("\n" + "="*60)
        print("CHECKING DATASET FOR FINE-TUNING")
        print("="*60 + "\n")

        success = True

        # Check structure
        if not self.check_structure():
            success = False
            print("\n✗ Structure check failed!")
            self.print_summary()
            return False

        # Check labels
        if not self.check_labels():
            success = False
            print("\n✗ Label check failed!")

        # Visualize samples
        try:
            self.visualize_samples()
        except Exception as e:
            print(f"⚠ Could not create visualizations: {e}")

        # Print summary
        self.print_summary()

        return success


def main():
    # Default paths (same as simple_fine.py)
    images_dir = "data/images"
    labels_dir = "data/labels"

    # Check if custom paths provided
    if len(sys.argv) >= 3:
        images_dir = sys.argv[1]
        labels_dir = sys.argv[2]
    elif len(sys.argv) == 2:
        print("Usage: python check_dataset.py [images_dir] [labels_dir]")
        print(f"Using default paths:")
        print(f"  Images: {images_dir}")
        print(f"  Labels: {labels_dir}")
    else:
        print(f"Using default paths:")
        print(f"  Images: {images_dir}")
        print(f"  Labels: {labels_dir}")
        print()

    # Run checks
    checker = DatasetChecker(images_dir, labels_dir)
    success = checker.run_full_check()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
