import os
import random
import shutil
from pathlib import Path

def create_train_val_split(base_path, val_ratio=0.2):
    """
    Create train/val split maintaining consistent GT/LQ/LLQ pairs.

    Args:
        base_path: Base directory containing train folder with GT, LQ, and LLQ subfolders
        val_ratio: Ratio of validation set (default: 0.2 for 20%)
    """
    # Setup paths
    base_path = Path(base_path)
    train_path = base_path / 'train'
    val_path = base_path / 'val'

    # Create val directory if it doesn't exist
    os.makedirs(val_path, exist_ok=True)
    os.makedirs(val_path / 'GT', exist_ok=True)
    os.makedirs(val_path / 'LQ', exist_ok=True)
    os.makedirs(val_path / 'LLQ', exist_ok=True)

    # Get noise types from GT directory
    gt_dir = train_path / 'GT'
    noise_types = [d for d in os.listdir(gt_dir) if os.path.isdir(gt_dir / d)]

    for noise_type in noise_types:
        print(f"Processing {noise_type}...")

        # Create directories in val if they don't exist
        os.makedirs(val_path / 'GT' / noise_type, exist_ok=True)
        os.makedirs(val_path / 'LQ' / noise_type, exist_ok=True)

        # Get all images in this noise type from GT
        gt_images = [f for f in os.listdir(train_path / 'GT' / noise_type) 
                    if f.endswith(('.png', '.jpg', '.jpeg', '.JPEG'))]

        # Randomly select validation images
        num_val = int(len(gt_images) * val_ratio)
        val_images = set(random.sample(gt_images, num_val))

        for img_name in val_images:
            # Move GT image
            gt_src = train_path / 'GT' / noise_type / img_name
            gt_dst = val_path / 'GT' / noise_type / img_name

            # Move LQ image
            lq_src = train_path / 'LQ' / noise_type / img_name
            lq_dst = val_path / 'LQ' / noise_type / img_name

            # Move LLQ images
            llq_subfolders = [
                folder for folder in os.listdir(train_path / 'LLQ') 
                if folder.startswith(noise_type + "_")
            ]

            for llq_subfolder in llq_subfolders:
                llq_src = train_path / 'LLQ' / llq_subfolder / img_name
                llq_dst_dir = val_path / 'LLQ' / llq_subfolder
                os.makedirs(llq_dst_dir, exist_ok=True)
                llq_dst = llq_dst_dir / img_name

                if os.path.exists(llq_src):
                    shutil.move(str(llq_src), str(llq_dst))
                else:
                    print(f"Warning: Missing {img_name} in {llq_subfolder}")

            # Move GT and LQ images if they exist
            if os.path.exists(gt_src) and os.path.exists(lq_src):
                shutil.move(str(gt_src), str(gt_dst))
                shutil.move(str(lq_src), str(lq_dst))
            else:
                print(f"Warning: Missing pair for {img_name} in {noise_type}")

        # Print statistics
        remaining_train = len([f for f in os.listdir(train_path / 'GT' / noise_type) 
                             if f.endswith(('.png', '.jpg', '.jpeg', '.JPEG'))])
        moved_val = len([f for f in os.listdir(val_path / 'GT' / noise_type) 
                        if f.endswith(('.png', '.jpg', '.jpeg', '.JPEG'))])

        print(f"  {noise_type}: Total={len(gt_images)}, "
              f"Train={remaining_train}, Val={moved_val}")

if __name__ == "__main__":
    # Your specific path
    BASE_PATH = "/sise/eliorsu-group/lielbin/Courses/Generative_Models_in_AI/daclip-uir-main/universal-image-restoration/config/daclip-sde/universal"
    VAL_RATIO = 0.2  # 20% for validation

    # Set random seed for reproducibility
    random.seed(42)

    # Create the split
    create_train_val_split(BASE_PATH, VAL_RATIO)

    print("\nSplit complete!")
    print(f"Train path: {os.path.join(BASE_PATH, 'train')}")
    print(f"Val path: {os.path.join(BASE_PATH, 'val')}")

