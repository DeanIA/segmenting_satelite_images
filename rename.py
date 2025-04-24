import os
import re

def rename_files_in_dir(directory):
    for filename in os.listdir(directory):
        # Match files with ._ before a number, e.g., '._0001.png'
        new_filename = re.sub(r'\._(\d+)', r'\1', filename)
        if new_filename != filename:
            src = os.path.join(directory, filename)
            dst = os.path.join(directory, new_filename)
            print(f"Renaming: {filename} -> {new_filename}")
            os.rename(src, dst)

# Example usage:
# Change these paths to your actual images/masks directories
train_img_dir = 'geoseg/data/Urban/train/images'
train_mask_dir = 'geoseg/data/Urban/train/masks'
val_img_dir = 'geoseg/data/Urban/val/images'
val_mask_dir = 'geoseg/data/Urban/val/masks'

rename_files_in_dir(train_img_dir)
rename_files_in_dir(train_mask_dir)
rename_files_in_dir(val_img_dir)
rename_files_in_dir(val_mask_dir)