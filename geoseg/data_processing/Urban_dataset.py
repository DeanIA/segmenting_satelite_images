# This file holds the class and functions for loading and augmenting the urban dataset, 
# for training and inference. 
 
from .transform import *
import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as albu
from PIL import Image
import random
import cv2

# Global constants used downstream 
CLASSES = ('background', 'building', 'road', 'water', 'barren', 'forest','agricultural','other')
PALETTE = [[255, 255, 255], [255, 0, 0], [255, 255, 0], [0, 0, 255],
           [159, 129, 183], [0, 255, 0], [255, 195, 128],[0,0,0]]

# Global constants used for training 
ORIGIN_IMG_SIZE = (1024, 1024)
INPUT_IMG_SIZE = (1024, 1024)
TEST_IMG_SIZE = (1024, 1024)

# Define a "None class" index as the length of the class list. If I have 0-7 classes, 8 is reserved for none. 
ignore_index = len(CLASSES) 
palette = [255, 255, 255, 255, 0, 0, 255, 255, 0, 0, 0, 255, 159, 129, 183, 0, 255, 0, 255, 195, 128, 0, 0, 0]

def get_training_transform():
    """
    Provides an Albumentations pipeline for training data augmentation of images and masks.

    Input:
        None

    What it does:
        - RandomRotate90(p=0.5): with 50% chance, rotates both image and mask by a random multiple of 90°
        - Normalize(): converts image to float, scales to [0,1], then standardizes by mean/std

    Output:
        albu.Compose: a callable transform which, when invoked as
            augmented = transform(image=img, mask=mask)
        returns a dict with keys 'image' and 'mask' containing the augmented data.
    """
    train_transform = [
        albu.RandomRotate90(p=0.5),
        albu.Normalize()]
    return albu.Compose(train_transform)

def train_aug(img, mask):
    """
    Applies a two‑stage augmentation pipeline to an RGB image and its segmentation mask for training.

    Stage 1: Geometric transforms
      - RandomHorizontalFlip(prob=0.5)
      - RandomVerticalFlip(prob=0.5)
      - RandomScale(scale_list=[0.5, 0.75, 1.0, 1.25, 1.5], mode='value')
      - SmartCropV1(crop_size=512, max_ratio=0.75, ignore_index=len(CLASSES), nopad=False)

    Stage 2: Albumentations training transforms
      - RandomRotate90(p=0.5)
      - Normalize()

    Args:
        img (PIL.Image.Image or np.ndarray):
            Input RGB image of shape (H, W, 3).
        mask (PIL.Image.Image or np.ndarray):
            Segmentation mask of shape (H, W), where each pixel is a class index.

    Returns:
        tuple:
            img (np.ndarray):
                Augmented image array of shape (H', W', 3),
                dtype float32, values in standardized range.
            mask (np.ndarray):
                Augmented mask array of shape (H', W'),
                dtype int64, with ignored regions marked by len(CLASSES).
    """
    aug1 = Compose([
        RandomHorizontalFlip(prob=0.5),
        RandomVerticalFlip(prob=0.5),
        RandomScale(scale_list=[0.5, 0.75, 1.0, 1.25, 1.5], mode='value'),
        SmartCropV1(crop_size=512, max_ratio=0.75, ignore_index=len(CLASSES), nopad=False)]) 
    img, mask = aug1(img, mask) # Returns cropped patches of images and corresponding masks 
    img, mask = np.array(img), np.array(mask)

    aug2 = get_training_transform()
    out = aug2(image=img.copy(), mask=mask.copy())
    img, mask = out['image'], out['mask']
    return img, mask

def get_val_transform():
    """
    Provides an Albumentations pipeline for validation data preprocessing.

    This transform is applied to each (image, mask) pair during the validation
    phase. It does not perform any geometric augmentation, only normalization
    to match the network’s expected input distribution.

    Returns:
        albu.Compose: A composed Albumentations transform which, when called as
            aug = get_val_transform()
            out = aug(image=img, mask=mask)
        produces a dict with keys:
            - 'image' (np.ndarray): float32 image in 0–1 range, standardized by mean/std.
            - 'mask'  (np.ndarray): integer mask unchanged except cast to array.
    """
    val_transform = [
        # albu.Resize(height=1024, width=1024, interpolation=cv2.INTER_CUBIC),
        albu.Normalize()
    ]
    return albu.Compose(val_transform)

def val_aug(img, mask):
    """
    Applies validation-time preprocessing to an RGB image and its segmentation mask.

    Steps:
      1. Convert inputs to NumPy arrays.
      2. Apply the Albumentations pipeline from get_val_transform():
         - Normalize image to [0,1] and standardize by mean/std.
         - (Optional: resize if uncommented in get_val_transform.)
      3. Return the processed image and mask arrays.

    Args:
        img (PIL.Image.Image or np.ndarray):
            Input RGB image of shape (H, W, 3).
        mask (PIL.Image.Image or np.ndarray):
            Segmentation mask of shape (H, W), with integer class indices.

    Returns:
        tuple:
            img (np.ndarray):
                Preprocessed image array, dtype float32, shape (H, W, 3),
                values scaled and standardized.
            mask (np.ndarray):
                Preprocessed mask array, dtype int64, shape (H, W),
                unchanged except cast to array.
    """
    img, mask = np.array(img), np.array(mask)
    aug = get_val_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask

def get_test_transform():
    """
    Provides an Albumentations pipeline for validation (and testing) data preprocessing.

    This transform is applied to each (image, mask) pair during the validation
    or test phase. It does *not* perform any geometric augmentation, only
    normalization to match the network’s expected input distribution.

    Args:
        None

    Returns:
        albu.Compose: A composed Albumentations transform which, when called as
            aug = get_infer_transform()
            out = aug(image=img, mask=mask)
        produces a dict with keys:
            - 'image' (np.ndarray): float32 image in 0–1 range, standardized by mean/std.
            - 'mask'  (np.ndarray): integer mask unchanged except cast to array.
    """
    infer_transform = [
        # albu.Resize(height=1024, width=1024, interpolation=cv2.INTER_CUBIC),
        albu.Normalize()
    ]
    return albu.Compose(infer_transform)

def test_aug(img, mask):
    """
    Applies inference‐time preprocessing to an RGB image and its segmentation mask.

    Steps:
      1. Convert inputs to NumPy arrays.
      2. Apply the Albumentations pipeline from get_infer_transform():
         - Normalize image to [0,1] and standardize by mean/std.
         - (Optional: resize if uncommented in get_infer_transform.)
      3. Return the processed image and mask arrays.

    Args:
        img (PIL.Image.Image or np.ndarray):
            Input RGB image of shape (H, W, 3).
        mask (PIL.Image.Image or np.ndarray):
            Segmentation mask of shape (H, W), with integer class indices.

    Returns:
        tuple:
            img (np.ndarray):
                Preprocessed image array, dtype float32, shape (H, W, 3),
                values scaled and standardized.
            mask (np.ndarray):
                Preprocessed mask array, dtype int64, shape (H, W),
                unchanged except cast to array.
    """
    img, mask = np.array(img), np.array(mask)
    aug = get_test_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask

def save_masks(masks, output_dir):
    """
    Save a list of torch.LongTensor masks (H×W) to PNG files.

    Args:
        masks (List[torch.Tensor]): predicted masks.
        output_dir (str): directory to write PNGs into.
    """
    os.makedirs(output_dir, exist_ok=True)
    for idx, mask in enumerate(masks):
        # Convert to uint8 numpy array
        mask_np = mask.numpy().astype(np.uint8)
        # Create PIL image (mode 'L' for 8‑bit grayscale)
        img = Image.fromarray(mask_np, mode='L')
        # Save as mask_000.png, mask_001.png, ...
        img.save(os.path.join(output_dir, f"mask_{idx:03d}.png"))

class UrbanDataset(Dataset):
    def __init__(self, data_root, 
                 mode, augmentation,
                 img_dir='images', 
                 mask_dir='masks',
                 img_suffix='.png', 
                 mask_suffix='.png', 
                 mosaic_ratio=0.0,
                 img_size=ORIGIN_IMG_SIZE):
        self.data_root = data_root
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.augmentation = augmentation
        self.mode = mode
        self.mosaic_ratio = mosaic_ratio
        self.img_size = img_size
        self.img_ids = self.get_img_ids(self.data_root, self.img_dir, self.mask_dir) # This is a class method

    # Python dunder method for defining the indexing hook for this class
    def __getitem__(self, index):
        """
        Retrieve and preprocess a sample (image + mask) by index.

        Depending on the random draw and mode, either loads a single
        image/mask pair or constructs a 4‑tile mosaic. Applies the
        configured augmentation function, then converts to PyTorch
        tensors. Mosaics splice four random crops into one image. 
        Only used in training loops to boost generalization.  

        Args:
            index (int):
                Index of the sample to retrieve.

        Returns:
            dict:
                img_id (str): Identifier for the sample (filename without suffix).
                img (torch.FloatTensor):
                    Tensor of shape (3, H, W), dtype float32,
                    image data after augmentation.
                gt_semantic_seg (torch.LongTensor):
                    Tensor of shape (H, W), dtype int64,
                    segmentation mask after augmentation.
        """
        p_ratio = random.random()
        if p_ratio > self.mosaic_ratio or self.mode == 'val' or self.mode == 'test':
            img, mask = self.load_img_and_mask(index)
            if self.augmentation:
                img, mask = self.augmentation(img, mask)
        else:
            img, mask = self.load_mosaic_img_and_mask(index)
            if self.augmentation:
                img, mask = self.augmentation(img, mask)

        img = torch.from_numpy(img).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).long()
        img_id = self.img_ids[index]
        results = dict(img_id=img_id, img=img, gt_semantic_seg=mask)
        return results

    # Python dunder method makes len(dataset) = number of img mask pairs by returning len(img_ids)
    def __len__(self):
        return len(self.img_ids)

    def get_img_ids(self, data_root, img_dir, mask_dir):
        """
        Gather paired image/mask identifiers from the dataset directories.

        Scans the `img_dir` and `mask_dir` under `data_root`, asserts they
        contain the same number of files, then extracts the base filename
        (without extension) for each mask file as the canonical ID.

        Args:
            data_root (str):
                Path to the dataset root containing image and mask subfolders.
            img_dir (str):
                Name of the subdirectory under `data_root` where images reside.
            mask_dir (str):
                Name of the subdirectory under `data_root` where masks reside.

        Returns:
            List[str]:
                A list of identifiers (filenames without extensions) corresponding
                to each mask (and image) file pair. Each index is a dict with three 
                entries: 
                    "img_id": the string ID (e.g. '0001')
                    "img": a torch.FloatTensor of shape (3, H, W)
                    "gt_semantic_seg": a torch.LongTensor of shape (H, W)
        """
        img_filename_list = os.listdir(osp.join(data_root, img_dir))
        mask_filename_list = os.listdir(osp.join(data_root, mask_dir))
        assert len(img_filename_list) == len(mask_filename_list),(
        "There aren't the same amount of images and masks"
        )
        img_ids = [str(id.split('.')[0]) for id in mask_filename_list]
        return img_ids

    def load_img_and_mask(self, index):
        """
        Convert image to RGB and load a paired image and segmentation mask from 
        disk by index.

        Args:
            index (int):
                Index of the sample to load, used to look up the img_id.
        
        Returns:
            tuple:
                img (PIL.Image.Image):
                    RGB image loaded from 
                    `{data_root}/{img_dir}/{img_id}{img_suffix}`, 
                    converted to 3‑channel “RGB” mode.
                mask (PIL.Image.Image):
                    Segmentation mask loaded from 
                    `{data_root}/{mask_dir}/{img_id}{mask_suffix}`, 
                    each pixel is a class index.
        """
        img_id = self.img_ids[index]
        img_name = osp.join(self.data_root, self.img_dir, img_id + self.img_suffix)
        mask_name = osp.join(self.data_root, self.mask_dir, img_id + self.mask_suffix)
        img = Image.open(img_name).convert('RGB')
        mask = Image.open(mask_name)
        return img, mask

    def load_mosaic_img_and_mask(self, index):
        """
        Used in training loop.
        Create a 4‑tile mosaic image+mask by stitching together four random crops.

        Steps:
          1. Pick the primary index plus three random indices.
          2. Load each image/mask pair as PIL.Image and convert to NumPy arrays.
          3. Determine a random splice center within the target size (self.img_size).
          4. Compute crop sizes for each quadrant (A: top‑left, B: top‑right,
             C: bottom‑left, D: bottom‑right).
          5. Apply an Albumentations RandomCrop to each array pair.
          6. Concatenate the four cropped sub‑images into one mosaic:
             ┌─────────┬─────────┐
             │   A     │    B    │
             ├─────────┼─────────┤
             │   C     │    D    │
             └─────────┴─────────┘
          7. Convert the final NumPy mosaic back to PIL.Image for downstream transforms.

        Args:
            index (int): Primary sample index used in the mosaic.

        Returns:
            tuple:
                img (PIL.Image.Image): Mosaic RGB image of shape (H, W, 3).
                mask (PIL.Image.Image): Mosaic segmentation mask of shape (H, W).
        """
        indexes = [index] + [random.randint(0, len(self.img_ids) - 1) for _ in range(3)]
        img_a, mask_a = self.load_img_and_mask(indexes[0])
        img_b, mask_b = self.load_img_and_mask(indexes[1])
        img_c, mask_c = self.load_img_and_mask(indexes[2])
        img_d, mask_d = self.load_img_and_mask(indexes[3])

        img_a, mask_a = np.array(img_a), np.array(mask_a)
        img_b, mask_b = np.array(img_b), np.array(mask_b)
        img_c, mask_c = np.array(img_c), np.array(mask_c)
        img_d, mask_d = np.array(img_d), np.array(mask_d)

        w = self.img_size[1]
        h = self.img_size[0]

        start_x = w // 4
        strat_y = h // 4
        # The coordinates of the splice center
        offset_x = random.randint(start_x, (w - start_x))
        offset_y = random.randint(strat_y, (h - strat_y))

        crop_size_a = (offset_x, offset_y)
        crop_size_b = (w - offset_x, offset_y)
        crop_size_c = (offset_x, h - offset_y)
        crop_size_d = (w - offset_x, h - offset_y)

        random_crop_a = albu.RandomCrop(width=crop_size_a[0], height=crop_size_a[1])
        random_crop_b = albu.RandomCrop(width=crop_size_b[0], height=crop_size_b[1])
        random_crop_c = albu.RandomCrop(width=crop_size_c[0], height=crop_size_c[1])
        random_crop_d = albu.RandomCrop(width=crop_size_d[0], height=crop_size_d[1])

        croped_a = random_crop_a(image=img_a.copy(), mask=mask_a.copy())
        croped_b = random_crop_b(image=img_b.copy(), mask=mask_b.copy())
        croped_c = random_crop_c(image=img_c.copy(), mask=mask_c.copy())
        croped_d = random_crop_d(image=img_d.copy(), mask=mask_d.copy())

        img_crop_a, mask_crop_a = croped_a['image'], croped_a['mask']
        img_crop_b, mask_crop_b = croped_b['image'], croped_b['mask']
        img_crop_c, mask_crop_c = croped_c['image'], croped_c['mask']
        img_crop_d, mask_crop_d = croped_d['image'], croped_d['mask']

        top = np.concatenate((img_crop_a, img_crop_b), axis=1)
        bottom = np.concatenate((img_crop_c, img_crop_d), axis=1)
        img = np.concatenate((top, bottom), axis=0)

        top_mask = np.concatenate((mask_crop_a, mask_crop_b), axis=1)
        bottom_mask = np.concatenate((mask_crop_c, mask_crop_d), axis=1)
        mask = np.concatenate((top_mask, bottom_mask), axis=0)
        mask = np.ascontiguousarray(mask)
        img = np.ascontiguousarray(img)

        img = Image.fromarray(img)
        mask = Image.fromarray(mask)

        return img, mask

def label2rgb(mask):
    """
    Convert a 2D segmentation mask of class indices into a 3‑channel RGB image.
    Helper function for img_writer
    
    Parameters:
        mask (np.ndarray):
            2D array of shape (H, W) with integer class labels in [0..6].

    Returns:
        np.ndarray:
            RGB image of shape (H, W, 3), dtype=uint8. Each class index is
            mapped to a fixed color:
              0 → [255, 255, 255] (white/background)
              1 → [255,   0,   0] (red)
              2 → [255, 255,   0] (yellow)
              3 → [  0,   0, 255] (blue)
              4 → [159, 129, 183] (lavender)
              5 → [  0, 255,   0] (green)
              6 → [255, 195, 128] (peach)
    """
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [255, 255, 255]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [255, 0, 0]
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [255, 255, 0]
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [0, 0, 255]
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [159, 129, 183]
    mask_rgb[np.all(mask_convert == 5, axis=0)] = [0, 255, 0]
    mask_rgb[np.all(mask_convert == 6, axis=0)] = [255, 195, 128]
    return mask_rgb

def img_writer(raw_masks):
    """
    Save a segmentation mask to disk as either a colorized RGB image or a paletted (indexed) PNG.

    Parameters:
        inp (tuple):
            mask (np.ndarray): 2D array of class indices for each pixel.
            output_path (str): Base directory where subfolders 'pre_rgb' and 'pre_p' live.
            mask_name (str): Filename (without extension) under which to save the mask.
            rgb (bool): If True, write an RGB‑colorized PNG in 'pre_rgb'; if False, write an indexed PNG with a palette in 'pre_p'.

    Returns:
        None
    """
    (mask,output_path, mask_name,rgb) = raw_masks
    if rgb:
        mask_RGB_name_png = os.path.join(output_path ,'pre_rgb',mask_name+ '.png')
        mask_rgb_png = label2rgb(mask)
        mask_rgb_png = cv2.cvtColor(mask_rgb_png, cv2.COLOR_RGB2BGR)
        cv2.imwrite(mask_RGB_name_png, mask_rgb_png)
    else:
        mask_P_name_png = os.path.join(output_path ,'pre_p',mask_name+ '.png')
        mask_p_png = mask.astype(np.uint8)
        mask_p_png = Image.fromarray(mask_p_png,'P')
        mask_p_png.putpalette(PALETTE)
        mask_p_png.save(mask_P_name_png)