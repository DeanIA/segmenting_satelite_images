import os
import torch
from segment_anything import sam_model_registry, get_sam_label
from PIL import Image
from tqdm import tqdm
import numpy as np


@torch.no_grad() # Run without training weights 

def get_sam_info(
    image,
    box_nms=0.7,
    min_mask_region_area=100,
    pred_iou_thresh=0.88,
    stability_score_thresh=0.92,
    *args, **kwargs):
    """
    Description:
        Runs the SAM-HQ model on the input image to generate segmentation outputs.

    Inputs:
        image (PIL.Image): The input image to segment.
        box_nms (float): Box NMS threshold for post-processing.
        min_mask_region_area (int): Minimum area for mask regions.
        pred_iou_thresh (float): IoU threshold for predictions.
        stability_score_thresh (float): Stability score threshold.
        *args, **kwargs: Additional arguments for get_sam_label.

    Outputs:
        label (PIL.Image): Segmentation label image (mask).
        blend (PIL.Image): Visualization image (mask blended with original).
        label_P (np.ndarray): Numpy array with per-pixel probabilities or labels.
    """
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        label, blend, label_P = get_sam_label(
            sam, image,
            box_nms=box_nms,
            min_mask_region_area=min_mask_region_area,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            *args, **kwargs)
        return label, blend, label_P

def process_data_img(image_file, img_name):
    """
    Description:
        Loads an image, runs SAM-HQ inference, and saves the resulting label, blend, and label_P outputs.

    Inputs:
        image_file (str): Directory containing the images.
        img_name (str): Filename of the image to process.

    Outputs:
        Saves three files to disk:
            - label: PNG mask image in lbl_dir
            - blend: Blended visualization PNG in bld_dir
            - label_P: Numpy array (.npy) in lblp_dir
    """
    img_name = img_name[:-4]+'.png'
    
    img = Image.open(os.path.join(image_file,img_name))
    label,blend,label_P = get_sam_info(img,box_nms=box_nms,min_mask_region_area =min_region,pred_iou_thresh=pred_iou_thresh,stability_score_thresh=stability_score_thresh)
    label.save(os.path.join(lbl_dir,img_name))
    np.save(os.path.join(lblp_dir,img_name.replace('.png','.npy')),label_P)
    blend.save(os.path.join(bld_dir,img_name))


def make_dirs():
    """
    Description:
        Creates output directories for saving SAM-HQ results if they do not exist.

    Inputs:
        None

    Outputs:
        Creates directories: sam_img_dir, lbl_dir, bld_dir, lblp_dir
    """
    os.makedirs(sam_img_dir,exist_ok=True)
    os.makedirs(lbl_dir,exist_ok=True)
    os.makedirs(bld_dir,exist_ok=True)
    os.makedirs(lblp_dir,exist_ok=True)


# For SAMHQ
# min_region = 500 # Minimum area (in pixels) for a mask region to be kept. 
# box_nms = 0.6 # Threshold for removing overlapping predicted boxes.
# pred_iou_thresh = 0.85 # Minimum predicted Intersection-over-Union. Masks with lower IoU scores are filtered out.
# stability_score_thresh = 0.85 # Minimum stability score threshold for keeping a mask.

# For SAM
min_region = 500
box_nms = 0.6
pred_iou_thresh = 0.85
stability_score_thresh = 0.85

# For SAMHQ
#sam_checkpoint = {'vit_h':"SAM/ckpt/sam_hq_vit_h.pth"}

# For SAM
sam_checkpoint = {'vit_h':"SAM/ckpt/sam_vit_h_4b8939.pth"}

model_type = "vit_h"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint[model_type])
sam.to(device=device)

image_dir = '/home/jupyter-dai7591/abc_sam_sessrs/geoseg/data/Urban/val/images'
sam_img_dir = '/home/jupyter-dai7591/abc_sam_sessrs/geoseg/data/Urban/sam_label/samhq'


lbl_dir = os.path.join(sam_img_dir,'label') # label: Segmentation mask image
bld_dir = os.path.join(sam_img_dir,'blend') # blend: Visualization (mask + original image)
lblp_dir = os.path.join(sam_img_dir,'label_p') # label_P: Numpy array with per-pixel labels 
make_dirs()

val_lines = os.listdir(image_dir)
for val_line in tqdm(val_lines):
    process_data_img(image_dir,val_line)
    