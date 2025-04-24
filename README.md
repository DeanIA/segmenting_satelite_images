## Data & models 
[Processed urban dataset from lovaDA](https://pan.baidu.com/s/1pJhxwH_Rp-YdNkH4hnjnyw?pwd=SERS) <br>
[GeoSeg model weights](https://pan.baidu.com/s/1p6t02G0dgerQX3Vyo0gr-g?pwd=sers)

___
## ENV set ups
*GeoSeg (Semantic Segmentation Model)*

```bash
conda create -n env_SESSRS_GeoSeg python=3.8
conda activate env_SESSRS_GeoSeg
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

*SAM & SESSRS*

```bash
conda create -n env_SESSRS_SAM python=3.9
conda activate env_SESSRS_SAM
pip3 install torch==1.13.1 torchvision==0.14.1 --extra-index-url https://download.pytorch.org/whl/cu113
conda install -c "nvidia/label/cuda-11.7.1" cuda-toolkit ## For running detectron on nvidia 
python -m pip install 'git+https://github.com/MaureenZOU/detectron2-xyz.git'
pip install git+https://github.com/cocodataset/panopticapi.git
python -m pip install -r requirements.txt
```

---

## Summary of repo
Quick summary of what all the files in this repo are doing. Each file will have more detailed annotation in the code. 

### Data
**data/Urban** <br>
Holds the images and masks for groundtruth comparision and optimization.

**data/mask_preprocess** <br>
converts the per-pixel class index saved as a .png into two files:
    1. Per pixel class index remapped to 0 indice
    2. RGB channel visualization of classes on image according to pallete settings

### GeoSeg
**GeoSeg/config/Urban/abcnet.py** <br>
Parameters for ABCNet model loaded into GeoSeg > train_supervision.

**GeoSeg/transform.py** <br>
Holds the generic functions for image augmentation.

**GeoSeg/Urban_dataset.py** <br>
Implements the image augmentation functions on the urban dataset. 

**GeoSeg/train_supervision.py** <br>
Loads the ABCNet semantic segmentation model with parameters defined in the .config file.

**GeoSeg/test.py** <br>
Runs inference on the datasets with the model loaded into the train_supervision outputs the masks as pngs into predetermined directories. 

### SAM
**SAM/ckpt/sam_vit_h_4b8939.pth** <br>
Holds the checkpoint of the SAM model for creating none semantic masks of dataset 

**SAM/segment_anything/automatic_mask_generator.py** <br>
The “prompt‐free” mask generator that sweeps a point grid over the image, prunes low‐quality or duplicate masks, and returns either binary masks or COCO‐style RLEs for every detected object region.

**SAM/segment_anything/build_sam.py** <br>
Builds the SAM model for inference with parameters from the .pth checkpoint file. 

**SAM/tasks/generate_mask_auto.py** <br>
Holds functions for processing the images and labels to match the SAM model and run inference on them. 

**SAM/tasks/generate_img_sam** <br>
Runs the functions in generate_mask_auto to get masks and them writes the mask images into a directory. 

**SAM/tasks/generate_json_sam** <br>
Produce per‐image COCO-style annotations JSONs with RLE segmentation fields.

___

## Source
All code taken from [this repo](https://github.com/qycools/SESSRS)
___