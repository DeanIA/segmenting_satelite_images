# SESSRS

Work in progress. Refactoring [this study](geoseg_train_graphs/graphs) to create a workflow for analyzing changes in satellite images. The workflow leverages the powerful Segment Anything model’s ability to create sharp masks over images, improving semantic segmentation and enabling accurate, class-specific masks of satellite images.

## To Do List

### ✅ Done
- [Train semantic segmentation model](geoseg_train_graphs/graphs)
- [Run Inference on Test Set](geoseg/geoseg.ipynb) 
- [Run Inference with SAM model](SAM/sam_batch_inference.ipynb)

### ⬜ Not Done
- Run SESSRS comparison algorithm

---

### Next Step
- Feed YOLO bounding boxes into SAM for instance segmentation of buildings.