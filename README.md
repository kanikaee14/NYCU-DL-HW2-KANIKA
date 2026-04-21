# NYCU Visual Recognition using Deep Learning 2026 Spring - HW2

- **Student ID**: 314540012
- **Name**: Kanika

---

## Introduction

This project implements **digit detection** using the **DETR (Detection Transformer)** model with a **ResNet-50 backbone**, as required by Homework 2 of the Visual Recognition using Deep Learning course (Spring 2026) at NYCU.

The task is to detect all digits in an RGB image and predict both the class label (digits 0–9) and the bounding box for each digit. The dataset follows COCO format with 30,062 training images, 3,340 validation images, and 13,068 test images.

**Key design choices:**
- DETR with ResNet-50 backbone (pretrained on ImageNet)
- Hungarian bipartite matching loss (no NMS required)
- Differential learning rates: backbone at 1e-5, Transformer heads at 1e-4
- Combined L1 + GIoU bounding box regression loss
- Scale jitter augmentation for robustness to digit size variation
- Best training loss achieved: **0.2145**

---

## Environment Setup

### Requirements

- Python 3.9 or higher
- PyTorch (with CUDA )
- torchvision

### Install dependencies

```bash
pip install -r requirements.txt
```

### requirements.txt

```
torch>=2.0.0
torchvision>=0.15.0
Pillow>=9.0.0
numpy>=1.23.0
```

### Dataset structure

Download the dataset and place it as follows:

```
nycu-hw2-data/
├── train/          # training images
├── valid/          # validation images
├── test/           # test images
├── train.json      # COCO-format annotations
└── valid.json      # COCO-format annotations
```

---

## Usage

### Training

```bash
python train_infer.py
```

This will:
1. Train the DETR model for 10 epochs on the training set
2. Save the best model checkpoint as `best_model.pth`
3. Automatically run inference on the test set after training
4. Save predictions to `pred.json`

You can adjust key settings at the top of `train_infer.py`:

| Variable | Default | Description |
|---|---|---|
| `NUM_EPOCHS` | 10 | Number of training epochs |
| `BATCH_SIZE` | 2 | Batch size (lower if OOM) |
| `LR` | 1e-4 | Base learning rate (Transformer) |
| `CONFIDENCE_THRESHOLD` | 0.3 | Min score for test predictions |

### Inference only

If you already have a trained `best_model.pth`, you can run inference directly:

```bash
python inference.py
```

Or call `run_inference()` from within `train_infer.py` after loading the checkpoint manually.

### Submission

The output file `pred.json` is in COCO results format:

```json
[
  {
    "image_id": 1,
    "bbox": [x_min, y_min, width, height],
    "score": 0.95,
    "category_id": 3
  },
  ...
]
```


## Performance Snapshot
![Project Snapshot](https://github.com/kanikaee14/NYCU-DL-HW2-KANIKA/blob/main/snapshot.png)



## References

- N. Carion et al. *End-to-End Object Detection with Transformers*. ECCV 2020.
- K. He et al. *Deep Residual Learning for Image Recognition*. CVPR 2016.
- T.-Y. Lin et al. *Microsoft COCO: Common Objects in Context*. ECCV 2014.
- PyTorch torchvision model zoo: https://github.com/pytorch/vision
- Facebook Research DETR: https://github.com/facebookresearch/detr
