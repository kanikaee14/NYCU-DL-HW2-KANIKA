"""
train_infer.py — Faster R-CNN digit detection for Mac (MPS) or CUDA.
Dataset folder: ./nycu-hw2-data/train, ./nycu-hw2-data/valid, ./nycu-hw2-data/test
Run: python train_infer.py
"""

import os
import re
import json
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models import ResNet50_Weights
from PIL import Image

# ── Device ────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

# ── Config ────────────────────────────────────────────────────────────────────
DATA_ROOT  = "./nycu-hw2-data"
NUM_EPOCHS = 10           # change to 5 for a quick test
BATCH_SIZE = 2            # safe for Mac unified memory
NUM_WORKERS = 0           # 0 is safest on Mac (avoids multiprocessing issues)
CONFIDENCE_THRESHOLD = 0.3
LR = 0.005


# ── Dataset ───────────────────────────────────────────────────────────────────
class DigitDataset(Dataset):
    def __init__(self, data_root, split="train"):
        self.split = split
        self.transform = transforms.Compose([transforms.ToTensor()])

        if split in ("train", "valid"):
            self.image_folder = os.path.join(data_root, split)
            anno_file = "train.json" if split == "train" else "valid.json"
            anno_path = os.path.join(data_root, anno_file)

            with open(anno_path) as f:
                coco = json.load(f)

            self.image_map = {img["id"]: img["file_name"]
                              for img in coco["images"]}
            self.img_ids = list(self.image_map.keys())
            self.boxes = {img_id: [] for img_id in self.img_ids}
            for ann in coco["annotations"]:
                self.boxes[ann["image_id"]].append(ann)
        else:
            self.image_folder = os.path.join(data_root, "test")
            self.test_files = sorted(os.listdir(self.image_folder))

    def __len__(self):
        if self.split == "test":
            return len(self.test_files)
        return len(self.img_ids)

    def __getitem__(self, idx):
        if self.split == "test":
            fname = self.test_files[idx]
            img_path = os.path.join(self.image_folder, fname)
            image = Image.open(img_path).convert("RGB")
            m = re.search(r"\d+", fname)
            img_id = int(m.group()) if m else idx
            return self.transform(image), img_id

        img_id = self.img_ids[idx]
        fname = self.image_map[img_id]
        img_path = os.path.join(self.image_folder, fname)
        image = Image.open(img_path).convert("RGB")
        image_tensor = self.transform(image)

        box_list, label_list = [], []
        for ann in self.boxes[img_id]:
            x, y, w, h = ann["bbox"]
            box_list.append([x, y, x + w, y + h])
            label_list.append(ann["category_id"])

        if box_list:
            t_boxes = torch.tensor(box_list, dtype=torch.float32)
            t_labels = torch.tensor(label_list, dtype=torch.int64)
        else:
            t_boxes = torch.zeros((0, 4), dtype=torch.float32)
            t_labels = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": t_boxes,
            "labels": t_labels,
            "image_id": torch.tensor([img_id]),
        }
        return image_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


# ── Model ─────────────────────────────────────────────────────────────────────
def build_model():
    model = fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=ResNet50_Weights.DEFAULT,
        num_classes=11,   # 10 digits + background
        min_size=400,
        max_size=640,
    )
    return model


# ── Train ─────────────────────────────────────────────────────────────────────
def train(model, loader, optimizer, epoch):
    model.train()
    total = 0.0
    for i, (imgs, targets) in enumerate(loader):
        imgs = [img.to(DEVICE) for img in imgs]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(imgs, targets)
        loss = sum(loss_dict.values())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total += loss.item()
        if (i + 1) % 100 == 0 or (i + 1) == len(loader):
            print(f"  Epoch {epoch} | step {i+1}/{len(loader)} "
                  f"| loss {loss.item():.4f}")
    return total / len(loader)


# ── Inference ─────────────────────────────────────────────────────────────────
def run_inference(model):
    print("\nRunning inference on test set...")
    test_dataset = DigitDataset(DATA_ROOT, split="test")
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=collate_fn,
    )

    model.eval()
    results = []

    with torch.no_grad():
        for imgs, img_ids in test_loader:
            imgs = [img.to(DEVICE) for img in imgs]
            preds = model(imgs)

            boxes  = preds[0]["boxes"].cpu().numpy()
            scores = preds[0]["scores"].cpu().numpy()
            labels = preds[0]["labels"].cpu().numpy()
            img_id = int(img_ids[0])

            for box, score, label in zip(boxes, scores, labels):
                if score < CONFIDENCE_THRESHOLD:
                    continue
                x1, y1, x2, y2 = box
                results.append({
                    "image_id":   img_id,
                    "bbox":       [float(x1), float(y1),
                                   float(x2 - x1), float(y2 - y1)],
                    "score":      float(score),
                    "category_id": int(label),
                })

    # Fallback: if threshold too high, lower it automatically
    if len(results) == 0:
        print("No predictions passed threshold — lowering to 0.05 for fallback")
        with torch.no_grad():
            for imgs, img_ids in test_loader:
                imgs = [img.to(DEVICE) for img in imgs]
                preds = model(imgs)
                boxes  = preds[0]["boxes"].cpu().numpy()
                scores = preds[0]["scores"].cpu().numpy()
                labels = preds[0]["labels"].cpu().numpy()
                img_id = int(img_ids[0])
                if len(scores) > 0:
                    # take top-1 per image
                    i = scores.argmax()
                    x1, y1, x2, y2 = boxes[i]
                    results.append({
                        "image_id":    img_id,
                        "bbox":        [float(x1), float(y1),
                                        float(x2 - x1), float(y2 - y1)],
                        "score":       float(scores[i]),
                        "category_id": int(labels[i]),
                    })

    with open("pred.json", "w") as f:
        json.dump(results, f, indent=2)

    from collections import Counter
    covered = len(set(r["image_id"] for r in results))
    cats = dict(sorted(Counter(r["category_id"] for r in results).items()))
    print(f"Saved {len(results)} predictions to pred.json")
    print(f"Images covered: {covered} / {len(test_dataset)}")
    print(f"Category dist : {cats}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # ── Datasets ──────────────────────────────────────────────────────────────
    train_dataset = DigitDataset(DATA_ROOT, split="train")
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, collate_fn=collate_fn,
    )
    print(f"Train images: {len(train_dataset)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model().to(DEVICE)

    # Backbone LR 10x lower than heads (standard practice)
    backbone_params = list(model.backbone.parameters())
    backbone_ids = set(id(p) for p in backbone_params)
    other_params = [p for p in model.parameters() if id(p) not in backbone_ids]

    optimizer = torch.optim.SGD(
        [
            {"params": backbone_params, "lr": LR * 0.1},
            {"params": other_params,    "lr": LR},
        ],
        momentum=0.9, weight_decay=0.0005,
    )
    # Drop LR at epoch 7 for 10-epoch run
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[7, 9], gamma=0.1
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_loss = float("inf")
    print(f"\nStarting training for {NUM_EPOCHS} epochs...")

    for epoch in range(1, NUM_EPOCHS + 1):
        avg_loss = train(model, train_loader, optimizer, epoch)
        scheduler.step()
        print(f"Epoch {epoch}/{NUM_EPOCHS} | avg loss: {avg_loss:.4f} "
              f"| LR: {scheduler.get_last_lr()[1]:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  --> Saved best model")

    print(f"\nTraining done. Best loss: {best_loss:.4f}")

    # ── Inference ─────────────────────────────────────────────────────────────
    print("\nLoading best model for inference...")
    model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
    run_inference(model)


if __name__ == "__main__":
    main()
