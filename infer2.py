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
NUM_WORKERS = 0

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

            self.image_map = {img["id"]: img["file_name"] for img in coco["images"]}
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


def build_model():
    # Keep the EXACT same resolution used during training
    model = fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=ResNet50_Weights.DEFAULT,
        num_classes=11,
        min_size=400,
        max_size=640,
    )
    return model


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
                # Keep everything the model gives us (internal thresh is already 0.05)
                # Do NOT add a high threshold here.
                x1, y1, x2, y2 = box
                results.append({
                    "image_id":    img_id,
                    "bbox":        [float(x1), float(y1),
                                    float(x2 - x1), float(y2 - y1)],
                    "score":       float(score),
                    "category_id": int(label),
                })

    with open("pred.json", "w") as f:
        json.dump(results, f, indent=2)

    from collections import Counter
    covered = len(set(r["image_id"] for r in results))
    cats = dict(sorted(Counter(r["category_id"] for r in results).items()))
    print(f"Saved {len(results)} predictions to pred.json")
    print(f"Images covered: {covered} / {len(test_dataset)}")
    print(f"Category dist : {cats}")


if __name__ == "__main__":
    model = build_model().to(DEVICE)

    # Allow more detections per image (default is 100; raise if images have many digits)
    model.roi_heads.detections_per_img = 300

    model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
    run_inference(model)
