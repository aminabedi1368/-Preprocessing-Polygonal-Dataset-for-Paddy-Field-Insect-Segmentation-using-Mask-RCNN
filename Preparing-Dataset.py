import os
import json
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import cv2
import torch.nn as nn
import torch.optim as optim
from torchvision.models.detection import maskrcnn_resnet50_fpn

# تعریف کلاس دیتاست
class PaddyPestDataset(Dataset):
    def __init__(self, image_dir, annotation_file):
        self.image_dir = image_dir
        with open(annotation_file) as f:
            self.annotations = json.load(f)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        img_path = os.path.join(self.image_dir, annotation["file_name"])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        masks, bboxes, labels = [], [], []
        
        for obj in annotation["annotations"]:
            polygon = np.array(obj["segmentation"][0]).reshape(-1, 2)
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            cv2.fillPoly(mask, [polygon.astype(np.int32)], 255)
            
            masks.append(mask)
            x_min, y_min = polygon.min(axis=0)
            x_max, y_max = polygon.max(axis=0)
            bboxes.append([x_min, y_min, x_max, y_max])
            labels.append(1)  # کلاس ثابت (1) برای حشرات

        # تبدیل به PyTorch Tensor
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        masks = torch.tensor(np.stack(masks), dtype=torch.uint8)
        
        target = {"boxes": bboxes, "labels": labels, "masks": masks}
        return image, target

# تابع بارگذاری دیتاست و تقسیم داده‌ها
def load_dataset(image_dir, annotation_file, train_ratio=0.6, batch_size=4):
    dataset = PaddyPestDataset(image_dir, annotation_file)
    
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
    
    return train_loader, test_loader

# تابع آموزش مدل
def train_model(train_loader, output_dir, num_epochs=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = maskrcnn_resnet50_fpn(weights="COCO_V1").to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.000001)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in tgt.items()} for tgt in targets]
            
            optimizer.zero_grad()
            loss_dict = model(images, targets)
            total_loss = sum(loss for loss in loss_dict.values())
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "mask_rcnn.pth"))
    print("Training complete! Model saved.")

if __name__ == "__main__":
    image_dir = "./paddy-with-pests/"
    annotation_file = "annotations.json"
    output_dir = "./output/"
    
    train_loader, test_loader = load_dataset(image_dir, annotation_file)
    train_model(train_loader, output_dir)

