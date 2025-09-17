import os
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import pandas as pd
from pose_dataset import PoseSequenceDataset
from pose_transformer import PoseTransformer

# --- New: ImagePoseDataset for image classification ---
class ImagePoseDataset(Dataset):
    def __init__(self, images_dir, label_file, transform=None, label_map=None):
        self.images_dir = images_dir
        self.data = pd.read_csv(label_file)
        self.transform = transform or transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
        self.label_map = label_map or {label: idx for idx, label in enumerate(self.data['label'].unique())}
        self.samples = self.data[['filename', 'label']].values.tolist()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img_full_path = os.path.join(self.images_dir, img_path)
        image = Image.open(img_full_path).convert('RGB')
        image = self.transform(image)
        label_idx = self.label_map[label]
        return image, label_idx

# --- New: Simple CNN for image classification ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def main():
    """Interactive training entrypoint. Only runs when this file is executed directly."""
    mode = input("Select mode: (1) Pose Sequence, (2) Image Classification: ").strip()

    if mode == '1':
        # --- Pose Sequence Training (existing logic) ---
        sequence_len = 30
        pose_dir = "pose_data"
        label_file = os.path.join(pose_dir, "labels.csv")
        dataset = PoseSequenceDataset(pose_dir, label_file, sequence_len)
        print("Label mapping:", dataset.label_map)
        num_classes = len(dataset.label_map)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        model = PoseTransformer(input_dim=99, model_dim=128, num_classes=num_classes, seq_len=sequence_len)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        print("Training on pose sequences...")
        for epoch in range(50):
            total_loss = 0
            model.train()
            for x_batch, y_batch in loader:
                optimizer.zero_grad()
                output = model(x_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch+1}/50, Loss: {avg_loss:.4f}")
        torch.save(model.state_dict(), "pose_model.pth")
        print("Model trained and saved as 'pose_model.pth'")
        print("Label Mapping:", dataset.label_map)
        # Evaluation
        model.eval()
        with torch.no_grad():
            for i in range(10):
                x, y_true = dataset[i]
                x = x.unsqueeze(0)
                y_pred = model(x)
                predicted_label = torch.argmax(y_pred, dim=1).item()
                reverse_map = {v: k for k, v in dataset.label_map.items()}
                print(f"Sample {i}: True = {reverse_map[y_true.item()]}, Predicted = {reverse_map[predicted_label]}")

    elif mode == '2':
        # --- Image Classification Training ---
        images_dir = "images"
        label_file = os.path.join(images_dir, "image_labels.csv")
        dataset = ImagePoseDataset(images_dir, label_file)
        print("Label mapping:", dataset.label_map)
        num_classes = len(dataset.label_map)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        model = SimpleCNN(num_classes=num_classes)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        print("Training on images...")
        for epoch in range(20):
            total_loss = 0
            model.train()
            for x_batch, y_batch in loader:
                optimizer.zero_grad()
                output = model(x_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch+1}/20, Loss: {avg_loss:.4f}")
        torch.save(model.state_dict(), "image_model.pth")
        print("Image model trained and saved as 'image_model.pth'")
        print("Label Mapping:", dataset.label_map)
        # Evaluation
        model.eval()
        with torch.no_grad():
            for i in range(10):
                x, y_true = dataset[i]
                x = x.unsqueeze(0)
                y_pred = model(x)
                predicted_label = torch.argmax(y_pred, dim=1).item()
                reverse_map = {v: k for k, v in dataset.label_map.items()}
                print(f"Sample {i}: True = {reverse_map[y_true]}, Predicted = {reverse_map[predicted_label]}")
    else:
        print("Invalid mode selected. Exiting.")


if __name__ == "__main__":
    main()


