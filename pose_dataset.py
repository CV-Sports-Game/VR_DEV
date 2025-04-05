import numpy as np
import torch
from torch.utils.data import Dataset

class PoseSequenceDataset(Dataset):
    def __init__(self, pose_file, sequence_len=30, num_classes=3):
        self.data = np.load(pose_file)
        self.sequence_len = sequence_len
        self.samples = []
        for i in range(len(self.data) - sequence_len):
            self.samples.append(self.data[i:i+sequence_len])
        
        # Dummy labels: Replace with your own labeling logic
        self.labels = np.random.randint(0, num_classes, size=len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = torch.tensor(self.samples[idx], dtype=torch.float32)  # [sequence_len, 99]
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y
