import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import os

class PoseSequenceDataset(Dataset):
    def __init__(self, pose_dir, label_file, sequence_len=30, label_map=None):
        self.sequence_len = sequence_len
        self.samples = []
        self.labels = []
        self.label_map = label_map or {}

        # Load label mappings
        label_df = pd.read_csv(label_file)
        if not self.label_map:
            unique_labels = label_df['label'].unique()
            self.label_map = {label: idx for idx, label in enumerate(unique_labels)}

        # Go through each labeled file
        for _, row in label_df.iterrows():
            file_path = os.path.join(pose_dir, row['filename'])
            if not os.path.isfile(file_path):
                continue  # skip missing files

            data = np.load(file_path)
            label = self.label_map[row['label']]
            for i in range(len(data) - sequence_len):
                self.samples.append(data[i:i + sequence_len])
                self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = torch.tensor(self.samples[idx], dtype=torch.float32)  # [seq_len, 99]
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y
