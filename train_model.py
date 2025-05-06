import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from pose_dataset import PoseSequenceDataset
from pose_transformer import PoseTransformer

# Params
sequence_len = 10
batch_size = 32
num_classes = 3
epochs = 10

pose_dir = "pose_data"
label_file = os.path.join(pose_dir, "labels.csv")
sequence_len = 30
num_classes = 3  # Update based on your real classes

dataset = PoseSequenceDataset(pose_dir, label_file, sequence_len)
print("Label mapping:", dataset.label_map)

num_classes = len(dataset.label_map)  # auto-detect number of classes
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = PoseTransformer(input_dim=99, model_dim=128, num_classes=num_classes, seq_len=sequence_len)



# Load dataset
#dataset = PoseSequenceDataset("pose_data/pose_sequences.npy", sequence_len, num_classes)
#loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model
#model = PoseTransformer(input_dim=99, model_dim=128, num_classes=num_classes, seq_len=sequence_len)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(epochs):
    total_loss = 0
    model.train()
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        output = model(x_batch)  # [batch, num_classes]
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# Save model
torch.save(model.state_dict(), "pose_model.pth")
print("Model trained and saved as 'pose_model.pth'")

print("Label Mapping:", dataset.label_map)

# Evaluation: Check predictions on a few samples
model.eval()
with torch.no_grad():
    for i in range(10):  # Check 10 samples
        x, y_true = dataset[i]
        x = x.unsqueeze(0)  # Add batch dim
        y_pred = model(x)
        predicted_label = torch.argmax(y_pred, dim=1).item()

        # Reverse lookup for readable label
        reverse_map = {v: k for k, v in dataset.label_map.items()}
        print(f"Sample {i}: True = {reverse_map[y_true.item()]}, Predicted = {reverse_map[predicted_label]}")


