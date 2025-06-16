import torch
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
import os

# Wczytanie mapy
map_path = "map_data/map.pgm"
print("Wczytuję mapę...")
map_img = np.array(Image.open(map_path))
binary_map = (map_img < 128).astype(np.float32)

original_height, original_width = binary_map.shape
target_size = 300
resized_map = np.zeros((target_size, target_size), dtype=np.float32)

scale_y = target_size / original_height
scale_x = target_size / original_width

for i in range(target_size):
    for j in range(target_size):
        orig_i = min(int(i / scale_y), original_height - 1)
        orig_j = min(int(j / scale_x), original_width - 1)
        resized_map[i, j] = binary_map[orig_i, orig_j]

distance_map = distance_transform_edt(1 - resized_map)
max_distance = np.max(distance_map)
normalized_distance = distance_map / max_distance

decay_factor = 2.0
power_factor = 0.6
gradient_map = np.exp(-decay_factor * (normalized_distance ** power_factor))
gradient_map = np.flipud(gradient_map)

X = []
y = []

x_coords = np.linspace(0, 3, target_size)
y_coords = np.linspace(0, 3, target_size)

for i in range(target_size):
    y_val = y_coords[i]
    for j in range(target_size):
        x_val = x_coords[j]
        X.append([x_val, y_val])
        y.append([gradient_map[i, j]])

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

class GradientModel(torch.nn.Module):
    def __init__(self):
        super(GradientModel, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(2, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
            torch.nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.model(x)

print("Rozpoczynam trening modelu...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Używane urządzenie: {device}")

X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

model = GradientModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

batch_size = 4096
n_samples = X.shape[0]
n_batches = n_samples // batch_size
epochs = 1000

best_loss = float('inf')
patience = 50
patience_counter = 0

for epoch in range(epochs):
    indices = torch.randperm(n_samples)
    X_shuffled = X_tensor[indices]
    y_shuffled = y_tensor[indices]
    
    model.train()
    epoch_loss = 0.0
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i+1) * batch_size, n_samples)
        
        X_batch = X_shuffled[start_idx:end_idx]
        y_batch = y_shuffled[start_idx:end_idx]
        
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item() * (end_idx - start_idx)
    
    epoch_loss /= n_samples
    
    print(f"Epoka {epoch+1}/{epochs}, Strata: {epoch_loss:.6f}")
    
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), 'gradient_model.pth')
        patience_counter = 0
        print(f"  Zapisano model (strata: {best_loss:.6f})")
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print(f"Early stopping po {epoch+1} epokach")
        break
    
    if epoch_loss < 1e-4:
        print(f"Osiągnięto zadowalającą dokładność.")
        break

print("Trening zakończony!")