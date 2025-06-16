import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors

# Definicja modelu
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

# Wczytaj model
device = torch.device('cpu')
model = GradientModel().to(device)
model.load_state_dict(torch.load('gradient_model.pth', map_location=device))
model.eval()
print("Model wczytany.")

# Generuj mapę gradientu 300x300
n_points = 300
x = np.linspace(0, 3, n_points)
y = np.linspace(0, 3, n_points)
X, Y = np.meshgrid(x, y)

# Przygotuj punkty do predykcji
points = np.column_stack((X.flatten(), Y.flatten()))
points_tensor = torch.tensor(points, dtype=torch.float32, device=device)

# Wykonaj predykcję w batczach
print("Generuję mapę gradientu...")
batch_size = 5000
predictions = []

with torch.no_grad():
    for i in range(0, len(points_tensor), batch_size):
        batch = points_tensor[i:i + batch_size]
        pred = model(batch).cpu().numpy()
        predictions.append(pred)

# Złącz wyniki i przekształć do siatki 2D
gradient_map = np.concatenate(predictions).reshape(n_points, n_points)
print("Gotowe.")

# Stwórz niebiesko-zieloną paletę kolorów
# Niebieskie (brak gradientu) -> Zielone (duży gradient)
blue_to_green = colors.LinearSegmentedColormap.from_list(
    'blue_to_green', 
    [(0, 'darkblue'), (0.3, 'royalblue'), (0.6, 'lightgreen'), (1.0, 'darkgreen')], 
    N=256
)

# Wyświetl mapę gradientu z nową kolorystyką
plt.figure(figsize=(10, 8))
plt.imshow(gradient_map, extent=[0, 3, 0, 3], origin='lower', cmap=blue_to_green)
plt.colorbar(label='Wartość gradientu')
plt.title('Mapa gradientu (zielony = duży gradient, niebieski = brak gradientu)')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.show()