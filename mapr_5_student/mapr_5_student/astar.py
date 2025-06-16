import rclpy
import time
from mapr_5_student.grid_map import GridMap
import heapq
from torch import nn
import torch
import math
import os

class EnhancedCostMapMLP(nn.Module):
        def __init__(self):
            super(EnhancedCostMapMLP, self).__init__()
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


def load_trained_model(model_path, device='cpu'):
    model = EnhancedCostMapMLP().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


class A_StarPoint:
    def __init__(self, point: tuple[int, int], cost: int):
        self.point = point
        self.cost = cost
    
    def get_cost_with_heuristic(self, heuristicFc):
        return self.cost + heuristicFc(self.point)

class A_StartParent:
    def __init__(self, parent: tuple[int, int], cost: int):
        self.parent = parent
        self.cost = cost

class ASTAR(GridMap):
    def __init__(self):
        super(ASTAR, self).__init__('astar_node')
        self.to_visit = []
        self.parents = {}
        self.g_costs = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = load_trained_model(os.path.join(os.getcwd(), "src/mapr_5_student/mapr_5_student/gradient_model.pth"), device=self.device)

    def heuristics(self, pos: tuple[int, int]):
        cost = self.predict_gradient(pos[0]*0.01, pos[1]*0.01)*1000
        # self.get_logger().info(f"Heuristic cost for position {pos[0]*0.01} {pos[1]*0.01}: gradient: {cost}, path_cost: {abs(pos[0] - self.end[0]) + abs(pos[1] - self.end[1])}")
        return abs(pos[0] - self.end[0]) + abs(pos[1] - self.end[1]) + cost
    

    def visualize_gradient_field(self, resolution=300):
        pass
        # x = np.linspace(0, 3.0, resolution)
        # y = np.linspace(0, 3.0, resolution)
        # X, Y = np.meshgrid(x, y)
        
        # # Tworzenie macierzy na wartości gradientu
        # Z = np.zeros((resolution, resolution))
        
        # # Wypełnianie macierzy wartościami gradientu
        # for i in range(resolution):
        #     for j in range(resolution):
        #         Z[i, j] = self.predict_gradient(X[i, j], Y[i, j])
        
        # # Wyświetlanie obrazka
        # plt.figure(figsize=(10, 10))
        # plt.imshow(Z, extent=[0, 3.0, 0, 3.0], origin='lower', cmap='viridis')
        # plt.colorbar(label='Gradient Value')
        # plt.xlabel('X')
        # plt.ylabel('Y')
        # plt.title('Gradient Field Visualization')
        # plt.show()


    def predict_gradient(self, x, y):
        device = self.device
        
        with torch.no_grad():
            point = torch.tensor([[ x, 3.0 - y]], dtype=torch.float32, device=device)
            return self.model(point).item()

    def check_neighbors(self, current_node: A_StarPoint):
        x, y = current_node.point
        neighbors = [
            (x, y + 1),
            (x - 1, y),
            (x, y - 1),
            (x + 1, y)
        ]
    
        for neighbor in neighbors:
            nx, ny = neighbor
        
            if (0 <= nx < self.map.info.width and 
                0 <= ny < self.map.info.height and 
                self.map.data[self.point_to_index(neighbor)] == 0):
                
                new_cost = current_node.cost + 1  # Koszt przejścia do sąsiada
                
                if neighbor not in self.g_costs or new_cost < self.g_costs[neighbor]:
                    self.g_costs[neighbor] = new_cost
                    f_cost = new_cost + self.heuristics(neighbor)  # f(n) = g(n) + h(n)
                    heapq.heappush(self.to_visit, (f_cost, neighbor))
                    self.parents[neighbor] = A_StartParent(current_node.point, new_cost)

    def check_if_valid(self, a, b, step_size = 0.01):
        # Oblicz wektor od (x0, y0) do (x1, y1)
        x0, y0 = a
        x1, y1 = b
        dx = x1 - x0
        dy = y1 - y0
        
        # Oblicz odległość między punktami
        distance = math.sqrt(dx**2 + dy**2)
        
        if distance == 0:
            print("Punkty są identyczne.")
            return
        
        # Normalizujemy wektor, aby uzyskać kierunek
        dx_normalized = dx / distance
        dy_normalized = dy / distance
        
        self.get_logger().info(f"Sprawdzanie dwóch punktów ({a[0]}, {a[1]}) oraz ({b[0]}, {b[1]})")

        current_distance = 0.0
        while current_distance < distance:
            x = round(x0 + current_distance * dx_normalized)
            y = round(y0 + current_distance * dy_normalized)
            
            if not (0 <= x < self.map.info.width and 0 <= y < self.map.info.height):
                return False
            if self.map.data[self.point_to_index((x, y))] != 50:
                return False
            
            current_distance += step_size
        
        return True

    def get_path(self, last_point: A_StarPoint):
        current_node = last_point.point
        path = [current_node]
        while self.parents[current_node].parent is not None:
            path.append(self.parents[current_node].parent)
            current_node = self.parents[current_node].parent
        path = path[::-1]

        if len(path) <= 2:
            return path

        smoothed_path = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1:
                if self.check_if_valid(path[i], path[j]):
                    break
                j -= 1
            smoothed_path.append(path[j])
            i = j
        return smoothed_path

    def point_to_index(self, point: tuple[int, int]):
        return point[1] * self.map.info.width + point[0]

    def search(self):
        self.visualize_gradient_field()
        start_f_cost = 0 + self.heuristics(self.start)
        heapq.heappush(self.to_visit, (start_f_cost, self.start))
        self.g_costs[self.start] = 0
        self.parents[self.start] = A_StartParent(None, 0)
        while self.to_visit:
            current_f_cost, current_point = heapq.heappop(self.to_visit)
            if current_point == self.end:
                path = self.get_path(A_StarPoint(current_point, self.g_costs[current_point]))
                return self.publish_path(path)
            self.map.data[self.point_to_index(current_point)] = 50
            self.publish_visited(0.001)
            current_node = A_StarPoint(current_point, self.g_costs[current_point])
            self.check_neighbors(current_node)

def main(args=None):
    rclpy.init(args=args)
    astar = ASTAR()
    
    # Oczekiwanie na dane mapy
    while not astar.data_received():
        astar.get_logger().info("Waiting for data...")
        rclpy.spin_once(astar)
        time.sleep(0.5)
    start_time = time.time()
    astar.get_logger().info("Start graph searching!")
    astar.publish_visited()
    time.sleep(1)
    astar.search()
    astar.get_logger().info(f"Elapsed time: {start_time - time.time()}")

if __name__ == '__main__':
    main()