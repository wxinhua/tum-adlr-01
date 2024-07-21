import numpy as np
import random
from visualization import MapGenerator
from Astar import AStar
import pickle

# generate map with dynamic obstacles
class DynamicObstacles:
    def __init__(self, static_obstacles, dynamic_obstacle_density, algorithm):
        self.static_obstacles = static_obstacles
        self.height, self.width = static_obstacles.shape
        self.num_dynamic_obstacles = int(dynamic_obstacle_density * self.height * self.width)
        self.algorithm = algorithm
        self.dynamic_obstacles = self.initialize_dynamic_obstacles(self.num_dynamic_obstacles)
        self.paths = self.calculate_paths()
        self.current_positions = [obs[0] for obs in self.dynamic_obstacles]
        self.directions = [1] * self.num_dynamic_obstacles  # 1:positive direction along path，-1:negative direction

    def initialize_dynamic_obstacles(self, num_dynamic_obstacles):
        dynamic_obstacles = []
        while len(dynamic_obstacles) < num_dynamic_obstacles:
            start = (np.random.randint(self.height), np.random.randint(self.width))
            goal = (np.random.randint(self.height), np.random.randint(self.width))
            if not self.static_obstacles[start] and not self.static_obstacles[goal] and start != goal:
                dynamic_obstacles.append((start, goal))
        return dynamic_obstacles

    def reset(self):
        self.obstacles = self.initialize_dynamic_obstacles(self.num_dynamic_obstacles)

    def calculate_paths(self):
        paths = []
        for start, goal in self.dynamic_obstacles:
            astar = self.algorithm(self.height, self.width, start, goal, self.static_obstacles)
            path = astar.a_star_search()
            paths.append(path)
        return paths

    def update_positions(self):
        new_positions = []
        for i, (current_position, path) in enumerate(zip(self.current_positions, self.paths)):
            if not path:
                new_positions.append(current_position)
                continue

            next_index = path.index(current_position) + self.directions[i]
            # 确保 next_index 在有效范围内
            if next_index >= len(path) or next_index < 0:
                self.directions[i] = -self.directions[i]
                next_index = path.index(current_position) + self.directions[i]

            # 再次检查修正后的 next_index 是否有效
            if 0 <= next_index < len(path):
                next_position = path[next_index]
                if next_position not in new_positions:
                    new_positions.append(next_position)
                    self.current_positions[i] = next_position
                else:
                    if random.random() < 0.9:
                        new_positions.append(current_position)
                    else:
                        self.directions[i] = -self.directions[i]
                        next_index = path.index(current_position) + self.directions[i]
                        # 额外的边界检查
                        if 0 <= next_index < len(path):
                            next_position = path[next_index]
                            self.current_positions[i] = next_position
                            new_positions.append(next_position)
                        else:
                            # 如果反向索引依然越界，则保留当前位置
                            new_positions.append(current_position)
            else:
                # 如果反向索引依然越界，则保留当前位置
                new_positions.append(current_position)

        self.current_positions = new_positions

    def get_positions(self):
        return self.current_positions

# # load static map and global path
# with open('maps_and_paths.pkl', 'rb') as f:
#     maps_and_paths = pickle.load(f)

# # demo: visualize one map with dynamic obstacles
# static_obstacles, global_path = maps_and_paths[0]
# start = (0, 0)
# goal = (99, 99)
# dynamic_obstacle_density = 0.05
# dynamic_obstacles = DynamicObstacles(static_obstacles, dynamic_obstacle_density, AStar)
# MapGenerator.visualize_with_dynamic_obstacles(static_obstacles, start, goal, global_path, dynamic_obstacles) 