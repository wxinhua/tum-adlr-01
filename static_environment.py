import numpy as np
import matplotlib.pyplot as plt

class MapGenerator:
    def __init__(self, height, width, obstacles_density):
        self.height = height
        self.width = width
        self.obstacles_density = obstacles_density
        self.static_obstacles = self.generate_obstacles()

    def generate_obstacles(self):
        return np.random.rand(self.height, self.width) < self.obstacles_density

    def ensure_start_goal_clear(self, start, goal):
        self.static_obstacles[start] = False
        self.static_obstacles[goal] = False

    def visualize(self, start, goal, path=None):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_xticks(np.arange(0, self.width, 10))
        ax.set_yticks(np.arange(0, self.height, 10))
        ax.grid(True)

        for i in range(self.height):
            for j in range(self.width):
                if self.static_obstacles[i, j]:
                    ax.add_patch(plt.Rectangle((j, self.height - i - 1), 1, 1, color='black'))

        start_patch = plt.Circle((start[1] + 0.5, self.height - start[0] - 0.5), 0.4, color='blue')
        goal_patch = plt.Circle((goal[1] + 0.5, self.height - goal[0] - 0.5), 0.4, color='red')
        ax.add_patch(start_patch)
        ax.add_patch(goal_patch)

        if path:
            for (x, y) in path:
                ax.add_patch(plt.Rectangle((y, self.height - x - 1), 1, 1, color='green', alpha=0.5))

        plt.show()



